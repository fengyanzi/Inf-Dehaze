"""Top-level InfDehaze model with training and memory-efficient inference."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange

try:
    import yaml

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

from inf_dehaze.backbone.bottleneck import INFBottleneck
from inf_dehaze.backbone.swin_backbone import InfDecoder, InfEncoder

__all__ = ["InfDehaze"]


def _load_yaml(path: Union[str, Path]) -> dict:
    if not _YAML_AVAILABLE:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class AbstractModel(nn.Module):
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight.data)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class _AsyncCPUTransfer:
    """Overlap CPU transfer with GPU compute via a background thread."""

    def __init__(self):
        self._result: Optional[torch.Tensor] = None
        self._thread: Optional[threading.Thread] = None

    def send(self, gpu_tensor: torch.Tensor) -> None:
        def _copy(t: torch.Tensor):
            self._result = t.cpu()

        self._result = None
        self._thread = threading.Thread(target=_copy, args=(gpu_tensor.clone(),), daemon=True)
        self._thread.start()

    def recv(self) -> torch.Tensor:
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        assert self._result is not None, "recv() called before send()"
        return self._result


class InfDehaze(AbstractModel):
    """End-to-end ultra-high-resolution image dehazing model."""

    def __init__(
        self,
        base_dim: int = 96,
        patch_size: int = 2,
        crop_size: int = 256,
        channels_last: bool = True,
        encoder_cfg: Optional[dict] = None,
        bottleneck_cfg: Optional[dict] = None,
        decoder_cfg: Optional[dict] = None,
        inference_batch_size: int = 4,
        use_residual_cache: bool = True,
        async_io: bool = True,
    ):
        super().__init__()
        encoder_cfg = encoder_cfg or {}
        bottleneck_cfg = bottleneck_cfg or {}
        decoder_cfg = decoder_cfg or {}

        self.channels_last = channels_last
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.inference_batch_size = inference_batch_size
        self.use_residual_cache = use_residual_cache
        self.async_io = async_io

        enc_depths = encoder_cfg.get("depths", [2, 3, 4, 2])
        enc_heads = encoder_cfg.get("num_heads", [2, 4, 8, 8])
        enc_window = encoder_cfg.get("window_size", 8)
        enc_mlp = encoder_cfg.get("mlp_ratio", 4.0)
        enc_dpr = encoder_cfg.get("drop_path_rate", 0.1)
        enc_blocks = encoder_cfg.get("block_types", None)
        num_stages = encoder_cfg.get("num_stages", len(enc_depths))

        self.encoder = InfEncoder(
            base_dim=base_dim,
            depths=enc_depths[:num_stages],
            num_heads=enc_heads[:num_stages],
            img_size=crop_size,
            window_size=enc_window,
            patch_size=patch_size,
            mlp_ratio=enc_mlp,
            drop_path_rate=enc_dpr,
            block_types=enc_blocks,
        )
        enc_dim_schedule = self.encoder.dim_schedule
        bottleneck_in_dim = enc_dim_schedule[-1]

        bn_nlayers = bottleneck_cfg.get("n_layers", 4)
        bn_layer_types = bottleneck_cfg.get("layer_types", None)
        bn_heads = bottleneck_cfg.get("num_heads", 8)
        bn_mlp = bottleneck_cfg.get("mlp_ratio", 4)
        bn_moe_scale = bottleneck_cfg.get("moe_scale", 0.1)

        self.bottleneck = INFBottleneck(
            in_dim=bottleneck_in_dim,
            n_layers=bn_nlayers,
            layer_types=bn_layer_types,
            num_heads=bn_heads,
            mlp_ratio=bn_mlp,
            moe_scale=bn_moe_scale,
        )

        dec_depths_cfg = decoder_cfg.get("depths", [])
        dec_heads_cfg = decoder_cfg.get("num_heads", [])
        dec_depths = dec_depths_cfg if dec_depths_cfg else list(reversed(enc_depths[:num_stages]))
        dec_heads = dec_heads_cfg if dec_heads_cfg else list(reversed(enc_heads[:num_stages]))
        dec_window = decoder_cfg.get("window_size", 8)
        dec_mlp = decoder_cfg.get("mlp_ratio", 4.0)
        dec_dpr = decoder_cfg.get("drop_path_rate", 0.1)

        self.decoder = InfDecoder(
            encoder_dim_schedule=enc_dim_schedule,
            depths=dec_depths,
            num_heads=dec_heads,
            window_size=dec_window,
            mlp_ratio=dec_mlp,
            drop_path_rate=dec_dpr,
            patch_size=patch_size,
            out_channels=3,
            crop_size=crop_size,
        )

        self._initialize_weights()

    @classmethod
    def from_config(cls, config: Union[str, Path, dict]) -> "InfDehaze":
        cfg = _load_yaml(config) if isinstance(config, (str, Path)) else config
        model_cfg = cfg.get("model", {})
        infer_cfg = cfg.get("inference", {})
        return cls(
            base_dim=model_cfg.get("base_dim", 96),
            patch_size=model_cfg.get("patch_size", 2),
            crop_size=model_cfg.get("crop_size", 256),
            channels_last=model_cfg.get("channels_last", True),
            encoder_cfg=model_cfg.get("encoder", {}),
            bottleneck_cfg=model_cfg.get("bottleneck", {}),
            decoder_cfg=model_cfg.get("decoder", {}),
            inference_batch_size=infer_cfg.get("batch_size", 4),
            use_residual_cache=infer_cfg.get("use_residual_cache", True),
            async_io=infer_cfg.get("async_io", True),
        )

    def forward(self, x: torch.Tensor):
        if self.training:
            return self._training_forward(x)
        return self._inference_forward(x)

    def _training_forward(self, x: torch.Tensor):
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        x_skip = x
        n_regions = x.shape[2] // self.crop_size

        x_regions = rearrange(
            x,
            "N C (HP HC) (WP WC) -> (N HP WP) C HC WC",
            HP=n_regions,
            WP=n_regions,
            HC=self.crop_size,
            WC=self.crop_size,
        )

        skips, last = self.encoder(x_regions, train=True)
        bottleneck_out, moe_loss = self.bottleneck(last, n_regions, train=True)
        output = self.decoder(
            bottleneck_out,
            skips,
            n_regions=n_regions,
            batch_size=self.inference_batch_size,
            train=True,
        )
        return output + x_skip, moe_loss

    def _inference_forward(self, x: torch.Tensor):
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)

        n_regions = x.shape[2] // self.crop_size
        if n_regions == 1:
            return self._inference_single(x)
        if self.use_residual_cache:
            if self.async_io:
                return self._inference_async_cached(x, n_regions)
            return self._inference_sync_cached(x, n_regions)
        return self._inference_no_cache(x, n_regions)

    def _inference_single(self, x: torch.Tensor) -> torch.Tensor:
        x_skip = x.cpu()
        skips_cpu, last = self.encoder(x, train=False)
        skips_batched = [[s] for s in skips_cpu]
        bottleneck_out = self.bottleneck(last, n_regions=1, train=False)
        output = self.decoder(
            bottleneck_out,
            skips_batched,
            n_regions=1,
            batch_size=1,
            train=False,
        )
        return output + x_skip.to(output.device)

    def _inference_sync_cached(self, x: torch.Tensor, n_regions: int) -> torch.Tensor:
        x_skip = x.cpu()
        x_regions = rearrange(
            x,
            "N C (HP HC) (WP WC) -> (N HP WP) C HC WC",
            HP=n_regions,
            WP=n_regions,
            HC=self.crop_size,
            WC=self.crop_size,
        )
        del x
        num_tiles = x_regions.shape[0]
        batch_size = self.inference_batch_size
        num_stages = self.encoder.num_stages

        skips_cache: List[List[torch.Tensor]] = [[] for _ in range(num_stages)]
        last_list: List[torch.Tensor] = []

        for start in range(0, num_tiles, batch_size):
            batch = x_regions[start : start + batch_size]
            skips_cpu, last = self.encoder(batch, train=False)
            for level, skip in enumerate(skips_cpu):
                skips_cache[level].append(skip)
            last_list.append(last)
            del batch

        del x_regions
        last_all = torch.cat(last_list, dim=0)
        del last_list
        bottleneck_out = self.bottleneck(last_all, n_regions, train=False)
        del last_all

        output = self.decoder(
            bottleneck_out,
            skips_cache,
            n_regions=n_regions,
            batch_size=batch_size,
            train=False,
        )
        return output + x_skip.to(output.device)

    def _inference_async_cached(self, x: torch.Tensor, n_regions: int) -> torch.Tensor:
        x_skip = x.cpu()
        x_regions = rearrange(
            x,
            "N C (HP HC) (WP WC) -> (N HP WP) C HC WC",
            HP=n_regions,
            WP=n_regions,
            HC=self.crop_size,
            WC=self.crop_size,
        )
        del x
        num_tiles = x_regions.shape[0]
        batch_size = self.inference_batch_size
        num_stages = self.encoder.num_stages

        skips_cache: List[List[torch.Tensor]] = [[] for _ in range(num_stages)]
        last_list: List[torch.Tensor] = []
        transfers: List[_AsyncCPUTransfer] = [_AsyncCPUTransfer() for _ in range(num_stages)]
        pending_skips: Optional[List[torch.Tensor]] = None

        for start in range(0, num_tiles, batch_size):
            batch = x_regions[start : min(start + batch_size, num_tiles)]
            skips_gpu, last = self.encoder(batch, train=False)
            last_list.append(last)
            del batch

            if pending_skips is not None:
                for level, transfer in enumerate(transfers):
                    skips_cache[level].append(transfer.recv())

            for level, skip in enumerate(skips_gpu):
                transfers[level].send(skip)
            pending_skips = skips_gpu

        if pending_skips is not None:
            for level, transfer in enumerate(transfers):
                skips_cache[level].append(transfer.recv())

        del x_regions
        last_all = torch.cat(last_list, dim=0)
        del last_list
        bottleneck_out = self.bottleneck(last_all, n_regions, train=False)
        del last_all

        output = self.decoder(
            bottleneck_out,
            skips_cache,
            n_regions=n_regions,
            batch_size=batch_size,
            train=False,
        )
        return output + x_skip.to(output.device)

    def _inference_no_cache(self, x: torch.Tensor, n_regions: int) -> torch.Tensor:
        x_skip = x
        x_regions = rearrange(
            x,
            "N C (HP HC) (WP WC) -> (N HP WP) C HC WC",
            HP=n_regions,
            WP=n_regions,
            HC=self.crop_size,
            WC=self.crop_size,
        )
        skips, last = self.encoder(x_regions, train=True)
        bottleneck_out = self.bottleneck(last, n_regions, train=False)
        skips_batched = [[s] for s in skips]
        output = self.decoder(
            bottleneck_out,
            skips_batched,
            n_regions=n_regions,
            batch_size=x_regions.shape[0],
            train=False,
        )
        return output + x_skip
