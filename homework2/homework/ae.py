import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "PatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


def hwc_to_chw(x: torch.Tensor) -> torch.Tensor:
    dims = list(range(x.dim()))
    dims = dims[:-3] + [dims[-1]] + [dims[-3]] + [dims[-2]]
    return x.permute(*dims)


def chw_to_hwc(x: torch.Tensor) -> torch.Tensor:
    dims = list(range(x.dim()))
    dims = dims[:-3] + [dims[-2]] + [dims[-1]] + [dims[-3]]
    return x.permute(*dims)


class PatchifyLinear(torch.nn.Module):
    def __init__(self, patch_size: int = 25, latent_dim: int = 128):
        super().__init__()
        self.patch_conv = torch.nn.Conv2d(3, latent_dim, patch_size, patch_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return chw_to_hwc(self.patch_conv(hwc_to_chw(x)))


class UnpatchifyLinear(torch.nn.Module):
    def __init__(self, patch_size: int = 25, latent_dim: int = 128):
        super().__init__()
        self.unpatch_conv = torch.nn.ConvTranspose2d(latent_dim, 3, patch_size, patch_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return chw_to_hwc(self.unpatch_conv(hwc_to_chw(x)))


class PatchAutoEncoderBase(abc.ABC):
    @abc.abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode an input image x (B, H, W, 3) into a tensor (B, h, w, bottleneck).
        """

    @abc.abstractmethod
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tensor x (B, h, w, bottleneck) into an image (B, H, W, 3).
        """


class PatchAutoEncoder(torch.nn.Module, PatchAutoEncoderBase):
    """
    Patch-level AutoEncoder.

    Encoder: patchify -> a couple of 3x3 convs (with GeLU) so neighboring patches
             interact -> 1x1 conv projecting to the bottleneck dimension.
    Decoder: mirror image of the encoder, ending in Tanh to match the [-1, 1] input range.
    """

    class PatchEncoder(torch.nn.Module):
        def __init__(self, patch_size: int, latent_dim: int, bottleneck: int):
            super().__init__()
            self.patchify = PatchifyLinear(patch_size, latent_dim)
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),
                torch.nn.GELU(),
                torch.nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),
                torch.nn.GELU(),
            )
            self.proj = torch.nn.Conv2d(latent_dim, bottleneck, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, H, W, 3)
            x = self.patchify(x)          # (B, h, w, latent_dim)
            x = hwc_to_chw(x)              # (B, latent_dim, h, w)
            x = self.net(x)
            x = self.proj(x)               # (B, bottleneck, h, w)
            return chw_to_hwc(x)           # (B, h, w, bottleneck)

    class PatchDecoder(torch.nn.Module):
        def __init__(self, patch_size: int, latent_dim: int, bottleneck: int):
            super().__init__()
            self.proj = torch.nn.Conv2d(bottleneck, latent_dim, kernel_size=1)
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),
                torch.nn.GELU(),
                torch.nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),
                torch.nn.GELU(),
            )
            self.unpatchify = UnpatchifyLinear(patch_size, latent_dim)
            self.out_act = torch.nn.Tanh()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, h, w, bottleneck)
            x = hwc_to_chw(x)              # (B, bottleneck, h, w)
            x = self.proj(x)
            x = self.net(x)
            x = chw_to_hwc(x)              # (B, h, w, latent_dim)
            x = self.unpatchify(x)         # (B, H, W, 3)
            return self.out_act(x)

    def __init__(self, patch_size: int = 25, latent_dim: int = 128, bottleneck: int = 128):
        super().__init__()
        self.encoder = self.PatchEncoder(patch_size, latent_dim, bottleneck)
        self.decoder = self.PatchDecoder(patch_size, latent_dim, bottleneck)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, {}

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)