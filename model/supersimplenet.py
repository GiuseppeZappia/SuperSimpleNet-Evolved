import math
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import MultiStepLR, LRScheduler
from torchvision.transforms import GaussianBlur

from common.perlin_noise import rand_perlin_2d
from .feature_extractor import FeatureExtractor


class SuperSimpleNet(nn.Module):
    """
    SuperSimpleNet model

    Args:
        image_size: tuple with image dims (h, w)
        config: dict with model properties
    """

    def __init__(self, image_size: tuple[int, int], config):
        super().__init__()
        self.image_size = image_size
        self.config = config
        self.feature_extractor = FeatureExtractor(
            #changing the backbone to "resnet18" from "wide_resnet50_2"
            #the new size of the channels are automatically managed in feature_extractor.py
            backbone=config.get("backbone", "resnet18"),
            layers=config.get("layers", ["layer2", "layer3"]),
            patch_size=config.get("patch_size", 3),
            image_size=image_size,
        )
        # feature channels, height and width
        fc, fh, fw = self.feature_extractor.feature_dim
        self.fh = fh
        self.noise_generator = LearnedNoiseGenerator(fc)#added to perform new noise
        self.fw = fw

        # Getting config param to choose Case (A or B)
        non_linear_adaptor = config.get("non_linear_adaptor", False)
        # giving to FeatureAdaptor the choice of Case
        self.feature_adaptor = FeatureAdaptor(projection_dim=fc, non_linear=non_linear_adaptor)
        self.adapt_cls_feat = config.get("adapt_cls_feat", False)

        self.discriminator = Discriminator(
            projection_dim=fc,
            hidden_dim=1024,
            feature_w=fw,
            feature_h=fh,
            config=config,
        )

        self.anomaly_generator = AnomalyGenerator(
            noise_mean=0,
            noise_std=config.get("noise_std", 0.015),
            feature_w=fw,
            feature_h=fh,
            f_dim=fc,
            config=config,
        )
        self.anomaly_map_generator = AnomalyMapGenerator(
            output_size=image_size, sigma=4
        )

    # Updated SuperSimpleNet.forward to return the noise tensor during training
    def forward(
        self,
        images: Tensor,
        mask: Tensor = None,
        label: Tensor = None,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # feature extraction, upscaling and neigh. aggregation
        features = self.feature_extractor(images)
        adapted = self.feature_adaptor(features)

        seg_feats = adapted
        if self.adapt_cls_feat:
            cls_feats = adapted
        else:
            cls_feats = features

        if self.training:
            full_noise = None
            # add noise to features
            if self.config["noise"]:
                epsilon_tilde = self.noise_generator(adapted)
                full_noise = torch.cat([epsilon_tilde, epsilon_tilde], dim=0)
                
                if self.adapt_cls_feat:
                    _, noised_adapt, mask, label = self.anomaly_generator(
                        features=None, adapted=adapted, mask=mask, labels=label, learned_noise=full_noise
                    )
                    seg_feats = noised_adapt
                    cls_feats = noised_adapt
                else:
                    noised_feat, noised_adapt, mask, label = self.anomaly_generator(
                        features=features, adapted=adapted, mask=mask, labels=label, learned_noise=full_noise
                    )
                    seg_feats = noised_adapt
                    cls_feats = noised_feat

            anomaly_map, anomaly_score = self.discriminator(
                seg_features=seg_feats, cls_features=cls_feats
            )
            # Added full_noise to return values for regularization loss calculation
            return anomaly_map, anomaly_score, mask, label, full_noise
        else:
            anomaly_map, anomaly_score = self.discriminator(
                seg_features=seg_feats, cls_features=cls_feats
            )
            anomaly_map = self.anomaly_map_generator(anomaly_map)
            return anomaly_map, anomaly_score

    def get_optimizers(self) -> tuple[Optimizer, LRScheduler]:
        seg_params, dec_params = self.discriminator.get_params()
        optim = AdamW(
            [
                {
                    "params": self.feature_adaptor.parameters(),
                    "lr": self.config["adapt_lr"],
                },
                {
                    "params": seg_params,
                    "lr": self.config["seg_lr"],
                    "weight_decay": 0.00001,
                },
                {
                    "params": dec_params,
                    "lr": self.config["dec_lr"],
                    "weight_decay": 0.00001,
                },
            ]
        )

        # Adversarial Generator Optimizer
        # We use a dedicated learning rate (if not in config, default 1e-4)
        gen_lr = self.config.get("gen_lr", 0.0001)
        optim_gen = AdamW(self.noise_generator.parameters(), lr=gen_lr)
        sched = MultiStepLR(
            optim,
            milestones=[self.config["epochs"] * 0.8, self.config["epochs"] * 0.9],
            gamma=self.config["gamma"],
        )

        return optim, optim_gen, sched

    def save_model(self, path: Path):
        path.mkdir(exist_ok=True, parents=True)
        state_dict = self.state_dict()
        # exclude feat extractor since it's pretrained
        saving_state_dict = OrderedDict(
            {
                n: k
                for n, k in state_dict.items()
                if not n.startswith("feature_extractor")
            }
        )

        torch.save(saving_state_dict, path / "weights.pt")

    def load_model(self, path):
        print(f"Loading model: {path}")
        self.load_state_dict(torch.load(path), strict=False)


def init_weights(m: nn.Module):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.constant_(m.weight, 1)


class FeatureAdaptor(nn.Module):    
    def __init__(self, projection_dim: int, non_linear: bool = False):
        super().__init__()
        if non_linear:
            # Case B: 3x3 Convolution + BatchNorm + ReLU
            self.projection = nn.Sequential(
                nn.Conv2d(
                    in_channels=projection_dim,
                    out_channels=projection_dim,
                    kernel_size=3,
                    padding=1, # keeping spatial dimensions
                ),
                nn.BatchNorm2d(projection_dim),
                nn.ReLU(inplace=True)
            )
        else:
            # Case A / Original: 1x1 Convolution (Linear)
            self.projection = nn.Conv2d(
                in_channels=projection_dim,
                out_channels=projection_dim,
                kernel_size=1,
                stride=1,
            )
        self.apply(init_weights)

    def forward(self, features: Tensor) -> Tensor:
        return self.projection(features)


def _conv_block(in_chanels, out_chanels, kernel_size, padding="same"):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_chanels,
            out_channels=out_chanels,
            kernel_size=kernel_size,
            padding=padding,
        ),
        nn.BatchNorm2d(out_chanels),
        nn.ReLU(inplace=True),
    )


class Discriminator(nn.Module):
    def __init__(
        self, projection_dim: int, hidden_dim: int, feature_w, feature_h, config
    ):
        super().__init__()
        self.fw = feature_w
        self.fh = feature_h
        self.stop_grad = config.get("stop_grad", False)

        # 1x1 conv - linear layer equivalent
        self.seg = nn.Sequential(
            nn.Conv2d(
                in_channels=projection_dim,
                out_channels=hidden_dim,
                kernel_size=1,
                stride=1,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=1,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )

        self.dec_head = _conv_block(
            in_chanels=projection_dim + 1, out_chanels=128, kernel_size=5
        )

        self.map_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.map_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        self.dec_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dec_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        self.fc_score = nn.Linear(in_features=128 * 2 + 2, out_features=1)

        self.apply(init_weights)

    def get_params(self):
        seg_params = self.seg.parameters()
        dec_params = list(self.dec_head.parameters()) + list(self.fc_score.parameters())
        return seg_params, dec_params

    def forward(
        self, seg_features: Tensor, cls_features: Tensor
    ) -> tuple[Tensor, Tensor]:
        # get anomaly map from seg head
        map = self.seg(seg_features)

        map_dec_copy = map
        if self.stop_grad:
            map_dec_copy = map_dec_copy.detach()
        # dec conv layer takes feat + map
        mask_cat = torch.cat((cls_features, map_dec_copy), dim=1)
        dec_out = self.dec_head(mask_cat)

        dec_max = self.dec_max_pool(dec_out)
        dec_avg = self.dec_avg_pool(dec_out)

        map_max = self.map_max_pool(map)
        if self.stop_grad:
            map_max = map_max.detach()

        map_avg = self.map_avg_pool(map)
        if self.stop_grad:
            map_avg = map_avg.detach()

        # final dec layer: conv channel max and avg and map max and avg
        dec_cat = torch.cat((dec_max, dec_avg, map_max, map_avg), dim=1).squeeze(
            dim=(2, 3)
        )
        score = self.fc_score(dec_cat).squeeze(dim=1)

        return map, score


class LearnedNoiseGenerator(nn.Module):
    def __init__(self, f_dim: int):
        super().__init__()
        # 3 linear layer (implemented as conv 1x1)
        self.model = nn.Sequential(
            nn.Conv2d(f_dim, f_dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f_dim // 2, f_dim // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f_dim // 2, f_dim, kernel_size=1)
        )
        self.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        # Modification A: Constrain output using tanh and a small scalar
        # Original paper uses sigma=0.015. 
        # We use 0.1 to allow more flexibility while preventing feature explosion.
        raw_noise = self.model(x)
        return torch.tanh(raw_noise) * 0.1
    

class AnomalyGenerator(nn.Module):
    def __init__(
        self,
        noise_mean: float,
        noise_std: float,
        feature_h: int,
        feature_w: int,
        f_dim: int,
        config: dict,
        perlin_range: tuple[int, int] = (0, 6),
    ):
        super().__init__()

        self.noise_mean = noise_mean
        self.noise_std = noise_std

        self.min_perlin_scale = perlin_range[0]
        self.max_perlin_scale = perlin_range[1]

        self.height = feature_h
        self.width = feature_w
        self.f_dim = f_dim

        self.config = config

        self.perlin_height = self.next_power_2(self.height)
        self.perlin_width = self.next_power_2(self.width)

    @staticmethod
    def next_power_2(num):
        return 1 << (num - 1).bit_length()

    def generate_perlin(self, batches) -> Tensor:
        """
        Generate 2d perlin noise masks with dims [b, 1, self.h, self.w]

        Args:
            batches: number of batches (different masks)

        Returns:
            tensor with b perlin binarized masks
        """
        perlin = []
        for _ in range(batches):
            perlin_scalex = (
                2
                ** (
                    torch.randint(
                        self.min_perlin_scale, self.max_perlin_scale, (1,)
                    ).numpy()[0]
                )
            )
            perlin_scaley = (
                2
                ** (
                    torch.randint(
                        self.min_perlin_scale, self.max_perlin_scale, (1,)
                    ).numpy()[0]
                )
            )

            perlin_noise = rand_perlin_2d(
                (self.perlin_height, self.perlin_width), (perlin_scalex, perlin_scaley)
            )
            # original is power of 2 scale, so fit to our size
            perlin_noise = F.interpolate(
                perlin_noise.reshape(1, 1, self.perlin_height, self.perlin_width),
                size=(self.height, self.width),
                mode="bilinear",
                align_corners=False,
            )
            threshold = self.config["perlin_thr"]
            # binarize
            perlin_thr = torch.where(perlin_noise > threshold, 1, 0)

            chance_anomaly = torch.rand(1).numpy()[0]
            if chance_anomaly > 0.5:
                if self.config["no_anomaly"] == "full":
                    # entire image is anomaly
                    perlin_thr = torch.ones_like(perlin_thr)
                elif self.config["no_anomaly"] == "empty":
                    # no anomaly
                    perlin_thr = torch.zeros_like(perlin_thr)
                # if none -> don't add

            perlin.append(perlin_thr)
        return torch.cat(perlin)

    def forward(
            self, features: Tensor | None, adapted: Tensor, mask: Tensor, labels: Tensor, learned_noise: Tensor = None
        ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            b, _, h, w = mask.shape


            adapted = torch.cat((adapted, adapted))
            mask = torch.cat((mask, mask))
            labels = torch.cat((labels, labels))
            if features is not None:
                features = torch.cat((features, features))

            # if learned noise is provided, use it; otherwise use standard Gaussian noise 
            if learned_noise is not None:
                noise = learned_noise
            else:
                noise = torch.normal(
                    mean=self.noise_mean,
                    std=self.noise_std,
                    size=adapted.shape,
                    device=adapted.device,
                    requires_grad=False,
                )

            noise_mask = torch.ones(
                b * 2, 1, h, w, device=adapted.device, requires_grad=False
            )

            if not self.config["bad"]:
                masking_labels = labels.reshape(b * 2, 1, 1, 1)
                noise_mask = noise_mask * (1 - masking_labels)

            if not self.config["overlap"]:
                noise_mask = noise_mask * (1 - mask)

            if self.config["perlin"]:
                perlin_mask = self.generate_perlin(b * 2).to(adapted.device)
                noise_mask = noise_mask * perlin_mask
            else:
                noise_mask[:b, ...] = 0

            mask = mask + noise_mask
            mask = torch.where(mask > 0, 1, 0)

            new_anomalous = mask.reshape(b * 2, -1).any(dim=1).type(torch.float32)
            labels = labels + new_anomalous
            labels = torch.where(labels > 0, 1, 0)

            perturbed_adapt = adapted + noise * noise_mask
            if features is not None:
                perturbed_feat = features + noise * noise_mask
            else:
                perturbed_feat = None

            return perturbed_feat, perturbed_adapt, mask, labels


class AnomalyMapGenerator(nn.Module):
    def __init__(self, output_size: tuple[int, int], sigma: float):
        super().__init__()
        self.size = output_size
        kernel_size = 2 * math.ceil(3 * sigma) + 1
        self.blur = GaussianBlur(kernel_size=kernel_size, sigma=4)

    def forward(self, input: Tensor) -> Tensor:
        # upscale & smooth
        anomaly_map = F.interpolate(input, size=self.size, mode="bilinear")
        anomaly_map = self.blur(anomaly_map)
        return anomaly_map
