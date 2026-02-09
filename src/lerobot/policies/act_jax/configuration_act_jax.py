from __future__ import annotations

from typing import Optional, Dict, List, Any
from flax import struct
import optax


# If you want to keep the exact names, you can use strings.
# If you already have a NormalizationMode Enum, replace this with your Enum.
NormalizationMode = str  # e.g. "mean_std" | "min_max"


@struct.dataclass
class ACTConfigJAX:
    # Input / output structure
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    normalization_mapping: Dict[str, NormalizationMode] = struct.field(
        default_factory=lambda: {
            "VISUAL": "mean_std",
            "STATE": "mean_std",
            "ACTION": "mean_std",
        }
    )

    # Architecture
    # vision_backbone: str = "resnet18"
    # pretrained_backbone_weights: Optional[str] = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: bool = False

    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1  # match bug-compatible behavior

    # VAE
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Inference
    temporal_ensemble_coeff: Optional[float] = None

    # Training / loss
    dropout: float = 0.1
    kl_weight: float = 10.0

    # Optim preset
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5  # if you separate params by backbone

    # (Optional) flags/features if your pipeline uses them
    image_features: bool = True
    env_state_feature: bool = False

    def validate(self) -> None:
        # equivalent to __post_init__ of mutable dataclass
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. "
                "This is because the policy needs to be queried every step to compute the ensembled action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                "The chunk size is the upper bound for the number of action steps per model invocation. "
                f"Got {self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `n_obs_steps={self.n_obs_steps}`"
            )

    def validate_features(self) -> None:
        if (not self.image_features) and (not self.env_state_feature):
            raise ValueError(
                "You must provide at least one image or the environment state among the inputs."
            )

    # Optax "preset" (equivalent to get_optimizer_preset)
    def make_optimizer(self) -> optax.GradientTransformation:
        return optax.adamw(
            learning_rate=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def make_scheduler(self):
        # equivalent to get_scheduler_preset -> None
        return None

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> List[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None