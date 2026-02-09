#!/usr/bin/env python

from __future__ import annotations
import jax
import jax.numpy as jnp
import jax.lax as lax
import flax.linen as nn
from typing import Optional, Dict, Any
from flax import struct
from flax.core import FrozenDict
import optax

from collections import deque
from lerobot.policies.pretrained import PreTrainedPolicy
# from lerobot.policies.act_jax.configuration_act_jax import ACTConfigJAX
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

XAVIER = nn.initializers.xavier_uniform()
ZEROS = nn.initializers.zeros

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
    image_features: Any = True
    env_state_feature: Optional[Any] = None
    
    # TODO: NOT FROM ACT CONFIG JAX BUT FROM PRETRAINED CONFIG
    robot_state_feature: Optional[Any] = None 
    action_feature: Optional[Any] = None

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
        if (not self.image_features) and (self.env_state_feature is None):
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

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

class ACTPolicyJAX:
    config_class = ACTConfigJAX
    name = "act_jax"

    def __init__(self, config: ACTConfigJAX):
        # TODO
        # Create a PretrainedPolicyJAX
        # super().__init__(config)
        
        # TODO still not implemented
        # config.validate_features()
        self.config = config
        self.model = ACTJAX(config)
        
        self.variables: Optional[FrozenDict] = None
        
        # Temporal emsembling
        self.temporal_ensembler = ACTTemporalEnsemblerJAX(coeff=0.1, chunk_size=self.config.chunk_size)
        self._ens_state: Optional[EnsemblerState] = None
        
        self.reset()

    def set_variables(self, variables: FrozenDict):
        self.variables = variables
    
    def reset(self):
        if self.temporal_ensembler is not None:
            self._ens_state = None
        else:
            n_action_steps = int(self.config.n_action_steps)
            self._action_queue = deque([], maxlen=n_action_steps)

    def _prepare_batch_images(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        img_feat = getattr(self.config, "image_features", False)
        if isinstance(img_feat, (list, tuple)):
            b = dict(batch)
            b[OBS_IMAGES] = [batch[k] for k in img_feat]
            return b
        return batch
    
    def predict_action_chunk(
        self,
        batch: Dict[str, Any],
        *,
        deterministic: bool = True,
        rngs: Optional[Dict[str, jax.random.Array]] = None,
    ) -> jnp.ndarray:
        if self.variables is None:
            raise ValueError("Model variables have not been set. Call `set_variables` before prediction.")

        batch = self._prepare_batch_images(batch)
        
        actions, _ = self.model.apply(
            self.variables,
            batch,
            deterministic=deterministic,
            rngs=rngs,
        )  # (B, chunk_size, action_dim)
        
        return actions
    
    def select_action(
        self,
        batch: Dict[str, Any],
        *,
        deterministic: bool = True,
        rngs: Optional[Dict[str, jax.random.Array]] = None,
    ) -> jnp.ndarray:
        # returns only one action (B, action_dim)
        if self.temporal_ensembler is not None:
            actions = self.predict_action_chunk(batch, deterministic=deterministic, rngs=rngs) # (B,S,A)
            B, S, A = actions.shape
            
            if self._ens_state is None:
                self._ens_state = self.temporal_ensembler.init_state(B, A)
            
            action, self._ens_state = self.temporal_ensembler.update(self._ens_state, actions)  # (B,A), new_state
        
            return action # (B,A)
    
        # Without temporal ensembling
        if len(self._action_queue) == 0:
            n_action_steps = int(self.config.n_action_steps)
            actions = actions[:, :n_action_steps, :] # (B, n_action_steps, A)
            
            for i in range(actions.shape[1]):
                self._action_queue.append(actions[:, i, :])  # append (B,A)
            
        return self._action_queue.popleft() # (B,A)
    
    # ----- training forward ----- (for loss computation)
    def forward(
        self,
        batch: Dict[str, Any],
        *,
        rng_dropout: jax.Array,
        rng_sample: Optional[jax.Array] = None,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        # Calculates l1 and KL
        # requires deterministic=False
        if self.variables is None:
            raise ValueError("Model variables have not been set. Call `set_variables` before prediction.")
        
        batch = self._prepare_batch_images(batch)
        rngs = {"dropout": rng_dropout}
        if rng_sample is not None:
            rngs["sample"] = rng_sample
        
        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model.apply(
            self.variables,
            batch, 
            deterministic=False,
            rngs=rngs,
        )
        
        pad = batch["action_is_pad"].astype(jnp.bool_) # (B,S) bool
        mask = (~pad).astype(actions_hat.dtype)[..., None]      # (B,S) float32
        l1_loss = jnp.mean(jnp.abs(batch[ACTION] - actions_hat) * mask) # scalar
        
        loss_dict: Dict[str, jnp.ndarray] = {"l1_loss": l1_loss}
        
        if getattr(self.config, "use_vae", False):
            kl_weight = jnp.asarray(getattr(self.config, "kl_weight", 1.0), dtype=actions_hat.dtype)
            mean_kld = (-0.5 * (1.0 + log_sigma_x2_hat - (mu_hat**2) - jnp.exp(log_sigma_x2_hat))).sum(-1).mean()
            loss_dict["kld_loss"] = mean_kld
            loss = l1_loss + kl_weight * mean_kld
        else:
            loss = l1_loss

        return loss, loss_dict
        
# ---------- ResNet Backbone ----------

class FrozenBatchNorm(nn.Module):
    """
    Frozen BatchNorm for NHWC, equivalent to torchvision's FrozenBatchNorm2d.

    Stores {mean, var, weight, bias} in the 'batch_stats' collection and does not update them
    during training (they are treated as fixed statistics/affine params).

    Defaults (initial values):
      - mean = 0, var = 1
      - weight = 1, bias = 0
    so the layer starts as an identity transform (up to eps). These values are typically
    overwritten when loading pretrained checkpoints.
    """
    features: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        mean = self.variable("batch_stats", "mean", lambda: jnp.zeros((self.features,), jnp.float32))
        var  = self.variable("batch_stats", "var",  lambda: jnp.ones((self.features,), jnp.float32))
        w    = self.variable("batch_stats", "weight", lambda: jnp.ones((self.features,), jnp.float32))
        b    = self.variable("batch_stats", "bias",   lambda: jnp.zeros((self.features,), jnp.float32))
        x = (x - mean.value) / jnp.sqrt(var.value + self.eps)
        return x * w.value + b.value


class BasicBlock(nn.Module):
    """ResNet BasicBlock for ResNet-18/34 (NHWC). expansion=1."""
    out_ch: int
    stride: int = 1
    dilation: int = 1
    use_proj: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        BN = FrozenBatchNorm

        def Conv3x3(c, s=1, d=1):
            return nn.Conv(
                features=c,
                kernel_size=(3, 3),
                strides=(s, s),
                padding="SAME",
                kernel_dilation=(d, d),
                use_bias=False,
                dtype=self.dtype,
            )

        def Conv1x1(c, s=1):
            return nn.Conv(
                features=c,
                kernel_size=(1, 1),
                strides=(s, s),
                padding="SAME",
                use_bias=False,
                dtype=self.dtype,
            )

        # Shortcut S(x)
        residual = x
        if self.use_proj:
            residual = Conv1x1(self.out_ch, s=self.stride)(residual)
            residual = BN(self.out_ch)(residual)

        # Main path F(x)
        y = Conv3x3(self.out_ch, s=self.stride, d=self.dilation)(x)
        y = BN(self.out_ch)(y)
        y = nn.relu(y)

        y = Conv3x3(self.out_ch, s=1, d=self.dilation)(y)
        y = BN(self.out_ch)(y)

        return nn.relu(y + residual)

class ResNet18BackboneJAX(nn.Module):
    """
    ResNet-18 backbone that returns {"feature_map": layer4}

    - Input: x in NHWC (B,H,W,C)
    - Output: dict with the final stage feature map (layer4)
    """
    replace_final_stride_with_dilation: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x) -> Dict[str, jnp.ndarray]:
        BN = FrozenBatchNorm

        # Stem: 7x7 conv (stride 2) + maxpool (stride 2)
        x = nn.Conv(64, (7, 7), strides=(2, 2), padding="SAME", use_bias=False, dtype=self.dtype)(x)
        x = BN(64)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")

        # ResNet-18 stages: [2,2,2,2] with base channels [64,128,256,512]
        def stage(x, blocks: int, out_ch: int, first_stride: int, dilation: int):
            # First block may change spatial size and/or channels => projection shortcut
            use_proj = (first_stride != 1) or (x.shape[-1] != out_ch)
            x = BasicBlock(out_ch, stride=first_stride, dilation=dilation, use_proj=use_proj, dtype=self.dtype)(x)
            # Remaining blocks keep shape => no projection, stride=1
            for _ in range(blocks - 1):
                x = BasicBlock(out_ch, stride=1, dilation=dilation, use_proj=False, dtype=self.dtype)(x)
            return x

        # layer1 keeps resolution
        x = stage(x, blocks=2, out_ch=64,  first_stride=1, dilation=1)  # layer1
        # layer2 downsamples
        x = stage(x, blocks=2, out_ch=128, first_stride=2, dilation=1)  # layer2
        # layer3 downsamples
        x = stage(x, blocks=2, out_ch=256, first_stride=2, dilation=1)  # layer3

        # layer4: normally downsamples (stride=2), or keep stride=1 and use dilation=2
        if self.replace_final_stride_with_dilation:
            x = stage(x, blocks=2, out_ch=512, first_stride=1, dilation=2)  # layer4 (dilated)
        else:
            x = stage(x, blocks=2, out_ch=512, first_stride=2, dilation=1)  # layer4 (normal)

        return {"feature_map": x}  # NHWC

# ---------- ACT Temporal Ensembler ----------
@struct.dataclass
class EnsemblerState:
    initialized: jnp.ndarray          # () bool
    queue: jnp.ndarray                # (B, S-1, A)
    count: jnp.ndarray                # (S-1, 1) int32

class ACTTemporalEnsemblerJAX:
    def __init__(self, coeff: float, chunk_size: int):
        self.coeff = coeff
        self.S = chunk_size

    def init_state(self, batch_size: int, action_dim: int) -> EnsemblerState:
        return EnsemblerState(
            initialized=jnp.array(False),
            queue=jnp.zeros((batch_size, self.S - 1, action_dim), dtype=jnp.float32),
            count=jnp.ones((self.S - 1, 1), dtype=jnp.int32),
        )

    def _weights(self, dtype):
        i = jnp.arange(self.S, dtype=dtype)
        w = jnp.exp(-jnp.asarray(self.coeff, dtype=dtype) * i)     # (S,)
        w_csum = jnp.cumsum(w)                                     # (S,)
        return w, w_csum

    def update(self, state: EnsemblerState, actions: jnp.ndarray):
        # actions: (B, S, A)
        _, S, _ = actions.shape
        assert S == self.S

        w, w_csum = self._weights(actions.dtype)

        def first(_):
            full = actions                              # (B,S,A)
            full_count = jnp.ones((S, 1), jnp.int32)     # (S,1)
            return full, full_count

        def subsequent(_):
            prev = state.queue          # (B,S-1,A)
            cnt  = state.count          # (S-1,1)  values 1..S-1

            idx_prev = (cnt - 1).squeeze(-1)  # (S-1,)
            idx_new  = cnt.squeeze(-1)        # (S-1,)

            prev_scale = w_csum[idx_prev]     # (S-1,)
            new_weight = w[idx_new]           # (S-1,)
            den        = w_csum[idx_new]      # (S-1,)

            prev = prev * prev_scale[None, :, None]
            prev = prev + actions[:, :-1, :] * new_weight[None, :, None]
            prev = prev / den[None, :, None]

            cnt = jnp.minimum(cnt + 1, self.S-1).astype(jnp.int32)   # (S-1,1)

            full = jnp.concatenate([prev, actions[:, -1:, :]], axis=1)           # (B,S,A)
            full_count = jnp.concatenate([cnt, jnp.ones((1,1), jnp.int32)], axis=0)  # (S,1)
            return full, full_count

        full, full_count = jax.lax.cond(state.initialized, subsequent, first, operand=None)

        action_to_execute = full[:, 0, :]         # (B,A)
        new_queue = full[:, 1:, :]                # (B,S-1,A)
        new_count = full_count[1:, :]             # (S-1,1)

        new_state = state.replace(initialized=jnp.array(True), queue=new_queue, count=new_count)
        return action_to_execute, new_state


class ACTJAX(nn.Module):
    config: ACTConfigJAX
    
    def setup(self):
        if self.config.use_vae:
            self.vae_encoder = ACTEncoderJAX(config=self.config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embed(
                num_embeddings=1,
                features=self.config.dim_model,
                # embedding_init=XAVIER
            )
            # Projection layer for joint-space configuration to hidden dimension.
            # TODO
            # robot_state_feature is in PretrainedConfig!!!!!!
            if self.config.robot_state_feature is not None:
                # input dim should be self.config.robot_state_feature.shape[0]
                self.vae_encoder_robot_state_input_proj = nn.Dense(self.config.dim_model, kernel_init=XAVIER, bias_init=ZEROS)
            # Projection layer for action (joint-space target) to hidden dimension
            # input dim should be self.config.action_feature.shape[0]
            self.vae_encoder_action_input_proj = nn.Dense(self.config.dim_model, kernel_init=XAVIER, bias_init=ZEROS)
            # Projection layer from the VAE encoder's output to the latent distribution's parameter space.
            self.vae_encoder_latent_output_proj = nn.Dense(self.config.latent_dim * 2, kernel_init=XAVIER, bias_init=ZEROS)
            # Fixed sinusoidal positional embedding for the input to the VAE encoder. Unsqueeze for batch
            # dimension.
            num_input_token_encoder = 1 + self.config.chunk_size
            
            if self.config.robot_state_feature is not None:
                num_input_token_encoder += 1
            
            pe = create_sinusoidal_pos_embedding_2(num_input_token_encoder, self.config.dim_model) # (S,D)
            pe = pe[None, :, :]  # instead of .unsqueeze(0) # (1,S+2,D)
            self.vae_encoder_pos_enc = pe # (1,S+2,D)
                
            
        # Backbone for image feature extraction.
        if self.config.image_features is not None:
            self.backbone = ResNet18BackboneJAX(
                replace_final_stride_with_dilation=self.config.replace_final_stride_with_dilation
            )


        # Transformer encoder and decoder.
        self.encoder = ACTEncoderJAX(self.config)
        self.decoder = ACTDecoderJAX(self.config)
        
        if self.config.robot_state_feature is not None:
            self.encoder_robot_state_input_proj = nn.Dense(self.config.dim_model, kernel_init=XAVIER, bias_init=ZEROS)
        if self.config.env_state_feature is not None:
            self.encoder_env_state_input_proj = nn.Dense(self.config.dim_model, kernel_init=XAVIER, bias_init=ZEROS)
        self.encoder_latent_input_proj = nn.Dense(self.config.dim_model, kernel_init=XAVIER, bias_init=ZEROS)
        if self.config.image_features is not None:
            self.encoder_img_feat_input_proj = nn.Conv(
                features=self.config.dim_model,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="SAME",
                use_bias=True,
                kernel_init=XAVIER,
                bias_init=ZEROS,
                dtype=jnp.float32,
            )
        
        # Transformer encoder positional embeddings
        n_1d_tokens = 1 # for the latent
        if self.config.robot_state_feature is not None:
            n_1d_tokens += 1
        if self.config.env_state_feature is not None:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embed(
            num_embeddings=n_1d_tokens,
            features=self.config.dim_model,
            # embedding_init=XAVIER
        )
        if self.config.image_features is not None:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionalEmbeddingJAX(self.config)
        
        # Transformer decoder
        # Learnable positional embedding for the transformer's decoder (in the style of DETR object queries).
        self.decoder_pos_embed = nn.Embed(num_embeddings=self.config.chunk_size, features=self.config.dim_model, embedding_init=XAVIER)
        
        # Final action regression head on the output of the transformer's decoder.
        assert self.config.action_feature is not None
        self.action_head = nn.Dense(self.config.action_feature.shape[0], kernel_init=XAVIER, bias_init=ZEROS)

    def __call__(self, batch, *, deterministic: bool):
        if self.config.use_vae and (not deterministic):
            assert ACTION in batch, "actions must be provided when using the variational objective in training mode."

        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]

        # --- VAE Encoder Forward --- (Training)
        if self.config.use_vae and (not deterministic):
            cls_idx = jnp.zeros((batch_size, 1), dtype=jnp.int32) # (B,1)
            cls_embed = self.vae_encoder_cls_embed(cls_idx) # (B,1,D)

            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])  # (B,S,D)

            if self.config.robot_state_feature is not None:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE])[:, None, :]  # (B,1,D)
                vae_encoder_input = jnp.concatenate([cls_embed, robot_state_embed, action_embed], axis=1)  # (B,S+2,D)
                n_prefix = 2
            else:
                vae_encoder_input = jnp.concatenate([cls_embed, action_embed], axis=1)  # (B,S+1,D)
                n_prefix = 1

            # (1,S+2,D) -> (S+2,1,D)
            pos_embed = jnp.transpose(self.vae_encoder_pos_enc, (1, 0, 2))  # (S+2,1,D)

            cls_joint_is_pad = jnp.zeros((batch_size, n_prefix), dtype=jnp.bool_) # (B, 1) or (B,2)
            action_is_pad = batch["action_is_pad"].astype(jnp.bool_) 
            key_padding_mask = jnp.concatenate([cls_joint_is_pad, action_is_pad], axis=1)  # (B,S+2) or (B,S+1)

            enc_out = self.vae_encoder(
                jnp.transpose(vae_encoder_input, (1, 0, 2)),  # (S+2,B,D)
                pos_embed=pos_embed,                          # (S+2,1,D)
                key_padding_mask=key_padding_mask,            # (B,S+2)
                deterministic=deterministic,                  # training mode 
            ) # (S+2,B,D)
            cls_token_out = enc_out[0]  # (B,D)

            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)  # (B,2L)
            mu = latent_pdf_params[:, : self.config.latent_dim] # (B,L)
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :] # (B,L)

            eps = jax.random.normal(self.make_rng("sample"), mu.shape, dtype=mu.dtype) # (B,L)
            latent_sample = mu + jnp.exp(0.5 * log_sigma_x2) * eps # (B,L)
        else:
            # --- No VAE Encoder --- (Inference)
            mu = None
            log_sigma_x2 = None
            latent_sample = jnp.zeros((batch_size, self.config.latent_dim), dtype=jnp.float32) # (B,L)

        # --- Transformer encoder ---
        encoder_tokens = []
        encoder_pos    = []

        pos_1d = self.encoder_1d_feature_pos_embed(jnp.arange(self.encoder_1d_feature_pos_embed.num_embeddings, dtype=jnp.int32))

        encoder_tokens.append(self.encoder_latent_input_proj(latent_sample)[None, ...])     # (1,B,D)
        encoder_pos.append(pos_1d[0][None, None, :])                                         # (1,1,D)
        idx = 1

        if self.config.robot_state_feature is not None:
            encoder_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE])[None, ...])  # (1,B,D)
            encoder_pos.append(pos_1d[idx][None, None, :])                                           # (1,1,D)
            idx += 1

        if self.config.env_state_feature is not None:
            encoder_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE])[None, ...]) # (1,B,D)
            encoder_pos.append(pos_1d[idx][None, None, :])                                            # (1,1,D)
            idx += 1


        if self.config.image_features is not None:
            for img in batch[OBS_IMAGES]:
                cam_feat = self.backbone(img)["feature_map"]
                cam_pos  = self.encoder_cam_feat_pos_embed(cam_feat).astype(cam_feat.dtype)  # (1,H,W,D)
                cam_feat = self.encoder_img_feat_input_proj(cam_feat)  # (B,H,W,D)

                B, Hf, Wf, D = cam_feat.shape
                cam_feat = cam_feat.reshape(B, Hf * Wf, D).transpose(1, 0, 2)         # (Simg,B,D)
                cam_pos  = cam_pos.reshape(1, Hf * Wf, D).transpose(1, 0, 2)           # (Simg,1,D)

                encoder_tokens.append(cam_feat)
                encoder_pos.append(cam_pos)

        encoder_in_tokens    = jnp.concatenate(encoder_tokens, axis=0)  # (S,B,D)
        encoder_in_pos_embed = jnp.concatenate(encoder_pos,    axis=0)  # (S,1,D)

        encoder_out = self.encoder(
            encoder_in_tokens,
            pos_embed=encoder_in_pos_embed,
            deterministic=deterministic,
        ) # (S,B,D)

        # --- Transformer decoder ---
        decoder_in = jnp.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_tokens.dtype,
        ) # (chunk,B,D)

        dec_idx = jnp.arange(self.config.chunk_size, dtype=jnp.int32)[:, None]  # (chunk,1)
        decoder_pos = self.decoder_pos_embed(dec_idx)                           # (chunk,1,D)

        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=decoder_pos,
            deterministic=deterministic,
        ) # (chunk,B,D)

        decoder_out = jnp.swapaxes(decoder_out, 0, 1)      # (B,chunk,D)
        actions = self.action_head(decoder_out)            # (B,chunk,action_dim)

        return actions, (mu, log_sigma_x2)


class ACTEncoderJAX(nn.Module):
    config: ACTConfigJAX
    is_vae_encoder: bool = False

    def setup(self):
        num_layers = self.config.n_vae_encoder_layers if self.is_vae_encoder else self.config.n_encoder_layers
        self.layers = [ACTEncoderLayerJAX(self.config) for _ in range(num_layers)]
        self.norm = nn.LayerNorm() if self.config.pre_norm else None

    def __call__(self, x, pos_embed=None, key_padding_mask=None, *, deterministic: bool):
        for layer in self.layers:
            x = layer(
                x,
                pos_embed=pos_embed,
                key_padding_mask=key_padding_mask,
                deterministic=deterministic,
            )
        if self.norm is not None:
            x = self.norm(x)
        return x

class ACTEncoderLayerJAX(nn.Module):
    config: ACTConfigJAX
    
    def setup(self):
        self.self_attn = nn.MultiHeadAttention(
            self.config.n_heads,
            dropout_rate=self.config.dropout,
            kernel_init=XAVIER,
        )
        
        # In Flax, nn.Dense infers the input dimension at initialization
        self.linear1 = nn.Dense(self.config.dim_feedforward, kernel_init=XAVIER, bias_init=ZEROS)
        self.dropout = nn.Dropout(rate=self.config.dropout)
        self.linear2 = nn.Dense(self.config.dim_model,kernel_init=XAVIER, bias_init=ZEROS)
        
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.dropout1 = nn.Dropout(rate=self.config.dropout)
        self.dropout2 = nn.Dropout(rate=self.config.dropout)
        
        self.activation = get_activation_fn(self.config.feedforward_activation)
        self.pre_norm = self.config.pre_norm

    def __call__(self, x, pos_embed = None, key_padding_mask = None, *, deterministic: bool):
        # Torch works with (S, B, D), Flax with (B, S, D)
        x_bsd = jnp.swapaxes(x, 0, 1)
        assert x_bsd.shape[-1] == self.config.dim_model
        
        pos_embed_bsd = None if pos_embed is None else jnp.swapaxes(pos_embed, 0, 1)
        
        # TODO understand this
        attn_mask = None
        if key_padding_mask is not None:
            keep = jnp.logical_not(key_padding_mask)      # (B,S) True=valid
            attn_mask = keep[:, None, None, :]            # (B,1,1,S)
        
        skip = x_bsd
        if self.pre_norm:
            x_bsd = self.norm1(x_bsd)

        q = x_bsd if pos_embed_bsd is None else x_bsd + pos_embed_bsd
        k = q
        
        # Torch attn function outputs (out, attn_weights), Flax only outputs (out,)
        attn_out = self.self_attn(q, k, x_bsd, mask=attn_mask, deterministic=deterministic)
        x_bsd = skip + self.dropout1(attn_out, deterministic=deterministic)
        if self.pre_norm:
            skip = x_bsd
            x_bsd = self.norm2(x_bsd)
        else:
            x_bsd = self.norm1(x_bsd)
            skip = x_bsd
        x_bsd = self.linear2(self.dropout(self.activation(self.linear1(x_bsd)), deterministic=deterministic))
        x_bsd = skip + self.dropout2(x_bsd, deterministic=deterministic)
        if not self.pre_norm:
            x_bsd = self.norm2(x_bsd)
        
        return jnp.swapaxes(x_bsd, 0, 1)

class ACTDecoderJAX(nn.Module):
    config: ACTConfigJAX
    
    def setup(self):
        self.layers = [ACTDecoderLayerJAX(self.config) for _ in range(self.config.n_decoder_layers)]
        self.norm = nn.LayerNorm()
    
    def __call__(
        self,
        x, 
        encoder_out,
        decoder_pos_embed = None,
        encoder_pos_embed = None,
        *,
        deterministic: bool
    ):
        for layer in self.layers:
            x = layer(
                x,
                encoder_out,
                decoder_pos_embed=decoder_pos_embed,
                encoder_pos_embed=encoder_pos_embed,
                deterministic=deterministic,
            )
        if self.norm is not None:
            x = self.norm(x)
        return x
    
class ACTDecoderLayerJAX(nn.Module):
    config: ACTConfigJAX
    
    def setup(self):
        # Attention layers
        self.self_attn = nn.MultiHeadAttention(
            self.config.n_heads,
            dropout_rate=self.config.dropout,
            kernel_init=XAVIER,
        )
        self.multihead_attn = nn.MultiHeadAttention(
            self.config.n_heads,
            dropout_rate=self.config.dropout,
            kernel_init=XAVIER,
        )
        
        # Feedforward layers
        self.linear1 = nn.Dense(self.config.dim_feedforward, kernel_init=XAVIER, bias_init=ZEROS)
        self.dropout = nn.Dropout(rate=self.config.dropout)
        self.linear2 = nn.Dense(self.config.dim_model, kernel_init=XAVIER, bias_init=ZEROS)
        
        # Normalization layers
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.norm3 = nn.LayerNorm()
        self.dropout1 = nn.Dropout(rate=self.config.dropout)
        self.dropout2 = nn.Dropout(rate=self.config.dropout)
        self.dropout3 = nn.Dropout(rate=self.config.dropout)
        
        self.activation = get_activation_fn(self.config.feedforward_activation)
        self.pre_norm = self.config.pre_norm
        
    def maybe_add_pos_embed(self, tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed
    
    def __call__(self,
                x,
                encoder_out,
                decoder_pos_embed = None,
                encoder_pos_embed = None,
                *,
                deterministic: bool):
        # Torch works with (S, B, D), Flax with (B, S, D)
        x_bsd = jnp.swapaxes(x, 0, 1)
        
        # When no qkv_features is provided, defaults are:
        # qkv_features = inputs_q.shape[-1]
        # out_features = inputs_q.shape[-1]
        assert x_bsd.shape[-1] == self.config.dim_model
        
        encoder_out_bsd = jnp.swapaxes(encoder_out, 0, 1)
        decoder_pos_embed_bsd = None if decoder_pos_embed is None else jnp.swapaxes(decoder_pos_embed, 0, 1)
        encoder_pos_embed_bsd = None if encoder_pos_embed is None else jnp.swapaxes(encoder_pos_embed, 0, 1)
        
        # Self-attention
        skip = x_bsd
        if self.pre_norm:
            x_bsd = self.norm1(x_bsd)
        
        k = self.maybe_add_pos_embed(x_bsd, decoder_pos_embed_bsd)
        q = k
        attn_out = self.self_attn(q, k, x_bsd, deterministic=deterministic)
        x_bsd = skip + self.dropout1(attn_out, deterministic=deterministic)
        if self.pre_norm:
            skip = x_bsd
            x_bsd = self.norm2(x_bsd)
        else:
            x_bsd = self.norm1(x_bsd)
            skip = x_bsd
            
        # Cross-attention
        attn_out = self.multihead_attn(
            self.maybe_add_pos_embed(x_bsd, decoder_pos_embed_bsd),
            self.maybe_add_pos_embed(encoder_out_bsd, encoder_pos_embed_bsd),
            encoder_out_bsd,
            deterministic=deterministic,
        )
        x_bsd = skip + self.dropout2(attn_out, deterministic=deterministic)
        if self.pre_norm:
            skip = x_bsd
            x_bsd = self.norm3(x_bsd)
        else:
            x_bsd = self.norm2(x_bsd)
            skip = x_bsd
        x_bsd = self.linear2(self.dropout(self.activation(self.linear1(x_bsd)), deterministic=deterministic))
        x_bsd = skip + self.dropout3(x_bsd, deterministic=deterministic)
        
        if not self.pre_norm:
            x_bsd = self.norm3(x_bsd)

        return jnp.swapaxes(x_bsd, 0, 1)
    
# Same logic as in modeling_act.py
def create_sinusoidal_pos_embedding_1(num_positions: int, dim_model: int):
    """1D sinusoidal positional embeddings."""

    def get_position_angle_vec(position):
        init = jnp.zeros((dim_model,), dtype=jnp.float32)

        def body(i, val):
            angle = position / jnp.power(10000.0, (2.0 * (i // 2)) / dim_model)
            return val.at[i].set(angle)

        return lax.fori_loop(0, dim_model, body, init)

    sinusoid_table = jnp.stack(
        [get_position_angle_vec(pos) for pos in range(num_positions)],
        axis=0
    )  # (num_positions, dim_model)

    sinusoid_table = sinusoid_table.at[:, 0::2].set(jnp.sin(sinusoid_table[:, 0::2]))
    sinusoid_table = sinusoid_table.at[:, 1::2].set(jnp.cos(sinusoid_table[:, 1::2]))

    return sinusoid_table.astype(jnp.float32)

# Alternative implementation
def create_sinusoidal_pos_embedding_2(num_positions: int, dim_model: int):
    """1D sinusoidal positional embeddings like in Attention is All You Need."""
    positions = jnp.arange(num_positions)[:, None]              # (P, 1)
    i = jnp.arange(0, dim_model, 2)[None, :]                    # (1, D/2)  -> 2k
    div_term = jnp.exp(-jnp.log(10000.0) * (i / dim_model))     # (1, D/2)

    pe = jnp.zeros((num_positions, dim_model))
    pe = pe.at[:, 0::2].set(jnp.sin(positions * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(positions * div_term))
    pe = jnp.asarray(pe).astype(jnp.float32)
    return pe

class ACTSinusoidalPositionalEmbeddingJAX(nn.Module):
    config: ACTConfigJAX

    def setup(self):
        self.dimension = self.config.dim_model // 2
        self._two_pi = 2.0 * jnp.pi
        self._eps = 1e-6
        self._temperature = 10000.0

    def __call__(self, x):  # x: (B,H,W,C) NHWC
        _, H, W, _ = x.shape
        not_mask = jnp.ones((1, H, W), dtype=jnp.float32)

        y_range = jnp.cumsum(not_mask, axis=1)
        x_range = jnp.cumsum(not_mask, axis=2)

        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inv_freq = self._temperature ** (
            2.0 * (jnp.arange(self.dimension, dtype=jnp.float32) // 2) / self.dimension
        )

        x_range = x_range[..., None] / inv_freq  # (1,H,W,dim/2)
        y_range = y_range[..., None] / inv_freq

        pos_x = jnp.stack((jnp.sin(x_range[..., 0::2]), jnp.cos(x_range[..., 1::2])), axis=-1)
        pos_x = pos_x.reshape(pos_x.shape[:3] + (-1,))  # (1,H,W,dim/2)

        pos_y = jnp.stack((jnp.sin(y_range[..., 0::2]), jnp.cos(y_range[..., 1::2])), axis=-1)
        pos_y = pos_y.reshape(pos_y.shape[:3] + (-1,))

        pos = jnp.concatenate([pos_y, pos_x], axis=3)  # (1,H,W,dim_model)
        return pos  # NHWC

def get_activation_fn(activation: str):
    if activation == "relu":
        return nn.relu
    elif activation == "gelu":
        return nn.gelu
    raise ValueError(f"Unsupported activation function: {activation}")

# ------------------- TRAINING SMOKE TEST (SIN env_state) -------------------
if __name__ == "__main__":
    import jax
    import jax.numpy as jnp

    # ===== dims =====
    B = 2
    S = 8
    action_dim = 6
    state_dim = 10
    H, W, C = 64, 64, 3  # imagen NHWC

    # ===== config: usa TODO excepto environment =====
    # - robot_state
    # - images (2 cams)
    # - actions + action_is_pad
    # - VAE (training)
    cfg = ACTConfigJAX(
        chunk_size=S,
        use_vae=True,
        image_features=("cam0", "cam1"),  # para que _prepare_batch_images cree OBS_IMAGES
        robot_state_feature=jnp.zeros((state_dim,), dtype=jnp.float32),  # "truthy" => habilita robot_state
        env_state_feature=None,  # NO usar environment
        action_feature=jnp.zeros((action_dim,), dtype=jnp.float32),
    )

    policy = ACTPolicyJAX(cfg)

    # ===== batch dummy (SIN OBS_ENV_STATE) =====
    batch = {
        OBS_STATE: jnp.zeros((B, state_dim), dtype=jnp.float32),
        "cam0": jnp.zeros((B, H, W, C), dtype=jnp.float32),
        "cam1": jnp.zeros((B, H, W, C), dtype=jnp.float32),
        ACTION: jnp.zeros((B, S, action_dim), dtype=jnp.float32),
        # False = válido, True = padding
        "action_is_pad": jnp.array(
            [[False] * (S - 2) + [True] * 2, [False] * S],
            dtype=jnp.bool_,
        ),
    }

    # ===== init variables en modo TRAIN (deterministic=False) =====
    # IMPORTANTE: como deterministic=False y use_vae=True, necesitas RNGs para:
    # - params
    # - dropout (por nn.Dropout)
    # - sample (por VAE reparam)
    key = jax.random.PRNGKey(0)
    k_params, k_dropout, k_sample = jax.random.split(key, 3)

    batch_prepared = policy._prepare_batch_images(batch)  # crea OBS_IMAGES = [cam0, cam1]
    variables = policy.model.init(
        {"params": k_params, "dropout": k_dropout, "sample": k_sample},
        batch_prepared,
        deterministic=False,
    )
    policy.set_variables(variables)

    # ===== forward (loss) =====
    k_dropout2, k_sample2 = jax.random.split(jax.random.PRNGKey(1), 2)
    loss, loss_dict = policy.forward(batch, rng_dropout=k_dropout2, rng_sample=k_sample2)

    print("loss:", float(loss))
    print("loss_dict:", {k: float(v) for k, v in loss_dict.items()})

    # ===== sanity: predict_action_chunk en eval =====
    actions_eval = policy.predict_action_chunk(batch, deterministic=True, rngs=None)
    print("actions_eval.shape:", actions_eval.shape)  # (B,S,action_dim)
