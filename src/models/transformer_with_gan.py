import torch
import torch.nn as nn
import math
import torch.nn.functional as F

from safetensors import safe_open
from functools import partial
from typing import Any, Dict, List, Optional, Union
from diffusers.utils import USE_PEFT_BACKEND, unscale_lora_layers, \
    is_torch_version, logging, scale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


########################################################################################################################
#                     TRANSFORMER WITH CLASSIFICATION HEAD FOR GAN DISTILLATION                                        #
########################################################################################################################


# Transformer with the interpolation layer (student)
# ----------------------------------------------------------------------------------------------------------------------
class PixNorm(nn.Module):

    def __init__(self, dim=None, eps=1e-4):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = self.dim or list(range(1, x.ndim))
        norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=x.dtype)
        norm = torch.add(self.eps, norm, alpha=math.sqrt(norm.numel() / x.numel()))
        return x / norm


class ResnetBlock2D(nn.Module):
    """Residual block without normalization.

    Args:
       channels (int): Channel number of intermediate features.
            Default: 64.
        residual_scale (float): Residual scale. Default: 1.0
    """

    def __init__(
            self,
            channels: int = 64,
            residual_scale: float = 1.0,
            norm: Optional[str] = None
    ):
        assert norm in [None, "rms"]
        super(ResnetBlock2D, self).__init__()
        self.residual_scale = residual_scale
        self.conv_res1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.conv_res2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        # define activation fn
        self.act = partial(F.leaky_relu, negative_slope=0.2)
        # define normalization fn
        self.norm = nn.Identity()
        if norm == "rms":
            self.norm = PixNorm(dim=1)
        # zero init last layer
        nn.init.zeros_(self.conv_res2.weight)
        nn.init.zeros_(self.conv_res2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.act(self.norm(self.conv_res1(x)))
        x = self.conv_res2(x)
        return skip + self.residual_scale * x


class SRResNet(nn.Module):

    def __init__(
            self,
            # Model params
            in_channels: int = 3,
            out_channels: int = 3,
            base_channels: int = 64,
            num_res_blocks: int = 1,
            residual_scale: float = 1.0,
            norm: Optional[str] = None,
    ):
        super(SRResNet, self).__init__()
        # Residual blocks
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        self.res_blocks = nn.ModuleList(
            [ResnetBlock2D(base_channels, residual_scale, norm)
             for _ in range(num_res_blocks)]
        )
        self.out_conv = nn.Conv2d(base_channels, out_channels, 3, 1, 1)
        # define activation fn
        self.act = partial(F.leaky_relu, negative_slope=0.2)
        # zero init last layer
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.act(self.in_conv(x))
        for block in self.res_blocks:
            x = block(x)
        x = self.act(x)
        x = skip + self.act(self.out_conv(x))
        return x

class Phi(nn.Conv2d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=ks,
            stride=1,
            padding=ks // 2,
        )
        self.resi_ratio = abs(quant_resi)

    def forward(self, h_BChw):
        return h_BChw.mul(1 - self.resi_ratio) + super().forward(h_BChw).mul_(
            self.resi_ratio
        )

class TransformerUp(nn.Module):
    def __init__(self, args, transformer):
        super().__init__()
        self.transformer = transformer
        self.path_to_upscalers = args.path_to_upscalers

        if self.path_to_upscalers:
            scales = sorted(set(args.scales))
            self.layer_idxs = {k: v for k, v in zip(scales, range(len(scales))) if k != 128}
            n_keys = self.layer_idxs.keys()

            self.cnn_layers = []
            for key in n_keys:
                layer = SRResNet(16, 16, 64, 16, 1.0, 'rms')
                path_to_layer_weights = f'{args.path_to_upscalers}/{key}/model.safetensors'
                print(f'Loading weights from {path_to_layer_weights}')
                layer_weights = self._load_safetensor(path_to_layer_weights)
                layer.load_state_dict(layer_weights)
                self.cnn_layers.append(layer)
            self.cnn_layers = nn.ModuleList(self.cnn_layers)
            self.cnn_layers.requires_grad_(False)

    def _load_safetensor(self, path):
        tensors = {}
        with safe_open(path, framework="pt", device=0) as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        return tensors

    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)

    def upscale(self, x, current_size, next_size):
        x = torch.nn.functional.interpolate(x, size=next_size, mode='bicubic')
        if self.path_to_upscalers and current_size != next_size:
            cnn_layer = self.cnn_layers[self.layer_idxs[current_size]]
            x = cnn_layer(x)
        return x
# ----------------------------------------------------------------------------------------------------------------------


# Transformer with the classification head (the teacher)
# ----------------------------------------------------------------------------------------------------------------------
def FeedForward(dim, outdim=None, mult=1):
    if outdim is None:
        outdim = dim
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.GELU(),
        nn.Linear(dim, outdim),
    )

class TransformerCls(nn.Module):
    def __init__(self, args, teacher_transformer):
        super().__init__()
        self.teacher_transformer = teacher_transformer

        dimensions = torch.linspace(
            teacher_transformer.inner_dim, 1, args.num_discriminator_layers + 1,
            dtype=int
        )
        self.list_of_layers = []
        for j, dim in enumerate(dimensions[:-1]):
            self.list_of_layers.append(FeedForward(dim.item(), dimensions[j+1].item()))
        self.cls_pred_branch = nn.Sequential(*self.list_of_layers)

        #self.teacher_transformer.requires_grad_(False)
        self.cls_pred_branch.requires_grad_(True)
        num_cls_params = sum(p.numel() for p in self.cls_pred_branch.parameters())
        logger.info(f'Classification head number of trainable params: {num_cls_params}')

    def forward(self, *args, **kwargs):
        return self.teacher_transformer(*args, **kwargs)

def forward_with_classify(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        skip_layers: Optional[List[int]] = None,
        classify_index_block: list = [1000],
        return_only_features = True,
        return_features = True,
) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
    """
    The [`SD3Transformer2DModel`] forward method.

    Args:
        hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
            Input `hidden_states`.
        encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
            Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
        pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`):
            Embeddings projected from the embeddings of input conditions.
        timestep (`torch.LongTensor`):
            Used to indicate denoising step.
        block_controlnet_hidden_states (`list` of `torch.Tensor`):
            A list of tensors that if specified are added to the residuals of transformer blocks.
        joint_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
            tuple.
        skip_layers (`list` of `int`, *optional*):
            A list of layer indices to skip during the forward pass.

    Returns:
        If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
        `tuple` where the first element is the sample tensor.
    """
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

    height, width = hidden_states.shape[-2:]

    hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
    temb = self.time_text_embed(timestep, pooled_projections)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states, ip_temb = self.image_proj(ip_adapter_image_embeds, timestep)

        joint_attention_kwargs.update(ip_hidden_states=ip_hidden_states, temb=ip_temb)

    hidden_states_collect = []
    for index_block, block in enumerate(self.transformer_blocks):
        # Skip specified layers
        is_skip = True if skip_layers is not None and index_block in skip_layers else False

        if torch.is_grad_enabled() and self.gradient_checkpointing and not is_skip:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                encoder_hidden_states,
                temb,
                joint_attention_kwargs,
                **ckpt_kwargs,
            )
        elif not is_skip:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        # controlnet residual
        if block_controlnet_hidden_states is not None and block.context_pre_only is False:
            interval_control = len(self.transformer_blocks) / len(block_controlnet_hidden_states)
            hidden_states = hidden_states + block_controlnet_hidden_states[int(index_block / interval_control)]

        if classify_index_block[0] > 0 and index_block in classify_index_block:
            hidden_states_collect.append(hidden_states)
            if index_block == classify_index_block[-1] and return_only_features:
                return hidden_states_collect

    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)

    # unpatchify
    patch_size = self.config.patch_size
    height = height // patch_size
    width = width // patch_size

    hidden_states = hidden_states.reshape(
        shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
    )
    hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
    output = hidden_states.reshape(
        shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
    )

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        output = (output,)
    else:
        output = output

    if return_features:
        return output, hidden_states_collect
    else:
        return output
# ----------------------------------------------------------------------------------------------------------------------
