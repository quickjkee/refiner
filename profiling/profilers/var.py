import torch
import pandas as pd

from models import build_vae_var
from models.clip import FrozenCLIPEmbedder
from profiling.profilers.base import BaseProfiler


class VARProfiler(BaseProfiler):
    def __init__(self, batch_size=1, reso=512, compile=None,
                 warmup=10, nruns=100, device='cuda', dtype=torch.bfloat16,
                 generator_only=True,
                 xattn_caching=True, lumina_arch=True, swiglu_ffn=True, depth=30,
                 self_attn_caching=False, disable_cfg_iter=10,
                 ):
        super().__init__(batch_size=batch_size, reso=reso, compile=compile,
                         warmup=warmup, nruns=nruns, device=device, dtype=dtype,
                         generator_only=generator_only,
                         )
        self.xattn_caching = xattn_caching
        self.self_attn_caching = self_attn_caching
        self.lumina_arch = lumina_arch
        self.swiglu_ffn = swiglu_ffn or lumina_arch
        self.disable_cfg_iter = disable_cfg_iter
        self.depth = depth
        self.model_name = "TVAR" if self_attn_caching else "SWiNAR"
        if lumina_arch:
            self.model_name += "_Lumina"
        if swiglu_ffn:
            self.model_name += "_SwiGLUFFN"
        if self.disable_cfg_iter < 14:
            self.model_name += "_NoCFG"

        if self.generator_only:
            batch = torch.load('stats/eval_prompts.pt', map_location='cpu')
            self.batch = {k: v[:self.batch_size] for k, v in batch.items()}
        else:
            self.prompts = pd.read_csv("eval_prompts/coco.csv")['captions']

    def init_model(self):
        if self.reso == 256:
            patch_nums = "1_2_3_4_5_6_8_10_13_16"
        elif self.reso == 512:
            patch_nums = "1_2_3_4_6_9_13_18_24_32"
        elif self.reso == 1024:
            patch_nums = "1_2_3_4_5_7_9_12_16_21_27_36_48_64"
        elif self.reso == 2048:
            patch_nums = "1_2_3_4_5_7_9_12_18_24_36_48_64_96_128"
        self.patch_nums = tuple(map(int, patch_nums.split("_")))

        vae, var = build_vae_var(
                device=self.device,
                patch_nums=self.patch_nums,
                depth=self.depth,
                attn_l2_norm=True,
                init_adaln=0.5,
                init_adaln_gamma=1e-6,
                init_head=0.02,
                init_std=-1,
                text_encoder_path=None,
                text_encoder_2_path=None,
                rope=True,
                rope_theta=10000,
                rope_size=128,
                dpr=0.,
                use_swiglu_ffn=self.swiglu_ffn,
                use_ar=self.self_attn_caching,
            )
        vae.load_state_dict(torch.load("vae_ft_checkpoint.pt", map_location='cpu'), strict=True)
        if not self.generator_only:
            var.text_encoder = FrozenCLIPEmbedder("openai/clip-vit-large-patch14",
                                                  device=self.device,
                                                  ).eval().to(self.dtype)
            var.text_encoder_2 = FrozenCLIPEmbedder("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                                                    device=self.device,
                                                    ).eval().to(self.dtype)
        if self.compile:
            torch.compile(var, mode=self.compile)
        var.to(self.dtype)
        vae.to(self.dtype)

        self.model = var.eval()

    @torch.inference_mode()
    def generator_call(self, *inputs)-> torch.Tensor:
        """
        reduced VAR inference, without VQVAE next scale prediction stage.
        returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        context, cond_vector, context_attn_bias = inputs
        B = context.shape[0]
    
        cond_vector = self.model.text_pooler(cond_vector)
        cond_BD = cond_vector

        lvl_pos = self.model.lvl_embed(self.model.lvl_1L)
        if not self.model.rope:
            lvl_pos += self.model.pos_1LC

        cur_L = 0
        for b in self.model.blocks:
            if self.self_attn_caching:
                b.attn.kv_caching(True)
            if self.xattn_caching:
                b.cross_attn.kv_caching(True)

        for _, pn in enumerate(self.patch_nums):  # si: i-th segment
            x_BLC_level = torch.randn(B, pn * pn, 1920,
                                      dtype=self.dtype, device=self.device)

            if self.model.rope:
                freqs_cis=self.model.freqs_cis[:, cur_L: cur_L + pn * pn]
            else:
                freqs_cis=self.model.freqs_cis
                
            for block in self.model.blocks:
                x_BLC_level = block(
                    x=x_BLC_level, cond_BD=cond_BD, attn_bias=None,
                    context=context, context_attn_bias=context_attn_bias,
                    freqs_cis=freqs_cis,
                )
            cur_L += pn * pn
            
        for b in self.model.blocks:
            b.attn.kv_caching(False)
            b.cross_attn.kv_caching(False)

    @torch.inference_mode()
    def full_inference_call(self, prompts):
        return self.model.autoregressive_infer_cfg(
            batch={"text": prompts},
            cfg=4,
            top_k=400,
            top_p=0.95,
            more_smooth=False,
            turn_off_cfg_start_si=self.disable_cfg_iter,
        )
    
    def prepare_inputs(self):
        if self.generator_only:
            context, cond_vector, context_attn_bias = self.model.parse_batch(self.batch)
            context = torch.randn_like(context)
            cond_vector = torch.randn_like(cond_vector)

            return (context.to(self.dtype), cond_vector.to(self.dtype),
                    context_attn_bias.to(self.dtype))
        else:
            return self.prompts.sample(self.batch_size).tolist()
