import torch
from huggingface_hub import hf_hub_download
from diffusers import (StableDiffusionXLPipeline, DPMSolverMultistepScheduler,
                       DiffusionPipeline, StableDiffusion3Pipeline,
                       UNet2DConditionModel, LCMScheduler,
)

from profiling.profilers.base import BaseProfiler


class DiffusionProfiler(BaseProfiler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_inputs(self):
        if self.generator_only:
            latent_size = self.reso // 8
            latent = torch.randn(self.batch_size, 4, latent_size, latent_size,
                                dtype=self.dtype, device=self.device,
                                )
            timestep = torch.zeros(self.batch_size, device=self.device, dtype=self.dtype)
            prompt_embeds = torch.randn(self.batch_size, 77, 2048,
                                        dtype=self.dtype, device=self.device,
                                        )
            time_ids = [self.reso, self.reso, 0, 0, self.reso, self.reso] * self.batch_size
            added_cond_kwargs = {"text_embeds": torch.randn(self.batch_size, 1280,
                                                            dtype=self.dtype,
                                                            device=self.device,
                                                            ),
                                "time_ids": torch.tensor([time_ids], dtype=self.dtype,
                                                        device=self.device,
                                                        ),
                                }
            
            return (latent, timestep, prompt_embeds, added_cond_kwargs)
        else:
            return self.prompts.sample(self.batch_size).tolist()

    @torch.inference_mode()
    def generator_call(self, *inputs):
        latent, timestep, prompt_embeds, added_cond_kwargs = inputs
        self.model(latent,
                   timestep,
                   encoder_hidden_states=prompt_embeds,
                   added_cond_kwargs=added_cond_kwargs,
                   ).sample

    @torch.inference_mode()
    def full_inference_call(self, prompts):
        self.model.set_progress_bar_config(disable=True)
        return self.model(prompts,
                          num_inference_steps=self.sampling_steps,
                          guidance_scale=self.guidance_scale,
                          height=self.reso,
                          width=self.reso,
                          ).images


class SDXLProfiler(DiffusionProfiler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        self.model_name = "SDXL_UNet"
        self.sampling_steps = 25
        self.guidance_scale = 7.5

    def init_model(self):
        scheduler = DPMSolverMultistepScheduler.from_pretrained(self.model_id,
                                                                subfolder="scheduler",
                                                                )
        pipesdxl = StableDiffusionXLPipeline.from_pretrained(
            self.model_id,
            scheduler=scheduler,
            safety_checker=None,
            variant='fp16',
            torch_dtype=self.dtype,
        ).to(self.device)
        
        if self.generator_only:
            self.model = pipesdxl.unet
        else:
            self.model = pipesdxl
            self.model.vae.config.force_upcast = False


class SDXLTurboProfiler(SDXLProfiler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_id = "stabilityai/sdxl-turbo"
        self.model_name = "SDXL-Turbo"
        self.sampling_steps = 4
        self.guidance_scale = 0


class DMD2Profiler(DiffusionProfiler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_id = "tianweiy/DMD2"
        self.model_name = "DMD2"
        self.sampling_steps = 4
        self.guidance_scale = 0

    def init_model(self):
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        ckpt_name = "dmd2_sdxl_4step_unet_fp16.bin"
        # Load model.
        unet = UNet2DConditionModel.from_config(base_model_id,
                                                subfolder="unet",
                                                ).to(self.device, self.dtype)
        unet.load_state_dict(torch.load(hf_hub_download(self.model_id, ckpt_name),
                                        map_location='cpu'),
                                        )
        if self.generator_only:
            self.model = unet
        else:
            pipe = DiffusionPipeline.from_pretrained(base_model_id,
                                                     unet=unet,
                                                     torch_dtype=self.dtype,
                                                     variant='fp16',
                                                     ).to(self.device)
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

            self.model = pipe
            self.model.vae.config.force_upcast = False


class SD3Profiler(DiffusionProfiler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_id = "stabilityai/stable-diffusion-3.5-medium"
        self.model_name = "SD3_DiT"
        self.guidance_scale = 0.0
        self.sampling_steps = 28

    def init_model(self):
        pipesd3 = StableDiffusion3Pipeline.from_pretrained(self.model_id,
                                                           torch_dtype=self.dtype,
                                                           )
        if self.generator_only:
            self.model = pipesd3.transformer.to(self.device)
        else:
            self.model = pipesd3.to(self.device)

    def prepare_inputs(self):
        if self.generator_only:
            latent_size = self.reso // 8
            latents = torch.randn(self.batch_size, 16, latent_size, latent_size,
                                device=self.device, dtype=self.dtype,
                                )
            prompt_embeds = torch.randn(self.batch_size, 589, 4096, device=self.device,
                                        dtype=self.dtype,
                                        )
            pooled_prompt_embeds = torch.randn(self.batch_size, 2048, device=self.device,
                                            dtype=self.dtype,
                                            )
            timesteps = torch.tensor([1000], device=self.device)

            return (latents, prompt_embeds, pooled_prompt_embeds, timesteps)
        else:
            return self.prompts.sample(self.batch_size).tolist()
    
    @torch.inference_mode()
    def generator_call(self, *inputs):
        latents, prompt_embeds, pooled_prompt_embeds, timesteps = inputs
        self.model(
            hidden_states=latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )

class LuminaProfiler(DiffusionProfiler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_id = "Alpha-VLLM/Lumina-Next-SFT-diffusers"
        self.model_name = "Lumina-Next"
        self.sampling_steps = 60
        self.guidance_scale = 7.5

    def init_model(self):
        lumina_pipe = DiffusionPipeline.from_pretrained(self.model_id,
                                                        torch_dtype=self.dtype,
                                                        )
        if self.generator_only:
            self.model = lumina_pipe.transformer.to(self.device)
        else:
            self.model = lumina_pipe.to(self.device)

    def prepare_inputs(self):
        if self.generator_only:
            vae_scale_factor = 8
            prompt_len = 64
            prompt_embed_dim = 2048

            latent_channels = self.model.config.in_channels
            latents = torch.randn(self.batch_size, latent_channels,
                                  self.reso // vae_scale_factor, 
                                  self.reso // vae_scale_factor,
                                  device=self.device, dtype=self.dtype,
                                  )
            timestep = torch.tensor([1000], device=self.device)
            prompt_embeds = torch.randn(self.batch_size, prompt_len, prompt_embed_dim,
                                        device=self.device, dtype=self.dtype,
                                        )
            attention_mask = torch.ones(self.batch_size, prompt_len, device=self.device)
            image_rotary_emb = torch.ones(384, 384, self.model.head_dim // 2,
                                          dtype=torch.cfloat, device=self.device,
                                          )
            return (latents, timestep, prompt_embeds, attention_mask, image_rotary_emb)
        else:
            return self.prompts.sample(self.batch_size).tolist()

    @torch.inference_mode()
    def generator_call(self, *inputs):
        latents, timestep, prompt_embeds, attention_mask, image_rotary_emb = inputs
        self.model(
                    hidden_states=latents,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_mask=attention_mask,
                    image_rotary_emb=image_rotary_emb,
                    cross_attention_kwargs={},
                    return_dict=False,
                )[0]
