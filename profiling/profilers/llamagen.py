import pandas as pd
import torch

from profiling.profilers.base import BaseProfiler
from models.llamagen.generate import generate
from models.llamagen.gpt import GPT_models

class LlamaGenProfiler(BaseProfiler):
    def __init__(self, batch_size=1, reso=512, compile=None, warmup=10,
                 nruns=100, device='cuda', dtype=torch.bfloat16,
                 # llamagen params
                 gpt_model="GPT-XL", downsample_size=16, cls_token_num=120,
                 ):
        super().__init__(batch_size=batch_size, reso=reso, compile=compile,
                         warmup=warmup, nruns=nruns, device=device, dtype=dtype,
                         )
        self.model_name = "LlamaGen"
        self.latent_size = self.reso // downsample_size
        self.gpt_model = gpt_model
        self.cls_token_num = cls_token_num
        batch = torch.load('llamagen_train_batch.pt', map_location='cpu')
        self.batch = {k: v[:self.batch_size] for k, v in batch.items()}
        self.model = self.init_model()

    def init_model(self):
        self.gpt_model = GPT_models[self.gpt_model](
            block_size=self.latent_size ** 2,
            cls_token_num=self.cls_token_num,
            model_type="t2i",
        ).to(device=self.device, dtype=self.dtype)
        self.gpt_model.eval()
        if self.generator_only:
            return
        else:
            self.vq_model = VQ_models[self.vq_model](
            codebook_size=self.codebook_size,
            codebook_embed_dim=self.codebook_embed_dim)
            self.vq_model.to(self.device, self.dtype)
            self.vq_model.eval()
            
            self.t5_model = T5Embedder(
            local_cache=True,
            device=self.device,
            cache_dir=self.t5_path, 
            dir_or_name=self.t5_model_type,
            torch_dtype=self.dtype,
            model_max_length=self.t5_feature_max_len,
        )
            self.t5_model.eval()

    @torch.inference_mode()
    def generator_call(self, *inputs):
        c_indices, c_emb_masks = inputs
        generate(
                self.gpt_model, c_indices, self.latent_size ** 2, 
                c_emb_masks, sample_logits=False,
                )
        
    def prepare_inputs(self):
        if self.generator_only:
            caption_embs, emb_masks = self.batch['caption_embs'], self.batch['emb_masks']
            caption_embs = torch.randn_like(caption_embs)
            c_indices = caption_embs * emb_masks[:,:, None]

            return (c_indices.to(self.device, self.dtype),
                    emb_masks.to(self.device, self.dtype))
        else:
            return self.prompts.sample(self.batch_size).tolist()
