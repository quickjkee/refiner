from tqdm import trange

import pandas as pd
import torch


class BaseProfiler:
    def __init__(self,
                 batch_size=1,
                 reso=1024,
                 compile=None,
                 scales=None,
                 generator_only=True,
                 sampling_steps=28,
                 warmup=10,
                 nruns=100,
                 device='cuda',
                 dtype=torch.bfloat16,
                 ):
        self.batch_size = batch_size
        self.reso = reso
        self.compile = compile
        self.warmup = warmup
        self.nruns = nruns
        self.device = device
        self.dtype = dtype
        self.generator_only = generator_only
        self.scales = scales
        self.sampling_steps = sampling_steps
        if not self.generator_only:
            self.prompts = pd.read_csv("eval_prompts/coco.csv")['captions']

    def init_model(self):
        raise NotImplementedError
    
    def generator_call(self, *inputs):
        raise NotImplementedError

    def full_inference_call(self, prompts):
        raise NotImplementedError

    def prepare_inputs(self):
        raise NotImplementedError
    
    @torch.inference_mode()
    def measure_run_time(self):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        inputs = self.prepare_inputs()
        for _ in range(self.warmup):
            if self.generator_only:
                for _ in range(self.sampling_steps):
                    self.generator_call(*inputs)
            else:
                self.full_inference_call(inputs)
        torch.cuda.synchronize()

        runtime = 0.0
        for _ in trange(self.nruns):
            torch.cuda.empty_cache()
            inputs = self.prepare_inputs()
            start.record()
            if self.generator_only: # 32, 48, 64, 80, 96, 128
                latents, prompt_embeds, pooled_prompt_embeds, timesteps = inputs
                for i in range(self.sampling_steps):
                    if self.scales is not None:
                        latents = torch.randn(self.batch_size, 16, self.scales[i], self.scales[i],
                                                device=self.device, dtype=self.dtype,
                                                )
                    self.generator_call(latents, prompt_embeds, pooled_prompt_embeds, timesteps)
            else:
                self.full_inference_call(inputs)
            end.record()
            torch.cuda.synchronize()
            runtime += start.elapsed_time(end)
        
        return runtime / self.nruns / self.batch_size
    
    def run(self):
        self.init_model()
        runtime = self.measure_run_time()

        return runtime
