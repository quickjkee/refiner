import os
from argparse import ArgumentParser

import pandas as pd
import torch

from profiling.profilers import (VARProfiler, SD3Profiler, SDXLProfiler,
                                 LuminaProfiler, LlamaGenProfiler,
                                 DMD2Profiler, SDXLTurboProfiler,
)
from utils.arg_util import DTYPE_MAP

ARCHITECTURE_PROFILER_MAP = {
      'var': VARProfiler,
      'sdxl': SDXLProfiler,
      'sd3': SD3Profiler,
      'sdxl-turbo': SDXLTurboProfiler,
      'dmd2': DMD2Profiler,
      'lumina': LuminaProfiler,
      'llamagen': LlamaGenProfiler,
}

def parse_args():
    parser = ArgumentParser()
    # common args
    parser.add_argument("--architecture", required=True, nargs='+',
                        choices=ARCHITECTURE_PROFILER_MAP.keys(),
                        )
    parser.add_argument("--reso", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--compile",
                        choices=["reduce-overhead", "max-autotune", "default"],
                        )
    parser.add_argument("--dtype", choices=DTYPE_MAP.keys(), default='bf16')
    parser.add_argument("--full_profiling", action="store_true",
                        help="Profile full text-to-image generation pipeline",
                        )
    # var args
    parser.add_argument("--depth", type=int, default=30)
    parser.add_argument("--scales", type=str, default=None)
    parser.add_argument("--sampling_steps", type=int, default=28)
    parser.add_argument("--lumina_arch", action='store_true')
    parser.add_argument("--swiglu_ffn", action='store_true')
    parser.add_argument("--xattn_caching", action='store_true')
    parser.add_argument("--non_autoregressive", action='store_true')
    parser.add_argument("--disable_cfg_iter", type=int, default=10)
    # most likely not args but constants
    parser.add_argument("--nruns", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", default='cuda')
    parser.add_argument("--res_path", default="profiling/profiling_results.csv")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    res_path = args.res_path
    if os.path.exists(res_path):
        profiling_results = pd.read_csv(res_path)
    else:
        profiling_results = pd.DataFrame({'Architecture': [], 'XAttn_caching':[],
                                          'Compile': [], 'Resolution': [],
                                          'BS': [], 'Time/image, ms': [],
                                          'Full Profiling': [],
                                          })
    dtype = DTYPE_MAP[args.dtype]

    args.scales = [int(x) for x in args.scales.split(",")]
    profiler_kwargs = dict(batch_size=args.batch_size, reso=args.reso,
                           compile=args.compile, warmup=args.warmup,
                           scales=args.scales, sampling_steps=args.sampling_steps,
                           nruns=args.nruns, device=args.device, dtype=dtype,
                           generator_only=not args.full_profiling,
                           )
    for architecture in args.architecture:
        profiler_cls = ARCHITECTURE_PROFILER_MAP[architecture]
        if architecture == 'var':
            model_kwargs = dict(xattn_caching=args.xattn_caching,
                                lumina_arch=args.lumina_arch,
                                swiglu_ffn=args.swiglu_ffn,
                                depth=args.depth,
                                disable_cfg_iter=args.disable_cfg_iter,
                                self_attn_caching=not args.non_autoregressive,
                                )
            profiler_kwargs.update(model_kwargs)
        profiler = profiler_cls(**profiler_kwargs)

        runtime = profiler.run()

        result_entry = {'Architecture': profiler.model_name,
                        'XAttn_caching': args.xattn_caching, 'Compile': args.compile,
                        'Resolution': args.reso, 'BS': args.batch_size,
                        'Time/image, ms': runtime, 'Full Profiling': args.full_profiling,
                        }
        profiling_results.loc[len(profiling_results)] = pd.Series(result_entry)
        profiling_results.to_csv(res_path, index=False)
