from itertools import product
import os

bs_range = [1, 2, 4, 8]
compile_modes = [None, "default", "reduce-overhead", "max-autotune"]

base_command = ("CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. python profiling/run_profiling.py"
                " --architecture {architecture} --batch_size {bs}")

for architecture, bs, compile_mode in product(['lumina', 'sdxl', 'dit', 'var',],
                                              bs_range, compile_modes,
                                              ):
    run_command = base_command.format(architecture=architecture, bs=bs)
    if compile_mode is not None:
        run_command += f" --compile {compile_mode}"
    if architecture == 'var':
        run_command += " --lumina_arch --swiglu_ffn --xattn_caching"
    os.system(run_command)

# var architecture choices profiling
# for lumina, swiglu_ffn, xattn_caching, bs, compile_mode in product([False, True],
#                                                                    [False, True],
#                                                                    [False, True],
#                                                                    bs_range,
#                                                                    compile_modes,
#                                                                    ):
#     run_command = base_command.format(architecture='var', bs=bs)
#     if lumina:
#         run_command += " --lumina_arch"
#     if swiglu_ffn:
#         run_command += " --swiglu_ffn"
#     if xattn_caching:
#         run_command += " --xattn_caching"
#     if compile_mode is not None:
#         run_command += f" --compile {compile_mode}"
#     os.system(run_command)
