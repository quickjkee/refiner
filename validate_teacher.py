import torch
from accelerate.logging import get_logger
from pathlib import Path

from src.utils.train_utils import distributed_sampling
from distill_sd3_scalewise import prepare_models, prepare_accelertor, prepare_prompt_embed_from_caption
from src.utils.flow_matching_sampler import FlowMatchingSolver
from src.pipelines.stable_diffusion_3 import ScaleWiseStableDiffusion3Pipeline
from src.utils.train_utils import unwrap_model
from src.utils.metrics import calculate_scores

logger = get_logger(__name__)


@torch.no_grad()
def validate_teacher(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = prepare_accelertor(args, logging_dir)

    pipeline_teacher = StableDiffusion3Pipeline.from_pretrained(args.pretrained_model_name_or_path,
                                                                torch_dtype=torch.bfloat16)
    images, prompts = distributed_sampling(pipeline, args, f'prompts/mjhq.csv',
                                           prepare_prompt_embed_from_caption, fm_solver, noise_scheduler,
                                           accelerator, logger, 0,
                                           pipeline_teacher=pipeline_teacher,
                                           cfg_scale=args.cfg_teacher)

    if accelerator.is_main_process:
        image_reward, pick_score, clip_score, fid_score = calculate_scores(
            args,
            images,
            prompts,
            ref_stats_path=args.mjhq_ref_stats_path,
        )
        logs = {
            f"fid": fid_score.item(),
            f"pick_score": pick_score.item(),
            f"clip_score": clip_score.item(),
            f"image_reward": image_reward.item(),
        }
        print(logs)
