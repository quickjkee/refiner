import torch
from accelerate.logging import get_logger
from pathlib import Path
from diffusers import StableDiffusion3Pipeline

from src.utils.train_utils import distributed_sampling
from distill_sd3_scalewise import prepare_models, prepare_accelertor, prepare_prompt_embed_from_caption
from src.utils.metrics import calculate_scores

logger = get_logger(__name__)


@torch.no_grad()
def validate_teacher(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = prepare_accelertor(args, logging_dir)

    pipeline_teacher = StableDiffusion3Pipeline.from_pretrained(args.pretrained_model_name_or_path,
                                                                torch_dtype=torch.bfloat16).to('cuda')
    pipeline_teacher.set_progress_bar_config(disable=True)

    images, prompts = distributed_sampling(None, args, f'prompts/mjhq.csv',
                                           prepare_prompt_embed_from_caption, None, None,
                                           accelerator, logger, 0,
                                           max_eval_samples=args.max_eval_samples,
                                           seed=args.seed,
                                           pipeline_teacher=pipeline_teacher,
                                           cfg_scale=args.cfg_teacher)
    additional_images = []
    if args.calc_diversity:
        for seed in [0, 1, 2, 3]:
                images, _ = distributed_sampling(None, args, f'prompts/mjhq.csv',
                                           prepare_prompt_embed_from_caption, None, None,
                                           accelerator, logger, 0,
                                           max_eval_samples=args.max_eval_samples,
                                           seed=seed,
                                           pipeline_teacher=pipeline_teacher,
                                           cfg_scale=args.cfg_teacher)
                additional_images.append(images)


    if accelerator.is_main_process:
        image_reward, pick_score, clip_score, hpsv_reward, fid_score, div_score = calculate_scores(
            args,
            images,
            prompts,
            ref_stats_path=args.mjhq_ref_stats_path,
            additional_images=additional_images
        )
        logs = {
            f"fid": fid_score.item(),
            f"pick_score": pick_score.item(),
            f"clip_score": clip_score.item(),
            f"image_reward": image_reward.item(),
            f"hpsv_reward": hpsv_reward.item(),
            f'diversity_score': div_score.item(),
        }
        print(logs)
