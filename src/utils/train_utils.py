from typing import Optional

import torch
import numpy as np
import random
import pandas as pd
import torch.distributed as dist
import tqdm

from contextlib import nullcontext
from diffusers.utils import is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from transformers import (
    CLIPTextModelWithProjection,
    T5EncoderModel,
)
from torchvision.transforms import ToPILImage

if is_wandb_available():
    import wandb


########################################################################################################################
#                                       UTILS FUNCTIONS FOR TRAIN                                                      #
########################################################################################################################


VALIDATION_PROMPTS = [
    "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
    'A girl with pale blue hair and a cami tank top',
    'cute girl, Kyoto animation, 4k, high resolution',
    "Four cows in a pen on a sunny day",
    "Three dogs sleeping together on an unmade bed",
    "The interior of a mad scientists laboratory, Cluttered with science experiments, tools and strange machines, Eerie purple light, Close up, by Miyazaki",
    "A green train is coming down the tracks",
    "A photograph of the inside of a subway train. There are frogs sitting on the seats. One of them is reading a newspaper. The window shows the river in the background.",
    "a family of four posing at the Grand Canyon",
    "A high resolution photo of a donkey in a clown costume giving a lecture at the front of a lecture hall. The blackboard has mathematical equations on it. There are many students in the lecture hall.",
    "A castle made of tortilla chips, in a river made of salsa. There are tiny burritos walking around the castle",
    "A tornado made of bees crashing into a skyscraper. painting in the style of Hokusai.",
    "A raccoon wearing formal clothes, wearing a tophat and holding a cane. The raccoon is holding a garbage bag. Oil painting in the style of abstract cubism.",
    "A castle made of cardboard.",
    "a cat sitting on a stairway railing",
    "a cat drinking a pint of beer",
    "a bat landing on a baseball bat",
    "a black dog sitting between a bush and a pair of green pants standing up with nobody inside them",
    "a basketball game between a team of four cats and a team of three dogs",
    "a cat jumping in the air",
    "a book with the words ’Don’t Panic!’ written on it",
    "A bowl of soup that looks like a monster made out of plasticine",
    "two motorcycles facing each other"
]


# ----------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def log_validation(
    transformer,
    args,
    prepare_prompt_embed_from_caption,
    noise_scheduler,
    accelerator,
    logger,
    seed=None,
    offloadable_encoders=None,
    cfg_scale=0.0,
    pipeline_teacher=None,
    step=0,
):
    offloadable_encoders = offloadable_encoders or []
    
    # Set validation prompts
    validation_prompts = VALIDATION_PROMPTS
        
    # run inference
    seed = seed if seed else args.seed
    generator = torch.Generator(device=accelerator.device).manual_seed(seed)
    weight_dtype = torch.float16
    
    # Load text encoders to device
    for encoder in offloadable_encoders:
        encoder.to(accelerator.device)
        
    image_logs = []
    images_teacher = None
    for _, prompt in enumerate(tqdm.tqdm(validation_prompts, disable=(not accelerator.is_main_process))):
        prompt = [prompt] * 5

        # Teacher + refining
        if transformer is not None:
            with pipeline_teacher.transformer.disable_adapter():
                latents_teacher = pipeline_teacher(
                    prompt,
                    num_inference_steps=28,
                    guidance_scale=cfg_scale,
                    generator=generator,
                    output_type="latent"
                ).images

            prompt_embeds, pooled_prompt_embeds = prepare_prompt_embed_from_caption(
                prompt, pipeline_teacher.tokenizer, pipeline_teacher.tokenizer_2, pipeline_teacher.tokenizer_3,
                pipeline_teacher.text_encoder, pipeline_teacher.text_encoder_2, pipeline_teacher.text_encoder_3,
            )
            refining_timestep_index = torch.tensor([args.refining_timestep_index] * len(prompt_embeds)).long()
            timesteps = noise_scheduler.timesteps[refining_timestep_index].to(device=latents_teacher.device)
            images = transformer(
                latents_teacher,
                prompt_embeds,
                pooled_prompt_embeds,
                timesteps,
                return_dict=False,
            )[0].to(weight_dtype)
            images = latents_teacher - args.refining_scale * images

            latent = (images / pipeline_teacher.vae.config.scaling_factor) + pipeline_teacher.vae.config.shift_factor
            images = pipeline_teacher.vae.decode(latent, return_dict=False)[0]
            images = pipeline_teacher.image_processor.postprocess(images, output_type='pil')

            latents_teacher = (latents_teacher / pipeline_teacher.vae.config.scaling_factor) + pipeline_teacher.vae.config.shift_factor
            images_teacher = pipeline_teacher.vae.decode(latents_teacher, return_dict=False)[0]
            images_teacher = pipeline_teacher.image_processor.postprocess(images_teacher, output_type='pil')

        # Teacher only
        else:
            images = pipeline_teacher(
                prompt,
                num_inference_steps=28,
                guidance_scale=cfg_scale,
                generator=generator,
            ).images

        image_logs.append({"validation_prompt": prompt[0], "images": images})
        if images_teacher is not None and not step > args.validation_steps:
            image_logs.append({"validation_prompt": f'teacher_{prompt[0]}', "images": images_teacher})
        
    # Offload text encoders back
    for encoder in offloadable_encoders:
        encoder.cpu()
    
    torch.cuda.empty_cache()
    if accelerator.is_main_process:
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                for log in image_logs:
                    images = log["images"]
                    validation_prompt = log["validation_prompt"]
                    formatted_images = []
                    for image in images:
                        formatted_images.append(np.asarray(image.resize((512, 512))))
                    formatted_images = np.stack(formatted_images)
                    tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
            else:
                logger.warn(f"image logging not implemented for {tracker.name}")
    return images
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def distributed_sampling(
    transformer,
    args,
    val_prompt_path,
    prepare_prompt_embed_from_caption,
    noise_scheduler,
    accelerator,
    logger,
    seed=None,
    max_eval_samples=None,
    offloadable_encoders=None,
    cfg_scale=0.0,
    pipeline_teacher=None,
):
    logger.info(f"Running sampling")

    weight_dtype = torch.float16
    offloadable_encoders = offloadable_encoders or []
    
    seed = seed if seed else args.seed
    generator = torch.Generator(device=accelerator.device).manual_seed(seed)
    max_eval_samples = max_eval_samples if max_eval_samples else args.max_eval_samples
    # Prepare validation prompts
    rank_batches, rank_batches_index, all_prompts = prepare_val_prompts(
        val_prompt_path, bs=args.eval_batch_size, max_cnt=max_eval_samples
    )

    local_images = []
    local_text_idxs = []
    
    # Load text encoders to device
    for encoder in offloadable_encoders:
        encoder.to(accelerator.device)
    torch.cuda.empty_cache()
                
    for cnt, mini_batch in enumerate(tqdm.tqdm(rank_batches, disable=(not accelerator.is_main_process))):

        # Teacher + refining
        if transformer is not None:
            with pipeline_teacher.transformer.disable_adapter():
                latents_teacher = pipeline_teacher(
                    list(mini_batch),
                    num_inference_steps=28,
                    guidance_scale=cfg_scale,
                    generator=generator,
                    output_type="latent"
                ).images

            prompt_embeds, pooled_prompt_embeds = prepare_prompt_embed_from_caption(
                list(mini_batch), pipeline_teacher.tokenizer, pipeline_teacher.tokenizer_2, pipeline_teacher.tokenizer_3,
                pipeline_teacher.text_encoder, pipeline_teacher.text_encoder_2, pipeline_teacher.text_encoder_3,
            )
            refining_timestep_index = torch.tensor([args.refining_timestep_index] * len(prompt_embeds)).long()
            timesteps = noise_scheduler.timesteps[refining_timestep_index].to(device=latents_teacher.device)
            images = transformer(
                latents_teacher,
                prompt_embeds,
                pooled_prompt_embeds,
                timesteps,
                return_dict=False,
            )[0].to(weight_dtype)
            images = latents_teacher - args.refining_scale * images

            latent = (images / pipeline_teacher.vae.config.scaling_factor) + pipeline_teacher.vae.config.shift_factor
            images = pipeline_teacher.vae.decode(latent, return_dict=False)[0]
            images = pipeline_teacher.image_processor.postprocess(images, output_type='pil')

        # Teacher only
        else:
            images = pipeline_teacher(
                    list(mini_batch),
                    num_inference_steps=28,
                    guidance_scale=cfg_scale,
                    generator=generator,
            ).images

        for text_idx, global_idx in enumerate(rank_batches_index[cnt]):
            img_tensor = torch.tensor(np.array(images[text_idx].resize((512, 512))))
            local_images.append(img_tensor)
            local_text_idxs.append(global_idx)

    # Offload text encoders back
    for encoder in offloadable_encoders:
        encoder.cpu()
    torch.cuda.empty_cache()
    accelerator.wait_for_everyone()

    local_images = torch.stack(local_images).cuda()
    local_text_idxs = torch.tensor(local_text_idxs).cuda()
    print(f"[RANK {accelerator.process_index}] local_images.shape = {local_images.shape}")

    gathered_images = accelerator.gather(local_images).cpu().numpy()
    gathered_text_idxs = accelerator.gather(local_text_idxs).cpu().numpy()
    print(f"[RANK {accelerator.process_index}] gathered_images.shape = {gathered_images.shape}")

    images, prompts = [], []
    if accelerator.is_main_process:
        for image, global_idx in zip(gathered_images, gathered_text_idxs):
            images.append(ToPILImage()(image))
            prompts.append(all_prompts[global_idx])
    accelerator.wait_for_everyone()

    return images, prompts
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def prepare_val_prompts(path, bs=20, max_cnt=5000):
    df = pd.read_csv(path)
    all_text = list(df['caption'])
    all_text = all_text[:max_cnt]

    num_batches = ((len(all_text) - 1) // (bs * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = np.array_split(np.array(all_text), num_batches)
    rank_batches = all_batches[dist.get_rank():: dist.get_world_size()]

    index_list = np.arange(len(all_text))
    all_batches_index = np.array_split(index_list, num_batches)
    rank_batches_index = all_batches_index[dist.get_rank():: dist.get_world_size()]
    return rank_batches, rank_batches_index, all_text
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def encode_prompt(
    text_encoder: CLIPTextModelWithProjection,
    text_encoder_2: CLIPTextModelWithProjection,
    text_encoder_3: T5EncoderModel,
    input_ids: torch.Tensor,
    input_ids_2: torch.Tensor,
    input_ids_3: torch.Tensor,
    # In case there are precomputed text embeds - use them
    prompt_embeds: Optional[torch.Tensor] = None,
    prompt_embeds_2: Optional[torch.Tensor] = None,
    t5_prompt_embeds: Optional[torch.Tensor] = None,
    pooled_prompt_embeds: Optional[torch.Tensor] = None,
    pooled_prompt_embeds_2: Optional[torch.Tensor] = None,
    device='cuda'
):
    # Prepare CLIP prompt embeds
    if prompt_embeds is None:
        prompt_embeds = text_encoder(input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
    else:
        prompt_embeds, pooled_prompt_embeds = prompt_embeds.to(device), pooled_prompt_embeds.to(device)
    if prompt_embeds_2 is None:
        prompt_embeds_2 = text_encoder_2(input_ids_2.to(device), output_hidden_states=True)
        pooled_prompt_embeds_2 = prompt_embeds_2[0]
        prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]
    else:
        prompt_embeds_2, pooled_prompt_embeds_2 = prompt_embeds_2.to(device), pooled_prompt_embeds_2.to(device)
    # Prepare T5 prompt embeds
    if t5_prompt_embeds is None:
        t5_prompt_embeds = text_encoder_3(input_ids_3.to(device))[0]
    else:
        t5_prompt_embeds = t5_prompt_embeds.to(device)
    # Join prompt embeds
    clip_prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embeds.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embeds], dim=-2)
    pooled_prompt_embeds = torch.cat([pooled_prompt_embeds, pooled_prompt_embeds_2], dim=-1)
    return prompt_embeds, pooled_prompt_embeds
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def tokenize_captions(examples, args, tokenizer, tokenizer_2, tokenizer_3, is_train=True):
    captions = []
    for caption in examples[args.text_column]:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                f"Caption column `{args.text_column}` should contain either strings or lists of strings."
            )
    input_ids = tokenize_prompt(tokenizer, captions)
    input_ids_2 = tokenize_prompt(tokenizer_2, captions)
    input_ids_3 = tokenize_prompt(tokenizer_3, captions)
    return {
            "input_ids": input_ids,
            "input_ids_2": input_ids_2,
            "input_ids_3": input_ids_3,
        }
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def get_logit_normal_probs(num_sizes: int, loc: float, scale: float, eps: float = 1e-5) -> torch.Tensor:
    ts = torch.linspace(0, 1, steps=num_sizes)
    logit_ts = ((ts + eps) / (1 - ts + eps)).log()
    numerator = (logit_ts - loc).div(2 * scale**2).mul_(-1).exp()
    denominator = 1 / (scale * (ts + eps) * (1 - ts + eps))
    probs = numerator / denominator
    return probs
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model
# ----------------------------------------------------------------------------------------------------------------------
