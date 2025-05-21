import torch
import torch.nn.functional as F


########################################################################################################################
#                            THE LOSSES NEEDED FOR THE DISTILLATION OF SD3                                            #
########################################################################################################################


# ----------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def sample_from_student(transformer, solver, noise_scheduler, idx_end,
                        prompt_embeds, uncond_prompt_embeds,
                        uncond_pooled_prompt_embeds, pooled_prompt_embeds,
                        accelerator, args):

    generator = None
    sigmas = noise_scheduler.sigmas[solver.boundary_idx]
    timesteps = noise_scheduler.timesteps[solver.boundary_start_idx]
    idx_start = torch.tensor([0] * len(prompt_embeds))
    if idx_end == 0:
        idx_end += 1
    idx_end = torch.tensor([idx_end] * len(prompt_embeds))

    sampling_fn = solver.flow_matching_sampling_stochastic if args.stochastic_case else solver.flow_matching_sampling
    images = sampling_fn(
                        transformer.module,
                        torch.randn((len(prompt_embeds), 16, 128, 128), generator=generator).to(accelerator.device),
                        prompt_embeds, pooled_prompt_embeds,
                        uncond_prompt_embeds, uncond_pooled_prompt_embeds,
                        idx_start, idx_end,
                        cfg_scale=0.0, do_scales=True if args.scales else False,
                        sigmas=sigmas, timesteps=timesteps, generator=generator
                        )

    return images
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def gan_loss_fn(cls_head, inner_features_fake, inner_features_true=None):
    logits_fake = 0
    for x in inner_features_fake:
        logits_fake += cls_head(x.float().mean(dim=1))
    logits_fake /= len(inner_features_fake)

    if inner_features_true is not None:
        logits_true = 0
        for x in inner_features_true:
            logits_true += cls_head(x.float().mean(dim=1))
        logits_true /= len(inner_features_true)

        classification_loss = F.softplus(logits_fake).mean() + F.softplus(-logits_true).mean()
    else:
        classification_loss = F.softplus(-logits_fake).mean()

    return classification_loss
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def dmd_loss(
        transformer, transformer_fake, transformer_teacher,
        prompt_embeds, pooled_prompt_embeds,
        model_input, timesteps,
        optimizer, lr_scheduler, params_to_optimize,
        weight_dtype, noise_scheduler,
        accelerator, args
):
    ## STEP 1. Make a prediction with the student to create fake samples
    ## ---------------------------------------------------------------------------
    optimizer.zero_grad(set_to_none=True)
    transformer.train()
    transformer_fake.eval()

    model_pred = transformer(
        model_input,
        prompt_embeds,
        pooled_prompt_embeds,
        timesteps,
        return_dict=False,
    )[0]
    sigma_start = noise_scheduler.sigmas[args.refining_timestep_index].to(device=model_pred.device)[:, None, None, None]
    fake_sample = model_input - sigma_start * model_pred

    ## Apply noise to the boundary points for the fake,
    idx_noisy = torch.randint(0, len(noise_scheduler.timesteps), (len(fake_sample),))
    sigma_noisy = noise_scheduler.sigmas[idx_noisy].to(device=model_pred.device)[:, None, None, None]
    timesteps_noisy = noise_scheduler.timesteps[idx_noisy].to(device=model_input.device)

    noise = torch.randn_like(fake_sample)
    noisy_fake_sample = noise_scheduler.scale_noise(fake_sample, timesteps_noisy, noise)
    ## ---------------------------------------------------------------------------

    ## STEP 2. Calculate DMD loss
    ## ---------------------------------------------------------------------------
    with torch.no_grad(), torch.autocast("cuda", dtype=weight_dtype), transformer_teacher.disable_adapter():
        real_pred = transformer_teacher(
            noisy_fake_sample,
            prompt_embeds,
            pooled_prompt_embeds,
            timesteps_noisy,
            return_dict=False,
        )[0]
        
        if args.cfg_teacher > 1.0:
            real_pred_uncond = transformer_teacher(
                noisy_fake_sample,
                uncond_prompt_embeds,
                uncond_pooled_prompt_embeds,
                timesteps_noisy,
                return_dict=False,
            )[0]
            real_pred = real_pred_uncond + args.cfg_teacher * (real_pred - real_pred_uncond)
        real_pred_x0 = noisy_fake_sample - sigma_noisy * real_pred

        fake_pred = transformer_fake(
            noisy_fake_sample,
            prompt_embeds,
            pooled_prompt_embeds,
            timesteps_noisy,
            return_features=False,
            return_dict=False
        )[0]

        if args.cfg_fake > 1.0:
            fake_pred_uncond = transformer_fake(
                noisy_fake_sample,
                uncond_prompt_embeds,
                uncond_pooled_prompt_embeds,
                timesteps_noisy,
                return_features=False,
                return_dict=False,
            )[0]
            fake_pred = fake_pred_uncond + args.cfg_fake * (fake_pred - fake_pred_uncond)
        fake_pred_x0 = noisy_fake_sample - sigma_noisy * fake_pred

        weight_factor = abs(fake_sample.to(torch.float32) - real_pred_x0.to(torch.float32)) \
            .mean(dim=[1, 2, 3], keepdim=True).clip(min=0.00001)

    loss = (fake_pred_x0 - real_pred_x0) * noisy_fake_sample / weight_factor
    loss = torch.mean(loss)
    ## ---------------------------------------------------------------------------

    ## STEP 3. Calculate GAN loss
    ## ---------------------------------------------------------------------------
    trainable_keys = [n for n, p in transformer_fake.named_parameters() if p.requires_grad]
    transformer_fake.requires_grad_(False).eval()

    inner_features_fake = transformer_fake(
                                      noisy_fake_sample,
                                      prompt_embeds,
                                      pooled_prompt_embeds,
                                      timesteps_noisy,
                                      return_dict=False,
                                      classify_index_block=args.cls_blocks,
                                      return_only_features=True
                                    )
    gan_loss = gan_loss_fn(transformer_fake.module.cls_pred_branch,
                           inner_features_fake,
                           inner_features_true=None)

    loss += gan_loss * args.gen_cls_loss_weight
    ## ---------------------------------------------------------------------------

    ## STEP 4. Calculate GAN loss
    ## ---------------------------------------------------------------------------
    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
    avg_loss += avg_loss.item() / args.gradient_accumulation_steps

    ## Backpropagate
    accelerator.backward(loss)
    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    ## ---------------------------------------------------------------------------

    transformer_fake.module.cls_pred_branch.requires_grad_(True).train()
    for n, p in transformer_fake.named_parameters():
        if n in trainable_keys:
            p.requires_grad_(True)

    return avg_loss
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def fake_diffusion_loss(
        transformer, transformer_fake,
        prompt_embeds, pooled_prompt_embeds,
        model_input, timesteps, target,
        optimizer, lr_scheduler, params_to_optimize,
        weight_dtype, noise_scheduler,
        accelerator, args
):
    ## STEP 1. Make the prediction with the student to create fake samples
    ## ---------------------------------------------------------------------------
    optimizer.zero_grad(set_to_none=True)
    transformer_fake.train()
    transformer.eval()

    with torch.no_grad(), torch.autocast("cuda", dtype=weight_dtype):
        model_pred = transformer(
            model_input,
            prompt_embeds,
            pooled_prompt_embeds,
            timesteps,
            return_dict=False,
        )[0]
    sigma_start = noise_scheduler.sigmas[args.refining_timestep_index].to(device=model_pred.device)[:, None, None, None]
    fake_sample = model_input - sigma_start * model_pred

    ## Apply noise to the boundary points for the fake sample
    idx_noisy = torch.randint(0, len(noise_scheduler.timesteps), (len(fake_sample),))
    timesteps_noisy = noise_scheduler.timesteps[idx_noisy].to(device=model_input.device)
    sigma_noisy = noise_scheduler.sigmas[idx_noisy].to(device=model_pred.device)[:, None, None, None]

    noise = torch.randn_like(fake_sample)
    noisy_fake_sample = noise_scheduler.scale_noise(fake_sample, timesteps_noisy, noise)
    ## ---------------------------------------------------------------------------

    ## STEP 2. Predict with fake net and calc diffusion loss
    ## ---------------------------------------------------------------------------
    fake_pred, inner_features_fake = transformer_fake(noisy_fake_sample,
                                                      prompt_embeds,
                                                      pooled_prompt_embeds,
                                                      timesteps_noisy,
                                                      classify_index_block=args.cls_blocks,
                                                      return_only_features=False,
                                                      return_dict=False)
    fake_pred_x0 = noisy_fake_sample - sigma_noisy * fake_pred[0]
    loss = F.mse_loss(fake_pred_x0.float(), fake_sample.float(), reduction="mean")
    ## ---------------------------------------------------------------------------

    ## STEP 3. Calculate real features and gan loss
    ## ---------------------------------------------------------------------------
    noisy_true_sample = noise_scheduler.scale_noise(target, timesteps_noisy, noise)
    inner_features_true = transformer_fake(noisy_true_sample,
                                           prompt_embeds,
                                           pooled_prompt_embeds,
                                           timesteps_noisy,
                                           classify_index_block=args.cls_blocks,
                                           return_only_features=True,
                                           return_dict=False)
    gan_loss = gan_loss_fn(transformer_fake.module.cls_pred_branch,
                           inner_features_fake,
                           inner_features_true)
    loss += gan_loss * args.guidance_cls_loss_weight
    ## ---------------------------------------------------------------------------

    ## STEP 3. Calculate diffusion loss
    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
    avg_loss += avg_loss.item() / args.gradient_accumulation_steps

    ## Backpropagate
    accelerator.backward(loss)
    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    ## ---------------------------------------------------------------------------

    transformer.train()
    transformer_fake.eval()
    return avg_loss
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def pdm_loss(
        transformer, transformer_fake,
        prompt_embeds, pooled_prompt_embeds,
        model_input, timesteps, target,
        optimizer, lr_scheduler, params_to_optimize,
        weight_dtype, noise_scheduler,
        accelerator, args
):
    ## STEP 1. Make the prediction with the student to create fake samples
    ## ---------------------------------------------------------------------------
    optimizer.zero_grad(set_to_none=True)
    transformer.train()

    model_pred = transformer(
        model_input,
        prompt_embeds,
        pooled_prompt_embeds,
        timesteps,
        return_dict=False,
    )[0]

    sigma_start = noise_scheduler.sigmas[args.refining_timestep_index].to(device=model_pred.device)[:, None, None, None]
    fake_sample = model_input - sigma_start * model_pred
    true_sample = target
    ## ---------------------------------------------------------------------------

    ## STEP 2. Apply noise and extract features
    ## ---------------------------------------------------------------------------
    idx_noisy = torch.randint(18, 25, (len(fake_sample),))
    timesteps_noisy = noise_scheduler.timesteps[idx_noisy].to(device=model_input.device)

    noise = torch.randn_like(fake_sample)
    noisy_true_sample = noise_scheduler.scale_noise(true_sample, timesteps_noisy, noise)
    noisy_fake_sample = noise_scheduler.scale_noise(fake_sample, timesteps_noisy, noise)

    trainable_keys = [n for n, p in transformer_fake.named_parameters() if p.requires_grad]
    transformer_fake.requires_grad_(False).eval()

    inner_features_fake = transformer_fake(
        noisy_fake_sample,
        prompt_embeds,
        pooled_prompt_embeds,
        timesteps_noisy,
        return_dict=False,
        classify_index_block=args.pdm_blocks
    )[0]
    inner_features_true = transformer_fake(
        noisy_true_sample,
        prompt_embeds,
        pooled_prompt_embeds,
        timesteps_noisy,
        return_dict=False,
        classify_index_block=args.pdm_blocks
    )[0]
    ## ---------------------------------------------------------------------------

    ## STEP 3. Calculate DM loss and update the generator
    ## ---------------------------------------------------------------------------
    inner_features_true = inner_features_true.mean(dim=(0, 1))
    inner_features_fake = inner_features_fake.mean(dim=(0, 1))

    c = args.huber_c
    loss = torch.sqrt((inner_features_true.float() - inner_features_fake.float()) ** 2 + c ** 2) - c
    loss = torch.mean(loss)
        
    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
    avg_loss += avg_loss.item() / args.gradient_accumulation_steps

    ## Backpropagate
    accelerator.backward(loss)
    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    transformer_fake.module.cls_pred_branch.requires_grad_(True).train()
    for n, p in transformer_fake.named_parameters():
        if n in trainable_keys:
            p.requires_grad_(True)
    ## ---------------------------------------------------------------------------

    return avg_loss
# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
def mmd_loss(x, y, sigma=200):
    alpha = 1 / (2 * sigma**2)

    xx = torch.bmm(x, x.permute(0, 2, 1))
    yy = torch.bmm(y, y.permute(0, 2, 1))
    xy = torch.bmm(x, y.permute(0, 2, 1))

    rx = torch.diagonal(xx, dim1=1, dim2=2).unsqueeze(1).expand_as(xx)
    ry = torch.diagonal(yy, dim1=1, dim2=2).unsqueeze(1).expand_as(yy)

    k_xx = torch.exp(- alpha * (rx.permute(0, 2, 1) + rx - 2*xx)).mean()
    k_xy = torch.exp(- alpha * (rx.permute(0, 2, 1) + ry - 2*xy)).mean()
    k_yy = torch.exp(- alpha * (ry.permute(0, 2, 1) + ry - 2*yy)).mean()

    return 100 * (k_xx + k_yy - 2 * k_xy)
# ----------------------------------------------------------------------------------------------------------------------
