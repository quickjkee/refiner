import torch
from tqdm import tqdm

########################################################################################################################
#                                            SAMPLER UTILS                                                             #
########################################################################################################################

# ----------------------------------------------------------------------------------------------------------------------
class FlowMatchingSolver:

    ## ---------------------------------------------------------------------------
    def __init__(
        self,
        noise_scheduler,
    ):
        self.noise_scheduler = noise_scheduler
    ## ---------------------------------------------------------------------------


    ## ---------------------------------------------------------------------------
    def flow_matching_single_step(self, sample, model_output, sigma, sigma_next):
        prev_sample = sample + (sigma_next - sigma) * model_output
        return prev_sample
    ## ---------------------------------------------------------------------------


    ## ---------------------------------------------------------------------------
    @torch.no_grad()
    def flow_matching_sampling(
        self, model, latent,
        prompt_embeds, pooled_prompt_embeds,
        uncond_prompt_embeds, uncond_pooled_prompt_embeds,
        cfg_scale=7.0,
    ):
        sigmas = self.noise_scheduler.sigmas
        timesteps = self.noise_scheduler.timesteps
        idx_start = torch.tensor([0] * len(prompt_embeds))
        idx_end = torch.tensor([len(sigmas) - 1] * len(prompt_embeds))

        while True:
            timestep = timesteps[idx_start].to(device=model.device)
            sigma = sigmas[idx_start].to(device=model.device)
            sigma_next = sigmas[idx_start + 1].to(device=model.device)

            with torch.autocast("cuda", dtype=torch.float16):
                noise_pred = model(
                    latent,
                    prompt_embeds,
                    pooled_prompt_embeds,
                    timestep,
                    return_dict=False,
                )[0]

                if cfg_scale > 1.0:
                    noise_pred_uncond = model(
                        latent,
                        uncond_prompt_embeds,
                        uncond_pooled_prompt_embeds,
                        timestep,
                        return_dict=False,
                    )[0]
                    noise_pred = noise_pred_uncond + cfg_scale * (noise_pred - noise_pred_uncond)

            latent = self.flow_matching_single_step(latent, noise_pred,
                                                    sigma[:, None, None, None],
                                                    sigma_next[:, None, None, None])

            if (idx_start + 1)[0].item() == idx_end[0].item():
                break
            idx_start = idx_start + 1

        return latent
    ## ---------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------