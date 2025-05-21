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
        num_boundaries=1,
        scales=None,
        boundaries=None
    ):
        """
        Set up boundaries indexes.
        For example, if num_boundaries = 3
        boundary_start_idx = [ 0,  9, 18], boundary_end_idx = [9, 18, 28]

        :param noise_scheduler: scheduler from the diffusers
        :param num_boundaries: (int)
        """
        if scales:
            assert len(scales) == num_boundaries
            self.min_scale = scales[0]
            scales = torch.tensor(scales)
            self.scales_pixels = scales * 8
        self.scales = scales

        self.num_boundaries = num_boundaries
        if num_boundaries == 0:
            boundary_idx = torch.tensor([0])
            self.boundary_start_idx = boundary_idx
        else:
            if boundaries is None:
                self.boundary_idx = torch.linspace(0,
                                                   len(noise_scheduler.timesteps),
                                                   num_boundaries + 1, dtype=int)
            else:
                self.boundary_idx = torch.tensor(boundaries, dtype=int)
            self.boundary_start_idx = self.boundary_idx[:-1]
            self.boundary_end_idx = self.boundary_idx[1:]

        self.noise_scheduler = noise_scheduler
    ## ---------------------------------------------------------------------------


    ## ---------------------------------------------------------------------------
    def sample_end_boundary_idx(self, batch_of_start_idx):
        """
        Sample indexes of the end boundaries for batch of start indexes.
        For example, if num_boundaries = 3 and batch_of_start_idx = [18,  0,  0,  9].
        Then, batch_of_end_idx = [28,  9,  9, 18]

        :param batch_of_start_idx: (tensor), [b_size]
        :return: batch_of_end_idx (tensor), [b_size]
        """

        mask = (batch_of_start_idx[None, :] == self.boundary_end_idx[:, None]).long()
        idx = torch.argmax(mask[[self.num_boundaries - 1] + list(range(0, self.num_boundaries - 1)), :], dim=0)
        batch_of_end_idx = self.boundary_end_idx[idx]

        return batch_of_end_idx
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