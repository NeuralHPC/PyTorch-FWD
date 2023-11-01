from typing import List, Tuple

import torch
import torch.nn as nn

from tqdm import tqdm


@torch.jit.script
def linear_noise_scheduler(
    current_time_step: int, max_steps: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample linear noise scheduler.

    Args:
        current_time_step (int): Current time
        max_steps (int): Maximum number of steps

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing alpha_bar, alpha and betas at current time
    """
    betas = torch.linspace(0.0001, 0.02, max_steps)
    alphas = 1 - betas
    alpha_ts = torch.cumprod(alphas, dim=0)
    return (
        alpha_ts[current_time_step],
        alphas[current_time_step],
        betas[current_time_step],
    )


def sample_noise(
    img: torch.Tensor,
    alpha_t: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Diffusion forward step.

    Args:
        img (torch.Tensor): Input image
        alpha_t (float): Alpha value for the corresponding timestep

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing the noised image and noise
    """
    noise = torch.randn_like(img)
    a = (alpha_t**0.5) * img
    b = ((1 - alpha_t) ** 0.5) * noise
    x = a + b
    return x, noise


# def linear_alpha(current_step: int, max_steps: int) -> float:
#     return current_step / max_steps


# def sample_noise_simple(
#     img: jnp.ndarray,
#     current_time_step: int,
#     key: jnp.ndarray,
#     max_steps: int,
#     alpha_function: callable = linear_alpha,
# ):
#     alpha = alpha_function(current_time_step, max_steps)
#     noise = jax.random.normal(key, shape=img.shape)
#     x = (1 - alpha) * img + noise * alpha
#     return x, img - x


# def sample_net_noise(
#     net_state: FrozenDict,
#     model: nn.Module,
#     key: int,
#     input_shape: List[int],
#     max_steps: int,
#     label: int = 3338,
#     sampling_function: callable = sample_noise,
# ):
#     prng_key = jax.random.PRNGKey(key)
#     process_array = jax.random.normal(prng_key, shape=[1] + input_shape)

#     for time in reversed(range(max_steps)):
#         de_noise = model.apply(
#             net_state,
#             (
#                 process_array,
#                 jnp.expand_dims(jnp.array(time), -1),
#                 jnp.expand_dims(jnp.array([label]), 0),
#             ),
#         )
#         if sampling_function == sample_noise_simple:
#             process_array += de_noise
#         else:
#             process_array -= de_noise

#         prng_key = jax.random.split(prng_key, 1)[0]
#         process_array = sampling_function(process_array, time, prng_key, max_steps)[0]
#     return process_array[0]


def sample_DDPM(
    class_labels: torch.Tensor,
    model: nn.Module,
    max_steps: int,
    input_shape: List[int],
    device: torch.device,
) -> torch.Tensor:
    """DDPM Sampling from https://arxiv.org/pdf/2006.11239.pdf.

    Args:
        class_labels (torch.Tensor): Labels for class conditioning
        model (nn.Module): Model instance
        max_steps (int): Maximum steps
        input_shape (List[int]): Input images shape
        device (torch.device): torch device to use

    Returns:
        torch.Tensor: Returned sampled image
    """
    # device = torch.device('cpu')
    model = model.to(device)
    x_t = torch.randn(input_shape, device=device)
    x_t_1 = x_t
    class_labels = class_labels.to(device)

    for time in tqdm(reversed(range(max_steps)), total=max_steps):
        alpha_t, alpha, _ = linear_noise_scheduler(time, max_steps)
        z = torch.randn_like(x_t_1, device=device)
        time = torch.unsqueeze(torch.tensor(time, device=device), dim=-1)
        denoise = model(x_t_1, time, return_dict=False)[0]
        x_mean = (x_t_1 - (denoise * ((1 - alpha) / ((1 - alpha_t) ** 0.5)))) / (
            alpha**0.5
        )
        x_t_1 = x_mean + ((1 - alpha) ** 0.5) * z
    return x_t_1


def sample_DDIM(
    class_labels: torch.Tensor,
    model: nn.Module,
    max_steps: int,
    input_shape: List[int],
    device: torch.device,
    eta: float = 0.0,
    tau_steps: int = 1,
) -> torch.Tensor:
    """DDIM Sampling from https://arxiv.org/pdf/2010.02502.pdf.

    Args:
        class_labels (torch.Tensor): Labels for class conditioning
        model (nn.Module): Model instance
        max_steps (int): Maximum steps
        input_shape (List[int]): Input images shape
        device (torch.device): torch device to use
        eta (float, optional): Eta for sigmal calculation, Defaults to 0.0.
        tau_steps (int, optional): Tau steps for DDIM, Defaults to 1.

    Returns:
        torch.Tensor: Sampled image
    """
    model = model.to(device)
    x_t = torch.rand(input_shape, device=device)
    x_t_1 = x_t
    class_labels = class_labels.to(device)

    for time in reversed(range(max_steps)):
        alpha_t, _, _ = linear_noise_scheduler(time, max_steps)
        alpha_t_1 = 1.0
        if time != 0:
            alpha_t_1, _, _ = linear_noise_scheduler(time - 1, max_steps)

        sigma_t = eta * (
            (torch.sqrt((1 - alpha_t_1) / (1 - alpha_t)))
            * (torch.sqrt((1 - alpha_t) / alpha_t_1))
        )

        z = torch.rand_like(x_t_1, device=device)
        time = torch.unsqueeze(torch.tensor(time, device=device), dim=-1)
        denoise = model(x_t_1, time, class_labels)
        # First term
        pred_x_0 = torch.sqrt(alpha_t_1 / alpha_t) * (
            x_t_1 - (torch.sqrt(1 - alpha_t)) * denoise
        )
        # Second term
        point_x_t = torch.sqrt(1 - alpha_t_1 - (sigma_t**2)) * denoise
        # Final term
        x_mean = pred_x_0 + point_x_t
        x_t_1 = x_mean + sigma_t * z
    return x_t_1
