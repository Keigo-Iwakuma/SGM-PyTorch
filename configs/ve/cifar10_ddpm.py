"""Train the original DDPM model with SMLD."""

from configs.default_cifar10_configs import get_default_configs


def get_config():
    config = get_default_configs()

    # training
    training = config.training
    training.sde = "vesde"
    training.continuous = False

    # sampling
    sampling = config.sampling
    sampling.method = "pc"
    sampling.predictor = "reverse_diffusion"
    sampling.corrector = "langevin"

    # model
    model = config.model
    model.name = "ddpm"
    model.scale_by_sigma = True
    model.ema_rate = 0.999
    model.normalization = "GroupNorm"
    model.nonlinearity = "swish"
    model.nf = 128
    model.ch_mult = (1, 2, 2, 2)
    model.num_res_blocks = 2
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.conv_size = 3

    return config