"""Training NCSN++ on CelebAHQ with VE SDEs."""

import ml_collections
import torch


def get_default_configs():
    config = ml_collections.ConfigDict()

    # training
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 8
    training.n_iters = 2400001
    training.snapshot_freq = 50000
    training.log_freq = 50
    training.eval_freq = 100
    # store additional checkpoints for preemption in cloud computing environments
    training.snapshot_freq_for_preemption = 5000
    # produce samples at each snapshot.
    training.snapshot_sampling = True
    training.sde = "vesde"
    training.likelihood_weighting = False
    training.continuous = True
    training.reduce_mean = False

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.method = "pc"
    sampling.predictor = "reverse_diffusion"
    sampling.corrector = "langevin"
    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.15

    # evaluation 
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.begin_ckpt = 1
    evaluate.end_ckpt = 96
    evaluate.batch_size = 1024
    evaluate.num_samples = 50000

    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = "CelebAHQ"
    data.image_size = 1024
    data.random_flip = True
    data.uniform_dequantization = False
    data.centered = False
    data.num_channels = 3
    # Plug in your own path to the tfrecords file.
    # data.tfrecords_path = "/atlas/u/yangsong/celeba_hq/-r10.tfrecords"

    # model
    config.model = model = ml_collections.ConfigDict()
    model.name = "ncsnpp"
    model.scale_by_sigma = True
    model.sigma_max = 1348
    model.sigma_min = 0.01
    model.num_scales = 2000
    model.ema_rate = 0.9999
    model.normalization = "GroupNorm"
    model.nonlinearity = "swish"
    model.nf = 16
    model.ch_mult = (1, 2, 4, 8, 16, 32, 32, 32)
    model.num_res_blocks = 1
    model.attn_resolutions = (16,)
    model.dropout = 0.
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = True
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = "biggan"
    model.progressive = "output_skip"
    model.progressive_input = "input_skip"
    model.progressive_combine = "sum"
    model.attntion_type = "ddpm"
    model.init_scale = 0.
    model.fourier_scale = 16
    model.conv_size = 3
    model.embedding_type = "fourier"

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = "Adam"
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.amsgrad = False
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.

    config.seed = 42
    config.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    return config