import json
import math
from pathlib import Path
from typing import Optional

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.models import AutoencoderKL
from PIL import Image
from tap import Tap
from tqdm import trange


from diffusion import create_diffusion
from download import find_model
from models import DiT_models
from adaptive_controller import LinearController


class Args(Tap):

    # Paths
    output_dir: str = 'samples'

    # Dataset
    dataset_name: str = "imagenet256"

    # Model
    model: str = "DiT-XL/2"
    vae: str = "mse"
    num_classes: int = 1000
    image_size: int = 256
    predict_v: bool = False
    use_zero_terminal_snr: bool = False
    unsupervised: bool = False
    dino_supervised: bool = False
    dino_supervised_dim: int = 768
    flow: bool = False
    debug: bool = False

    # Fixed Point settings
    fixed_point: bool = False
    fixed_point_pre_depth: int = 2
    fixed_point_post_depth: int = 2
    fixed_point_iters: Optional[int] = None
    fixed_point_pre_post_timestep_conditioning: bool = False
    fixed_point_reuse_solution: bool = False

    # Sampling
    ddim: bool = False
    cfg_scale: float = 4.0
    num_sampling_steps: int = 250
    batch_size: int = 32
    ckpt: str = '...'
    global_seed: int = 0
    
    # Parallelization
    sample_index_start: int = 0
    sample_index_end: Optional[int] = 50_000

    # Adaptive
    adaptive: bool = False
    adaptive_type: str = "increasing" # currently only support increasing, fixed, and decreasing

    def process_args(self) -> None:
        """Additional argument processing"""
        if self.debug:
            self.log_with = 'tensorboard'
            self.name = 'debug'

        # Defaults
        self.fixed_point_iters = self.fixed_point_iters or (28 - self.fixed_point_pre_depth - self.fixed_point_post_depth)
        
        # Checks
        if self.cfg_scale < 1.0:
            raise ValueError("In almost all cases, cfg_scale should be >= 1.0")
        if self.unsupervised:
            assert self.cfg_scale == 1.0
            self.num_classes = 1
        elif self.dino_supervised:
            raise NotImplementedError()
        if not Path(self.ckpt).is_file():
            raise ValueError(self.ckpt)

        # Create output directory
        output_parent = Path(self.output_dir) / Path(self.ckpt).parent.parent.name
        if self.debug:
            output_dirname = 'debug'
        else:
            output_dirname = f'num_sampling_steps-{self.num_sampling_steps}--cfg_scale-{self.cfg_scale}'
            if self.fixed_point:
                output_dirname += f'--fixed_point_iters-{self.fixed_point_iters}--fixed_point_reuse_solution-{self.fixed_point_reuse_solution}--fixed_point_pptc-{self.fixed_point_pre_post_timestep_conditioning}'
        if self.ddim:
            output_dirname += f'--ddim'
        self.output_dir = str(output_parent / output_dirname)
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)

        if self.adaptive:
            self.budget = self.num_sampling_steps * self.deq_iters
            if self.adaptive_type == "increasing" or self.adaptive_type == "decreasing" or self.adaptive_type == "fixed":
                self.iteration_controller = LinearController(self.budget, self.num_sampling_steps, type = self.adaptive_type)
            else:
                raise NotImplementedError()

def main(args: Args):

    # Setup accelerator, logging, randomness
    accelerator = Accelerator()
    set_seed(args.global_seed + args.sample_index_start)

    # Load model
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = H_lat = W_lat = args.image_size // 8
    # print(f"!!! Pre model init, fixed_point_reuse_solution: {args.fixed_point_reuse_solution}")
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=(args.dino_supervised_dim if args.dino_supervised else args.num_classes),
        is_label_continuous=args.dino_supervised,
        class_dropout_prob=0,
        learn_sigma=(not args.flow),  # TODO: Implement learned variance for flow-based models
        use_gradient_checkpointing=False,
        fixed_point=args.fixed_point,
        fixed_point_pre_depth=args.fixed_point_pre_depth,
        fixed_point_post_depth=args.fixed_point_post_depth,
        fixed_point_no_grad_min_iters=0, 
        fixed_point_no_grad_max_iters=0,
        fixed_point_with_grad_min_iters=args.fixed_point_iters, 
        fixed_point_with_grad_max_iters=args.fixed_point_iters,
        fixed_point_reuse_solution=args.fixed_point_reuse_solution,
        fixed_point_pre_post_timestep_conditioning=args.fixed_point_pre_post_timestep_conditioning,
        adaptive=args.adaptive,
        iteration_controller=args.iteration_controller,
    ).to(accelerator.device)
    print(f'Loaded model with params: {sum(p.numel() for p in model.parameters()):_}')

    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py
    state_dict = find_model(args.ckpt)
    model.load_state_dict(state_dict)
    model.eval() 
    diffusion = create_diffusion(
        str(args.num_sampling_steps), 
        use_flow=args.flow,
        predict_v=args.predict_v,
        use_zero_terminal_snr=args.use_zero_terminal_snr,
    )
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(accelerator.device).eval()
    using_cfg = args.cfg_scale > 1.0

    # Generate pseudorandom class labels and noises. Note that these are generated using the global 
    # seed, which is shared between all processes. As a result, all processes will generate the same 
    # list of class labels and noises. Then, we take a subset of these based on the `process_index` 
    # of the current process.
    N = 50_000  # this assumes we will never sample more than 50K samples, which I think is reasonable
    generator = torch.Generator(device=accelerator.device)
    generator.manual_seed(args.global_seed)
    class_labels = torch.randint(0, args.num_classes, size=(N,), device=accelerator.device)
    generator.manual_seed(args.global_seed)
    latents = torch.randn(N, model.in_channels, H_lat, W_lat, device=accelerator.device, generator=generator)
    class_labels = class_labels[args.sample_index_start:args.sample_index_end]
    latents = latents[args.sample_index_start:args.sample_index_end]
    indices = list(range(args.sample_index_start, args.sample_index_end))
    print(f'Using pseudorandom class labels and latents (start={args.sample_index_start} and end={args.sample_index_end})')

    # Create output path
    output_dir = Path(args.output_dir)
    args.save(output_dir / 'args.json')
    print(f'Saving samples to {output_dir.resolve()}')

    # Load class labels for helpful filenames
    if args.dataset_name == 'imagenet256':
        with open("utils/imagenet-labels.json", "r") as f:
            label_names: list[str] = json.load(f)
            label_names = [l.lower().replace(' ', '-').replace('\'', '') for l in label_names]
    elif args.unsupervised:
        assert args.cfg_scale == 1.0
        label_names = ["unlabeled"]
    else:
        raise NotImplementedError()

    # Disable gradient
    with torch.inference_mode():

        # Sample loop
        num_batches = math.ceil(len(class_labels) / args.batch_size)
        for batch_idx in trange(num_batches, disable=(not accelerator.is_main_process)):

            # Get pre-sampled inputs
            z = latents[batch_idx*args.batch_size:(batch_idx + 1)*args.batch_size]
            y = class_labels[batch_idx*args.batch_size:(batch_idx + 1)*args.batch_size]
            idxs = indices[batch_idx*args.batch_size:(batch_idx + 1)*args.batch_size]
            output_paths = [output_dir / f'{idx:05d}--{y_i:03d}--{label_names[y_i]}.png' for y_i, idx in zip(y.tolist(), idxs)]

            # Skip files that already exist
            if all(output_path.is_file() for output_path in output_paths):
                print(f'Files already exist (batch {batch_idx}). Skipping.')
                continue

            # Setup classifier-free guidance
            if using_cfg:
                y_null = torch.tensor([1000] * args.batch_size, device=accelerator.device)
                y = torch.cat([y, y_null], 0)
                z = torch.cat([z, z], 0)
                model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
                sample_fn = model.forward_with_cfg
            else:
                model_kwargs = dict(y=y)
                sample_fn = model.forward

            if args.adaptive:
                model.threshold_controller.init_image()

            # Sample latent images
            sample_kwargs = dict(model=sample_fn, shape=z.shape, noise=z, clip_denoised=False, model_kwargs=model_kwargs, 
                progress=False, device=accelerator.device)
            if args.ddim:
                samples = diffusion.ddim_sample_loop(**sample_kwargs)
            else:
                samples = diffusion.p_sample_loop(**sample_kwargs)
            


            if using_cfg:
                samples, _ = samples.chunk(2, dim=0)

            
            # Reset model (resets the initial solution to None)
            model.reset()

            # Decode latents
            samples = vae.decode(samples / vae.config.scaling_factor).sample
            samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            # Save samples to disk as individual .png files
            for sample, output_path in zip(samples, output_paths):
                Image.fromarray(sample).save(output_path)


if __name__ == "__main__":
    args = Args(explicit_bool=True).parse_args()
    main(args)
