import logging
import os
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from pathlib import Path
from time import time
from typing import Callable, Optional

import numpy as np
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration
from PIL import Image
from tap import Tap
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from diffusion import create_diffusion
from models import DiT_models

try:
    from streaming import StreamingDataset
except ImportError:
    StreamingDataset = Dataset


class Args(Tap):

    # Paths
    feature_path: Optional[str] = None
    dataset_name: str = "imagenet256"
    name: Optional[str] = None
    output_dir: str = "results"
    output_subdir: str = "runs"

    # Model
    model: str = "DiT-XL/2"
    num_classes: int = 1000
    image_size: int = 256
    predict_v: bool = False
    use_zero_terminal_snr: bool = False
    unsupervised: bool = False
    dino_supervised: bool = False
    dino_supervised_dim: int = 768
    flow: bool = False
    fixed_point: bool = False
    fixed_point_pre_depth: int = 1
    fixed_point_post_depth: int = 1
    fixed_point_no_grad_min_iters: int = 0
    fixed_point_no_grad_max_iters: int = 10
    fixed_point_with_grad_min_iters: int = 1
    fixed_point_with_grad_max_iters: int = 12
    fixed_point_pre_post_timestep_conditioning: bool = False
    
    # Training
    epochs: int = 1400
    global_batch_size: int = 512
    global_seed: int = 0
    num_workers: int = 4
    log_every: int = 100
    ckpt_every: int = 100_000
    lr: float = 1e-4
    log_with: str = "wandb"
    resume: Optional[str] = None
    # use_streaming_dataset: bool = False
    compile: bool = False
    debug: bool = False

    def process_args(self) -> None:
        """Additional argument processing"""
        if self.debug:
            self.log_with = 'tensorboard'
            self.output_subdir = 'debug'
        
        # Auto-generated name
        if self.name is None:
            experiment_index = len(glob(os.path.join(self.output_dir, self.output_subdir, "*")))
            model_string_name = self.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
            model_string_name += '--flow' if self.flow else '--diff'
            model_string_name += '--unsupervised' if self.unsupervised else '--dino_supervised' if self.dino_supervised else ''
            model_string_name += f'--{self.lr:f}'
            model_string_name += f'--v' if self.predict_v else ''
            model_string_name += f'--zero_snr' if self.use_zero_terminal_snr else ''
            ngmin, ngmax = self.fixed_point_no_grad_min_iters, self.fixed_point_no_grad_max_iters
            gmin, gmax = self.fixed_point_with_grad_min_iters, self.fixed_point_with_grad_max_iters
            model_string_name += (f'--fixed_point-pre_depth-{self.fixed_point_pre_depth}-post_depth-{self.fixed_point_post_depth}' + 
                f'-no_grad_iters-{ngmin:02d}-{ngmax:02d}-with_grad_iters-{gmin:02d}-{gmax:02d}' + 
                f'-pre_post_time_cond_{self.fixed_point_pre_post_timestep_conditioning}' 
                if self.fixed_point else '--dit')
            self.name = f'{experiment_index:03d}-{model_string_name}'
        
        # Copy data to scratch
        if self.feature_path is None:
            assert os.getenv('SLURM_JOBID', None) is not None
            os.environ['TMPDIR'] = TMPDIR = os.path.join('/opt/dlami/nvme/slurm_tmpdir', os.getenv('SLURM_JOBID'))
            self.feature_path = os.path.join('/opt/dlami/nvme/slurm_tmpdir', os.getenv('SLURM_JOBID'))
            features_dir = f"{self.feature_path}/{self.dataset_name}_features"
            labels_dir = f"{self.feature_path}/{self.dataset_name}_{'dino_vitb8' if self.dino_supervised else 'labels'}"
            assert Path(features_dir).is_dir() == Path(labels_dir).is_dir()
            if Path(features_dir).is_dir():
                print(f'Features already exist in {TMPDIR}')
            else:
                start = time()
                print(f'Copying features to {TMPDIR}')
                copy_cmd_1 = f'cp ./features/{self.dataset_name}_npy.tar {TMPDIR}'
                copy_cmd_2 = f'tar xf {os.path.join(TMPDIR, self.dataset_name)}_npy.tar -C {TMPDIR}'
                print(copy_cmd_1)
                os.system(copy_cmd_1)
                print(copy_cmd_2)
                os.system(copy_cmd_2)
                print(f'Finished copying features to {TMPDIR} in {time() - start:.2f}s')

        # Create output directory
        self.output_dir = os.path.join(self.output_dir, self.output_subdir, self.name)
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class CustomDataset(Dataset):
    def __init__(self, features_dir, labels_dir):
        self.features_dir = features_dir
        self.labels_dir = labels_dir

        self.features_files = os.listdir(features_dir)
        self.labels_files = os.listdir(labels_dir)

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]

        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))
        return torch.from_numpy(features), torch.from_numpy(labels)


class CustomStreamingDataset(StreamingDataset):
    def __init__(
        self, 
        local: str, 
        remote: Optional[str] = None, 
        shuffle: bool = False, 
        batch_size: int = 1, 
        transform: Optional[Callable] = None,
    ):
        remote = local if remote is None else remote
        super().__init__(remote=remote, local=local, shuffle=shuffle, batch_size=batch_size)
        self.transform = transform

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        feats = item['features'].squeeze(0)
        label = item['class']
        if self.transform is not None:
            feats = self.transform(feats)
        return feats, label
    

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args: Args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup an experiment folder:
    checkpoint_dir = f"{args.output_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.save(f"{args.output_dir}/args.json")

    # Setup accelerator:
    find_unused_parameters = False  # args.fixed_point and args.fixed_point_b_solver != 'backprop'
    print(f'Using {find_unused_parameters = }')
    accelerator_ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)
    accelerator = Accelerator(
        log_with=args.log_with,
        project_config=ProjectConfiguration(project_dir=args.output_dir),
        kwargs_handlers=[accelerator_ddp_kwargs],
        dynamo_backend=("inductor" if args.compile else None),
    )
    device = accelerator.device

    # Create trackers
    if accelerator.is_main_process:
        accelerator.init_trackers("dit", config=args.as_dict(), init_kwargs={"wandb": {"name": args.name}})
        if args.log_with == 'wandb':
            accelerator.get_tracker("wandb", unwrap=True).log_code()
    print(args)
    
    # Create model
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=(1 if args.unsupervised else (args.dino_supervised_dim if args.dino_supervised else args.num_classes)),
        is_label_continuous=args.dino_supervised,
        class_dropout_prob=(0.0 if args.unsupervised else 0.1),
        learn_sigma=(not args.flow),  # TODO: Implement learned variance for flow-based models
        use_gradient_checkpointing=(not args.compile and not find_unused_parameters),
        fixed_point=args.fixed_point,
        fixed_point_pre_depth=args.fixed_point_pre_depth,
        fixed_point_post_depth=args.fixed_point_post_depth,
        fixed_point_no_grad_min_iters=args.fixed_point_no_grad_min_iters, 
        fixed_point_no_grad_max_iters=args.fixed_point_no_grad_max_iters,
        fixed_point_with_grad_min_iters=args.fixed_point_with_grad_min_iters, 
        fixed_point_with_grad_max_iters=args.fixed_point_with_grad_max_iters,
        fixed_point_pre_post_timestep_conditioning=args.fixed_point_pre_post_timestep_conditioning,
    ).to(device)
    print(f'Loaded model with params: {sum(p.numel() for p in model.parameters()):_}')

    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    diffusion = create_diffusion(
        timestep_respacing="", 
        use_flow=args.flow,
        predict_v=args.predict_v,
        use_zero_terminal_snr=args.use_zero_terminal_snr,
    )  # default: 1000 steps, linear noise schedule
    # # Note: the VAE is not used because we assume all images are already preprocessed
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    print(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    # Setup data:
    batch_size = int(args.global_batch_size // accelerator.num_processes)
    # if args.use_streaming_dataset:
    #     data_dir = f"{args.feature_path}/{args.dataset_name}_streaming"
    #     dataset = CustomStreamingDataset(data_dir, shuffle=True, batch_size=batch_size)
    #     load_kwargs = dict()
    # else:
    features_dir = f"{args.feature_path}/{args.dataset_name}_features"
    labels_dir = f"{args.feature_path}/{args.dataset_name}_{'dino_vitb8' if args.dino_supervised else 'labels'}"
    dataset = CustomDataset(features_dir, labels_dir)
    load_kwargs = dict(shuffle=True, pin_memory=True, drop_last=True)
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=args.num_workers, **load_kwargs
    )
    print(f"Dataset contains {len(dataset):,} images ({args.feature_path})")

    # Load from checkpoint
    train_steps = 0
    if args.resume is not None:
        checkpoint: dict = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        ema.load_state_dict(checkpoint['ema'])
        opt.load_state_dict(checkpoint['opt'])
        train_steps = checkpoint['train_steps'] if 'train_steps' in checkpoint else int(Path(args.resume).stem)
        print(f'Resuming from checkpoint: {args.resume}')

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    model, opt, loader = accelerator.prepare(model, opt, loader)
    
    # Train
    log_steps = 0
    running_loss = 0
    steps_per_sec = 0
    start_time = time()
    progress_bar = tqdm()
    print(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            print(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.squeeze(dim=1)
            y = y.squeeze(dim=-1)
            if args.unsupervised:  # replace class labels with zeros
                y = torch.zeros_like(y)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            loss_float = loss.item()
            opt.zero_grad()
            accelerator.backward(loss)
            if train_steps < 5 and accelerator.is_main_process:  # debug
                print(f'[Step {train_steps}] Params total:     {sum(p.numel() for p in model.parameters()):_}')
                print(f'[Step {train_steps}] Params req. grad: {sum(p.numel() for p in model.parameters() if p.requires_grad):_}')
                print(f'[Step {train_steps}] Params with grad: {sum(p.numel() for p in model.parameters() if p.requires_grad and p.grad is not None):_}')
            opt.step()
            update_ema(ema, model)

            # Log every step
            if train_steps % 5 == 0 and accelerator.is_main_process:
                accelerator.log({
                    "train/step": train_steps, "train/loss": loss_float, 
                }, step=train_steps)
                progress_bar.set_description_str((f"train/step: {train_steps}, train/steps_per_sec: {steps_per_sec:.2f}, train/loss: {loss_float:.4f}"))

            # Print periodically
            running_loss += loss_float
            log_steps += 1
            train_steps += 1
            progress_bar.update()
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item() / accelerator.num_processes
                print(f"\n(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                accelerator.log({"train/steps_per_sec": steps_per_sec}, step=train_steps)  # also log steps per second
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()
            
            # Save DiT checkpoint:
            if train_steps > 0 and accelerator.is_main_process and (train_steps % 5000 == 0 or train_steps % args.ckpt_every == 0):
                checkpoint = {
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args.as_dict(),
                    "train_steps": train_steps,
                }
                if train_steps % 5000 == 0:
                    checkpoint_path = f"{checkpoint_dir}/latest.pt"
                    torch.save(checkpoint, checkpoint_path)
                if train_steps % args.ckpt_every == 0:
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    accelerator.end_training()
    print("Done!")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    args = Args(explicit_bool=True).parse_args()
    main(args)