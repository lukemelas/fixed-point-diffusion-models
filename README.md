## Fixed Point Diffusion Models<br><sub>Anonymous Supplementary Code</sub>
![DiT samples](visuals/splash-figure-v1.png)

## Setup

We provide an [`environment.yml`](environment.yml) file that can be used to create a Conda environment. If you only want 
to run pre-trained models locally on CPU, you can remove the `cudatoolkit` and `pytorch-cuda` requirements from the file.

```bash
conda env create -f environment.yml
conda activate DiT
```

## Model

Our model definition, including all fixed point functionality, is included in `models.py`.

## Training

Example training scripts:
```bash
# Standard model
accelerate launch --config_file aconfs/1_node_1_gpu_ddp.yaml --num_processes 8 train.py

# Fixed Point Diffusion Model
accelerate launch --config_file aconfs/1_node_1_gpu_ddp.yaml --num_processes 8 train.py --fixed_point True --deq_pre_depth 1 --deq_post_depth 1

# With v-prediction and zero-SNR
accelerate launch --config_file aconfs/1_node_1_gpu_ddp.yaml --num_processes 8 train.py --output_subdir v_pred_exp --predict_v True --use_zero_terminal_snr True --fixed_point True --deq_pre_depth 1 --deq_post_depth 1

# With v-prediction and zero-SNR, with 4 pre- and post-layers
accelerate launch --config_file aconfs/1_node_1_gpu_ddp.yaml --num_processes 8 train.py --output_subdir v_pred_exp --predict_v True --use_zero_terminal_snr True --fixed_point True --deq_pre_depth 4 --deq_post_depth 4
```

## Sampling

Example sampling scripts:
```bash
# Sample
python sample.py --ckpt {checkpoint-path-from-above} --fixed_point True --fixed_point_pre_depth 1 --fixed_point_post_depth 1 --cfg_scale 4.0 --num_sampling_steps 20

# Sample with fewer iterations per timestep and more timesteps
python sample.py --ckpt {checkpoint-path-from-above} --fixed_point True --fixed_point_pre_depth 1 --fixed_point_post_depth 1 --cfg_scale 4.0 --fixed_point_iters 12 --num_sampling_steps 40 --fixed_point_reuse_solution True
```
