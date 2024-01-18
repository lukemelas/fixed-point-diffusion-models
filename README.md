<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>

<div align="center">

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]

### Fixed Point Diffusion Models

[Project Page](https://lukemelas.github.io/fixed-point-diffusion-models/) · [Paper](https://arxiv.org/abs/2401.08741)

<hr>

</div>

![DiT samples](visuals/splash-figure-v1.png)

### Table of Contents
- [Abstract](#abstract)
- [Setup & Installation](#setup)
- [Model](#model)
- [Training](#training)
- [Sampling](#sampling)
- [Contribution](#contribution)
- [Acknowledgements](#acknowledgements)

### Roadmap

- [x] Code and paper release 🎉🎉
- [x] Jupyter notebook example
- [ ] Pretrained model release _(coming soon)_
- [ ] Code walkthrough and tutorial

### Abstract

We introduce the Fixed Point Diffusion Model (FPDM), a novel approach to image generation that integrates the concept of fixed point solving into the framework of diffusion-based generative modeling. Our approach embeds an implicit fixed point solving layer into the denoising network of a diffusion model, transforming the diffusion process into a sequence of closely-related fixed point problems. Combined with a new stochastic training method, this approach significantly reduces model size, reduces memory usage, and accelerates training. Moreover, it enables the development of two new techniques to improve sampling efficiency: reallocating computation across timesteps and reusing fixed point solutions between timesteps. We conduct extensive experiments with state-of-the-art models on ImageNet, FFHQ, CelebA-HQ, and LSUN-Church, demonstrating substantial improvements in performance and efficiency. Compared to the state-of-the-art DiT model, FPDM contains 87% fewer parameters, consumes 60% less memory during training, and improves image generation quality in situations where sampling computation or time is limited.

### Setup

We provide an [`environment.yml`](environment.yml) file that can be used to create a Conda environment. If you only want 
to run pre-trained models locally on CPU, you can remove the `cudatoolkit` and `pytorch-cuda` requirements from the file.

```bash
conda env create -f environment.yml
conda activate DiT
```

### Model

Our model definition, including all fixed point functionality, is included in `models.py`.

### Training

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

### Sampling

Example sampling scripts:
```bash
# Sample
python sample.py --ckpt {checkpoint-path-from-above} --fixed_point True --fixed_point_pre_depth 1 --fixed_point_post_depth 1 --cfg_scale 4.0 --num_sampling_steps 20

# Sample with fewer iterations per timestep and more timesteps
python sample.py --ckpt {checkpoint-path-from-above} --fixed_point True --fixed_point_pre_depth 1 --fixed_point_post_depth 1 --cfg_scale 4.0 --fixed_point_iters 12 --num_sampling_steps 40 --fixed_point_reuse_solution True
```

### Contribution

Pull requests are welcome!

### Acknowledgements

* The strong baseline from DiT:
    ```
    @article{Peebles2022DiT,
    title={Scalable Diffusion Models with Transformers},
    author={William Peebles and Saining Xie},
    year={2022},
    journal={arXiv preprint arXiv:2212.09748},
    }
    ```

* The fast-DiT code from [chuanyangjin](https://github.com/chuanyangjin/fast-DiT):

* All the great work from the [CMU Locus Lab](https://github.com/locuslab) on Deep Equilibrium Models, which started with:
    ```
    @inproceedings{bai2019deep,
    author    = {Shaojie Bai and J. Zico Kolter and Vladlen Koltun},
    title     = {Deep Equilibrium Models},
    booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
    year      = {2019},
    }
    ``` 

* L.M.K. thanks the Rhodes Trust for their scholarship support.


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/lukemelas/fixed-point-diffusion-models.svg?style=for-the-badge
[contributors-url]: https://github.com/lukemelas/fixed-point-diffusion-models/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/lukemelas/fixed-point-diffusion-models.svg?style=for-the-badge
[forks-url]: https://github.com/lukemelas/fixed-point-diffusion-models/network/members
[stars-shield]: https://img.shields.io/github/stars/lukemelas/fixed-point-diffusion-models.svg?style=for-the-badge
[stars-url]: https://github.com/lukemelas/fixed-point-diffusion-models/stargazers
[issues-shield]: https://img.shields.io/github/issues/lukemelas/fixed-point-diffusion-models.svg?style=for-the-badge
[issues-url]: https://github.com/lukemelas/fixed-point-diffusion-models/issues
[license-shield]: https://img.shields.io/github/license/lukemelas/fixed-point-diffusion-models.svg?style=for-the-badge
[license-url]: https://github.com/lukemelas/fixed-point-diffusion-models/blob/master/LICENSE.txt
