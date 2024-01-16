// Project title
export const title = "Fixed Point Diffusion Models"

// Short version of the abstract
export const description = "We introduce the Fixed Point Diffusion Model (FPDM), a novel approach to image generation that integrates the concept of fixed point solving into the framework of diffusion-based generative modeling."

// Abstract
export const abstract = "We introduce the Fixed Point Diffusion Model (FPDM), a novel approach to image generation that integrates the concept of fixed point solving into the framework of diffusion-based generative modeling. Our approach embeds an implicit fixed point solving layer into the denoising network of a diffusion model, transforming the diffusion process into a sequence of closely-related fixed point problems. Combined with a new stochastic training method, this approach significantly reduces model size, reduces memory usage, and accelerates training. Moreover, it enables the development of two new techniques to improve sampling efficiency: reallocating computation across timesteps and reusing fixed point solutions between timesteps. We conduct extensive experiments with state-of-the-art models on ImageNet, FFHQ, CelebA-HQ, and LSUN-Church, demonstrating substantial improvements in performance and efficiency. Compared to the state-of-the-art DiT model, FPDM contains 87\% fewer parameters, consumes 60\% less memory during training, and improves image generation quality in situations where sampling computation or time is limited."

// Institutions
export const institutions = {
  1: "Oxford University",
}

// Authors
export const authors = [
  {
    'name': 'Xingjian Bai',
    'institutions': [1],
    'url': "https://xingjianbai.com/"
  },
  {
    'name': 'Luke Melas-Kyriazi',
    'institutions': [1],
    'url': "https://github.com/lukemelas/"
  },
]

// Links
export const links = {
  'paper': "#", // "https://arxiv.org/abs/2002.00733",
  'github': "https://github.com/lukemelas/fixed-point-diffusion-models"
}

// Acknowledgements
export const acknowledgements = "L.M.K. is supported by the Rhodes Trust."

// Citation
export const citationId = "bai2024fixedpoint"
export const citationAuthors = "Xingjian Bai and Luke Melas-Kyriazi"
export const citationYear = "2024"
export const citationBooktitle = "Arxiv"

// Video
export const video_url = "https://www.youtube.com/embed/ScMzIvxBSi4"