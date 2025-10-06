# Fast constrained sampling in pre-trained diffusion models [NeurIPS 2025]

[[arXiv]](https://arxiv.org/abs/2410.18804)

Code for **Fast constrained sampling in pre-trained diffusion models**


| Image | Inpainting Steps (~15s) |
| - | - |
| <img src="assets/thecat_masked.jpg" alt="drawing" width="300"/> | <img src="assets/thecat_inpainting.gif" alt="drawing" width="300"/> |

## Usage
We provide two different implementations of the proposed sampling algorithm.

- In `stable-diffusion` we provide an implementation based on the [LDM repository](https://github.com/CompVis/stable-diffusion).
  - `inpaint.ipynb` performs inpainting on a given image and mask.
  - `superres.ipynb` performs super-resolution on an image and a given downsampling rate.
  - `style.ipynb` generates an image from a given caption, following the style provided in the reference image. We utilize the second layer features from a CLIP ViT-B/16 to compare the style between the generated and reference images.
  - `superres_vae_newton.ipynb` also performs super-resolution but skips backpropagating through the Stable Diffusion decoder using a second Newton approximation. 

- In `diffusers` we provide an implementation using the [diffusers](https://huggingface.co/docs/diffusers/index) library.
  - `inpaint.ipynb` performs inpainting on a given image.


## Bibtex

```
@article{graikos2024fast,
  title={Fast constrained sampling in pre-trained diffusion models},
  author={Graikos, Alexandros and Jojic, Nebojsa and Samaras, Dimitris},
  journal={arXiv preprint arXiv:2410.18804},
  year={2024}
}
```
