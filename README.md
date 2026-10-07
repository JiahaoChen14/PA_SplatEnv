# SplatEnv

**Background and Environment Modeling for 3D Gaussian Splatting**

SplatEnv investigates how to introduce explicit background modeling into 3D Gaussian Splatting (3DGS) for outdoor and unbounded scenes, thereby improving the reconstruction quality of the sky, distant views, and regions without valid observations.

This project implements a unified foreground/background modeling framework on top of two existing Gaussian Splatting methods:

| Method | Camera Model | Base Method |
| --- | --- | --- |
| `HTGS_bgm` | Perspective camera | [HTGS](https://github.com/nerficg-project/HTGS) |
| `SPaGS_bgm` | Panoramic camera | [SPaGS](https://github.com/nerficg-project/SPaGS) |

The foreground is represented by the original Gaussian scene model from HTGS or SPaGS, while the background is represented by a separate environment model. During rendering, the two are composited according to the foreground transmittance.

## Background Models

SplatEnv provides a unified implementation of six background representations in HTGS and SPaGS:

| Model | Description |
| --- | --- |
| **SH** | Represents environment colors using global spherical harmonics |
| **MLP** | A neural network that predicts background colors from viewing directions |
| **TEX** | A learnable spherical environment texture map (environment map) |
| **MSI** | Multi-Sphere Image, a multilayer spherical texture representation |
| **HG** | A single-layer environment Gaussian representation |
| **VET** | A multilayer environment Gaussian representation |

These background models are integrated into both perspective and panoramic camera reconstruction pipelines, enabling comparisons within a unified training and evaluation framework.

## Key Extensions

In addition to the unified implementation and integration of different background representations, SplatEnv includes:

- **Semantic Sky Prior**: Uses semantic segmentation to obtain sky masks and extract the dominant sky color and complexity information.
- **Foreground Suppression**: Suppresses the opacity of foreground Gaussians in sky regions to reduce foreground leakage into the sky.
- **Background Completion**: Propagates colors and opacity across Gaussian background regions that lack valid supervision, reducing black holes and discontinuities in the sky.
- **Scene-Adaptive Configuration**: Adjusts the number, scale, and related parameters of Gaussian environment models according to the scene scale and sky complexity.
- **Staged Background Optimization**: Optimizes and freezes background parameters in stages during training.

The six background representations themselves originate from existing research. SplatEnv focuses on integrating them into HTGS and SPaGS under a unified framework and studying how different background representations perform in perspective and panoramic 3DGS scenes.

## NeRFICG 2.0

The current version has been adapted for **NeRFICG 2.0**, using the unified dataset API and supporting the new training and rendering interfaces. The additional dataloader and `move_dataloader.sh` required by the original Project Thesis implementation are no longer needed.

## Installation

First, follow the [NeRFICG](https://github.com/nerficg-project/nerficg) installation instructions to prepare the base environment, and then install the desired SplatEnv method. Run all commands below from the repository root.

Clone the repository and its submodules:

```bash
git clone --recursive https://github.com/JiahaoChen14/PA_SplatEnv.git
cd PA_SplatEnv
```

The two SplatEnv methods are located under `src/Methods/PA_SplatEnv/`.

For the perspective-camera version, copy and install `HTGS_bgm`:

```bash
cp -r src/Methods/PA_SplatEnv/HTGS_bgm src/Methods/HTGS_bgm
python scripts/install.py -m HTGS_bgm
```

For the panoramic-camera version, copy and install `SPaGS_bgm`:

```bash
cp -r src/Methods/PA_SplatEnv/SPaGS_bgm src/Methods/SPaGS_bgm
python scripts/install.py -m SPaGS_bgm
```

### Semantic Sky Prior Dependencies

SplatEnv uses SegFormer to generate semantic sky masks. The relevant dependencies are:

| Package | Version | Purpose |
| --- | --- | --- |
| `Pillow` | Unpinned | Image loading and processing |
| `transformers` | `4.53.2` | Loading the SegFormer image processor and segmentation model |
| `accelerate` | `1.9.0` | Supporting dependency required to run the model |

Install them in the current Python environment:

```bash
python -m pip install pillow transformers==4.53.2 accelerate==1.9.0
```

`transformers` and `accelerate` are declared in the NeRFICG extension configuration for both methods and are normally installed automatically with the methods. Please also ensure that `Pillow` is installed in the environment.

The semantic sky prior uses the following pretrained model:

```text
nvidia/segformer-b0-finetuned-ade-512-512
```

When generating sky masks for the first time, the model weights will be downloaded automatically if they are not already available in the local Hugging Face cache. Access to Hugging Face is therefore required.

For method-specific instructions, see:

- [`HTGS_bgm`](HTGS_bgm/README.md)
- [`SPaGS_bgm`](SPaGS_bgm/README.md)

## Training

Create a training configuration, replacing `<DATASET_TYPE>` and `<CONFIG_NAME>` with the actual dataset type and configuration name:

```bash
python scripts/create_config.py \
    -m HTGS_bgm \
    -d <DATASET_TYPE> \
    -o <CONFIG_NAME>
```

For panoramic-camera reconstruction, replace `HTGS_bgm` with `SPaGS_bgm`. In the generated YAML configuration, set the dataset path and select the desired background model and its parameters.

Start training:

```bash
python scripts/train.py -c configs/<CONFIG_NAME>.yaml
```

## Repository Structure

| Path | Contents |
| --- | --- |
| `src/Methods/PA_SplatEnv/HTGS_bgm/` | HTGS-based background modeling implementation for perspective cameras |
| `src/Methods/PA_SplatEnv/SPaGS_bgm/` | SPaGS-based background modeling implementation for panoramic cameras |
| `configs/` | Training configurations |
| `scripts/` | NeRFICG training and utility scripts |
| `resources/` | Project resources |

## Project Thesis

This project originated as a Project Thesis at the TU Braunschweig Computer Graphics Lab:

**An Investigation of Background Modeling Techniques for 3D Gaussian Splatting**

## License and Acknowledgments

SplatEnv is built on the following projects:

- [NeRFICG](https://github.com/nerficg-project/nerficg)
- [HTGS](https://github.com/nerficg-project/HTGS)
- [SPaGS](https://github.com/nerficg-project/SPaGS)

The original Gaussian representations, training pipelines, and rasterization backends of HTGS and SPaGS were developed by their respective authors.

When using or redistributing the relevant code, please comply with the original licenses and copyright notices of NeRFICG, HTGS, and SPaGS.
