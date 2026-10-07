# SPaGS_bgm

**SplatEnv implementation for panoramic cameras based on SPaGS**

`SPaGS_bgm` adds explicit background modeling to the spherical panoramic 3D Gaussian Splatting pipeline of [SPaGS](https://github.com/nerficg-project/SPaGS). SPaGS remains responsible for foreground reconstruction and panoramic rasterization, while a separate background model represents the distant environment.

During rendering, the foreground color $C_{fg}$, foreground opacity $\alpha_{fg}$, and background color $C_{bg}$ are composited as follows:

$$
C = C_{fg} + (1-\alpha_{fg})C_{bg}.
$$

## Design for Panoramic Cameras

A complete equirectangular panorama covers the entire sphere, so the camera model has no out-of-view directions or zenith blind spot of the kind found in perspective cameras. The environment model must remain consistent with this omnidirectional observation.

`SPaGS_bgm` therefore uses the following design:

- **Full-sphere initialization**: HG and VET place environment Gaussians across the entire sphere to cover every viewing direction in a panorama.
- **Uniform scale**: Environment Gaussian scale does not vary with latitude within a scene. Panoramas directly observe different latitudes, so assigning larger Gaussians to the zenith provides none of the efficiency benefit found in the perspective-camera version.
- **Panoramic rasterization**: Both the foreground and Gaussian background use the omnidirectional ray-splat rasterization pipeline from SPaGS, avoiding the problems that perspective projection causes at panoramic seams and poles.
- **Visibility filtering**: Unlike a perspective camera, a complete equirectangular panorama provides full-sphere coverage at the camera-model level and therefore has no inherent zenith blind spot caused by a limited field of view. Real data may still contain local gaps due to occlusion, cropping, or incomplete capture, but these differ from the systematic out-of-view regions of a perspective camera.
- **Semantic sky constraint**: A sky mask still suppresses foreground Gaussian leakage into sky regions and improves consistency along foreground-background boundaries.

Because complete panoramas already cover every direction, the SPaGS version does not need the SH sky-cone constraint designed for the zenith blind spot in HTGS. It also does not rely on KNN color propagation or opacity completion to infer content outside the camera field of view. Here, semantic priors are used mainly for foreground-background separation, initialization, and boundary stabilization.

## Background Models

| Configuration value | Background representation |
| --- | --- |
| `SH` | Global spherical harmonics |
| `MLP` | MLP that predicts color from the viewing direction |
| `TEX` | Learnable spherical environment texture map |
| `MSI` | Multi-Sphere Image |
| `HG` | Single-layer environment Gaussians |
| `VET` | Multi-layer environment Gaussians |

All background models are composited with the SPaGS foreground result through foreground transmittance. HG and VET render full-sphere environment Gaussians using the panoramic rasterization backend from SPaGS.

## SplatEnv Extensions

Compared with the original SPaGS implementation, this directory adds:

- six interchangeable explicit background representations;
- SegFormer-based sky-mask generation;
- dominant sky-color and complexity estimation;
- sky-aware foreground suppression;
- full-sphere HG / VET initialization;
- Gaussian count and uniform-scale adjustment based on scene scale and sky complexity;
- staged background optimization, with background updates enabled at the beginning and end of training and frozen during the middle stage.

## NeRFICG 2.0

This version supports **NeRFICG 2.0** and reads panoramic data through its unified dataset API. The additional dataloader and `move_dataloader.sh` from the original SPaGS release are no longer needed.

## Installation

Install NeRFICG 2.0 first, then run the following commands from the repository root:

```bash
cp -r src/Methods/PA_SplatEnv/SPaGS_bgm src/Methods/SPaGS_bgm
python scripts/install.py -m SPaGS_bgm
```

Install the dependencies for the semantic sky prior:

```bash
python -m pip install pillow transformers==4.53.2 accelerate==1.9.0
```

Sky segmentation uses:

```text
nvidia/segformer-b0-finetuned-ade-512-512
```

The model weights are downloaded the first time sky masks are generated.

## Training

Create a configuration:

```bash
python scripts/create_config.py \
    -m SPaGS_bgm \
    -d <DATASET_TYPE> \
    -o <CONFIG_NAME>
```

In the generated YAML file, set `BACKGROUND_MODEL` to `SH`, `MLP`, `TEX`, `MSI`, `HG`, or `VET`, then configure the panoramic dataset path and other training parameters.

Start training:

```bash
python scripts/train.py -c configs/<CONFIG_NAME>.yaml
```

To train multiple configurations sequentially, use:

```bash
python scripts/sequential_train.py -d configs/<CONFIG_DIRECTORY>
```

### HG / VET Configuration

The Gaussian environment models in the panoramic version provide the following key parameters:

| Parameter | Description |
| --- | --- |
| `USE_AUTO_CONFIG` | Automatically configure capacity and scale from scene scale and sky complexity |
| `BASE_GAUSSIANS` | Base number of environment Gaussians |
| `NUM_LAYERS` | Number of spherical shells for VET; HG always uses one layer |
| `BETA_INTERVAL` | Radial spacing between adjacent spherical shells |
| `SCALE_RATIO` | Uniform relative Gaussian scale across the full sphere |

The panoramic version does not provide `HALF_BALL`, `MIN_SCALE_RATIO`, `MAX_SCALE_RATIO`, or `KNN_COLOR`. Environment Gaussians cover the complete sphere at a uniform scale within each scene, so no color propagation is required for directions outside a perspective field of view.

## Upstream Project

This implementation is based on the official SPaGS project:

**SPaGS: Fast and Accurate 3D Gaussian Splatting for Spherical Panoramas**<br>
Junbo Li, Florian Hahlbohm, Timon Scholz, Martin Eisemann, Jan-Philipp Tauscher, and Marcus Magnor.<br>
[Project repository](https://github.com/nerficg-project/SPaGS)

The panoramic Gaussian representation, rasterization, and CUDA backend in SPaGS belong to the upstream project. The SplatEnv background models, semantic sky prior, and related extensions were implemented by **Jiahao Chen**.
