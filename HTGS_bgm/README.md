# HTGS_bgm

**SplatEnv implementation for perspective cameras based on HTGS**

`HTGS_bgm` adds explicit background modeling to the perspective-camera 3D Gaussian Splatting pipeline of [HTGS](https://github.com/nerficg-project/HTGS). HTGS remains responsible for foreground reconstruction and perspective-correct rasterization, while a separate environment model represents the sky and distant scenery.

During rendering, the foreground color $C_{fg}$, foreground opacity $\alpha_{fg}$, and background color $C_{bg}$ are composited as follows:

$$
C = C_{fg} + (1-\alpha_{fg})C_{bg}.
$$

## Design for Perspective Cameras

A perspective image covers only a limited field of view, so the training images may not observe every background direction, especially near the zenith and in sky regions occluded by foreground geometry. This can produce unstable colors, gray regions, black holes, or flickering as the viewpoint changes.

`HTGS_bgm` addresses these issues with the following design:

- **Hemisphere initialization**: By default, HG and VET initialize environment Gaussians only on the primarily visible upper hemisphere, avoiding large numbers of ineffective points in the lower hemisphere that perspective cameras cannot observe.
- **Latitude-dependent scale**: Areas near the horizon usually contain richer high-frequency detail and therefore use smaller Gaussians. Closer to the zenith, supervision is sparser and textures are generally smoother, so larger Gaussians are used.
- **SH sky-cone constraint**: For SH backgrounds, directions near the zenith are sampled and the dominant color estimated from the semantic sky prior provides a lower luminance bound, preventing unobserved directions from turning black.
- **Color completion**: For HG and VET, environment Gaussians without valid image supervision are detected first, then KNN propagation transfers colors from reliable neighboring regions.
- **Opacity correction**: Low-opacity points are corrected using nearby converged Gaussians, reducing dark holes caused by occlusion and insufficient supervision.
- **Semantic sky constraint**: A sky mask suppresses foreground Gaussian leakage into sky regions, while sky-color statistics stabilize the overall background tone.

These completion operations primarily target directions unobserved by perspective cameras and are the most important implementation difference between `HTGS_bgm` and the panoramic version.

## Background Models

| Configuration value | Background representation |
| --- | --- |
| `SH` | Global spherical harmonics |
| `MLP` | MLP that predicts color from the viewing direction |
| `TEX` | Learnable spherical environment texture map |
| `MSI` | Multi-Sphere Image |
| `HG` | Single-layer environment Gaussians |
| `VET` | Multi-layer environment Gaussians |

All background models are composited with the HTGS foreground result through foreground transmittance. HG and VET render their environment Gaussians using the perspective rasterization backend from HTGS.

## SplatEnv Extensions

Compared with the original HTGS implementation, this directory adds:

- six interchangeable explicit background representations;
- SegFormer-based sky-mask generation;
- dominant sky-color and complexity estimation;
- sky-aware foreground suppression;
- color propagation and opacity correction for unobserved directions;
- Gaussian count and scale adjustment based on scene scale and sky complexity;
- staged background optimization, with background updates enabled at the beginning and end of training and frozen during the middle stage.

## NeRFICG 2.0

This version supports **NeRFICG 2.0** and uses its unified dataset API. The additional dataloader and `move_dataloader.sh` required by the original Project Thesis version are no longer needed.

## Installation

Install NeRFICG 2.0 first, then run the following commands from the repository root:

```bash
cp -r src/Methods/PA_SplatEnv/HTGS_bgm src/Methods/HTGS_bgm
python scripts/install.py -m HTGS_bgm
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
    -m HTGS_bgm \
    -d <DATASET_TYPE> \
    -o <CONFIG_NAME>
```

In the generated YAML file, set `BACKGROUND_MODEL` to `SH`, `MLP`, `TEX`, `MSI`, `HG`, or `VET`, then configure the dataset path and other training parameters.

Start training:

```bash
python scripts/train.py -c configs/<CONFIG_NAME>.yaml
```

### HG / VET Configuration

The Gaussian environment models in the perspective-camera version provide the following key parameters:

| Parameter | Description |
| --- | --- |
| `HALF_BALL` | Use hemisphere initialization; recommended for perspective scenes |
| `USE_AUTO_CONFIG` | Automatically configure capacity and scale from scene scale and sky complexity |
| `BASE_GAUSSIANS` | Base number of environment Gaussians |
| `NUM_LAYERS` | Number of spherical shells for VET; HG always uses one layer |
| `BETA_INTERVAL` | Radial spacing between adjacent spherical shells |
| `MIN_SCALE_RATIO` | Relative Gaussian scale near the horizon |
| `MAX_SCALE_RATIO` | Relative Gaussian scale at high latitudes |
| `KNN_COLOR` | Neighborhood size used for color propagation |

When `USE_AUTO_CONFIG` is enabled, the Gaussian count and scale are adjusted automatically according to the scene radius and sky complexity.

## Upstream Project

This implementation is based on the official HTGS project:

**Efficient Perspective-Correct 3D Gaussian Splatting Using Hybrid Transparency**<br>
Florian Hahlbohm et al.<br>
[Project repository](https://github.com/nerficg-project/HTGS)

The Gaussian representation, perspective-correct rasterization, and CUDA backend in HTGS belong to the upstream project. The SplatEnv background models, semantic sky prior, and related extensions were implemented by **Jiahao Chen**.
