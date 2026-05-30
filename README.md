# monocular-depth-estimation

## Repository structure

```md
monocular-depth-estimation/
├── minutes/
└── references/
    └── res.bib
```

# To start the project
Check that you have python 3.12 and uv installed on your machine already, then create
and start a virtual environment: 
```bash
uv venv
source .venv/bin/activate
```
How to add dependencies to the project:
```
uv add torch
```

## Running monocular depth estimation

```bash
# Depth Anything V2 on the perspective ZED camera (default model)
uv run python run_depth.py --data_dir ../ --sensor ZED_B

# Depth Anything V2 on the fisheye camera (auto valid-region masking)
uv run python run_depth.py --data_dir ../ --sensor G1_A
```

Outputs land in `outputs/`: an RGB copy, a colorized depth visualization, a raw
`.npy` depth array, and (for fisheye) a binary valid-region mask.

### Depth Any Camera (DAC) — metric depth for fisheye

DAC ([Guo et al., CVPR 2025](https://github.com/yuliangguo/depth_any_camera)) gives
zero-shot **metric** depth and handles fisheye intrinsics natively. One-time setup:

```bash
git clone https://github.com/yuliangguo/depth_any_camera.git third_party/depth_any_camera
```

The compiled deformable-attention C++ op is **not** required (the ResNet101 configs
use `attn_dec=false`); it is mocked automatically. Weights download from HuggingFace
on first run (~700 MB, cached afterwards).

```bash
# DAC outdoor ResNet101 on the fisheye camera
uv run python run_depth.py --data_dir ../ --sensor G1_A --model dac-outdoor-resnet101

# Other variants
uv run python run_depth.py --data_dir ../ --sensor G1_A --model dac --variant dac-indoor-resnet101
```

Available variants: `dac-outdoor-resnet101`, `dac-outdoor-swinl`,
`dac-indoor-resnet101`, `dac-indoor-swinl`. For indoor scenes the `indoor` variants
are better calibrated; `outdoor` variants (trained on KITTI) tend to overestimate
indoor distances.

