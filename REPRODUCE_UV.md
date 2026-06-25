# AD-GBC uv Reproduction Notes

## Environment

The project keeps a shared Python environment at the workspace root:

```bash
cd /Volumes/SAMSUNG/PhD/Granular_Ball_segmentation
.venv/bin/python -c "import torch; print(torch.__version__)"
```

If the environment needs to be rebuilt, create it at the project root and
install AD-GBC dependencies from `repos/AD-GBC`:

```bash
uv venv --python 3.11 .venv
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==2.5.1 torchvision==0.20.1
.venv/bin/python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r repos/AD-GBC/requirements.txt
```

On this macOS arm64 machine, importing OpenCV from the wheel can hang while
loading the native `cv2` extension. The local training and validation scripts
use PIL-based image IO and lightweight transforms, so OpenCV/albumentations are
not needed at runtime.

If `import cv2` hangs after installing the original requirements, keep only the
headless OpenCV wheel:

```bash
.venv/bin/python -m pip uninstall -y opencv-python opencv-python-headless
.venv/bin/python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python-headless==4.10.0.84
```

## Shared Dataset Root

All methods should use the root-level shared datasets:

```text
../../Dataset/segmentation/<dataset>/images
../../Dataset/segmentation/<dataset>/masks/0
```

AD-GBC resolves this path automatically, or you can pass it explicitly:

```bash
--data_root ../../Dataset/segmentation
```

For fixed train/validation splits shared across methods, use:

```text
../../Dataset/splits/segmentation/<dataset>/seed_<seed>_train.txt
../../Dataset/splits/segmentation/<dataset>/seed_<seed>_val.txt
```

Generated checkpoints, logs, predictions, and TensorBoard runs are stored
outside the code repository:

```text
../../Artifacts/AD-GBC/
```

## Smoke Reproduction

`../../Dataset/segmentation/busi_smoke` contains five samples for a local
pipeline check.

```bash
../../.venv/bin/python train_GBC.py \
  --dataset busi_smoke \
  --name smoke_adgbc_cpu \
  --data_root ../../Dataset/segmentation \
  --artifact_root ../../Artifacts/AD-GBC \
  --arch GBC_Rolling_Unet_S \
  --loss BCEDiceWithGeometryLoss \
  --div_weight 0.01 \
  --scale_weight 0.01 \
  --epochs 1 \
  --batch_size 1 \
  --num_workers 0 \
  --input_w 64 \
  --input_h 64 \
  --gbc_num_balls 4 \
  --device cpu
```

```bash
../../.venv/bin/python val_GBC.py \
  --name smoke_adgbc_cpu \
  --data_root ../../Dataset/segmentation \
  --artifact_root ../../Artifacts/AD-GBC \
  --device cpu
```

This confirms model construction, AD-GBC forward/backward, geometry loss,
checkpoint saving, validation loading, and output writing.

## Full Paper Reproduction

Download the complete datasets from the README links and place raw files under
`../../Dataset/raw/`, then run:

```bash
../../.venv/bin/python scripts/prepare_public_datasets.py
```

This produces `../../Dataset/segmentation/<dataset>/images` and
`../../Dataset/segmentation/<dataset>/masks/0`. Then run the official commands,
for example:

```bash
../../.venv/bin/python train_GBC.py \
  --dataset busi \
  --name busi_RU_GBC_L_div0.01_sca0.1 \
  --data_root ../../Dataset/segmentation \
  --artifact_root ../../Artifacts/AD-GBC \
  --arch GBC_Rolling_Unet_L \
  --loss BCEDiceWithGeometryLoss \
  --div_weight 0.01 \
  --scale_weight 0.1 \
  --device auto
```

For exact CVPR results, use the paper settings: full datasets, the agreed fixed
splits under `Dataset/splits`, 400 epochs, batch size 8, and a CUDA GPU
comparable to the reported single NVIDIA A100 setup.
