# AD-GBC
The official implementation of the paper "AD-GBC: Anisotropic Granular-Ball Skip-Connection Refiner for UNet-Based Medical Image Segmentation" accepted in CVPR-2026.

### Abstract
Prototype or region-attention modules have recently improved medical image segmentation but still suffer from two fundamental limitations: 1) they represent each semantic concept as a point or isotropic region, failing to capture the inherently anisotropic geometry of real feature distributions; and 2) many rely on non-differentiable clustering or one-way kernel weighting, which restricts their ability to form coherent region-level representations. We address these issues with the Anisotropic Differentiable Granular-Ball (AD-GBC) module, which generalizes prototypes into learnable geometric regions parameterized by a center and an anisotropic vector scale. AD-GBC aggregates local features into region-level semantics and redistributes the refined representation back to pixels in a fully differentiable manner, enabling geometry-aware refinement within modern UNet-style architectures. Two geometric regularizers, a Wasserstein-based diversity loss and a scale consistency loss, mitigate center collapse and encourage stable, well-formed region geometry.AD-GBC yields consistent improvements across four widely used medical segmentation benchmarks (BUSI, GlaS, CVC-ClinicDB, ISIC17) when integrated into two strong backbones (Rolling-UNet and U-KAN), demonstrating that the proposed geometric region formulation generalizes well across different imaging conditions.

### Motivation
<p align="center">
  <img src="imgs/motivation.png" width="400"/>
</p>

### Overview
<p align="center">
  <img src="imgs/overview of AD-GBC integration.png" width="400"/>
</p>


### Datasets
1) BUSI - [Link](https://www.kaggle.com/aryashah2k/breast-ultrasound-images-dataset)
2) GlaS - [Link](https://websignon.warwick.ac.uk/origin/slogin?shire=https%3A%2F%2Fwarwick.ac.uk%2Fsitebuilder2%2Fshire-read&providerId=urn%3Awarwick.ac.uk%3Asitebuilder2%3Aread%3Aservice&target=https%3A%2F%2Fwarwick.ac.uk%2Ffac%2Fcross_fac%2Ftia%2Fdata%2Fglascontest&status=notloggedin)
3) CVC-ClinicDB - [Link](https://polyp.grand-challenge.org/CVCClinicDB/)
4) ISIC 2017 - [Link](https://challenge.isic-archive.com/data/)


### Data Format
- Make sure to put the files as the following structure. For binary segmentation, just use folder 0.
```
inputs
└── <dataset name>
    ├── images
    |   ├── 001.png
    │   ├── 002.png
    │   ├── 003.png
    │   ├── ...
    |
    └── masks
        └── 0
            ├── 001.png
            ├── 002.png
            ├── 003.png
            ├── ...
```


### Training and Validation
- Train the model.
```
python train_GBC.py \
  --dataset busi \
  --name busi_RU_GBC_L_div0.01_sca0.1 \
  --arch GBC_Rolling_Unet_L \
  --loss 'BCEDiceWithGeometryLoss' \
  --div_weight 0.01 \
  --scale_weight 0.1
```
```
python train_GBC.py \
  --dataset glas \
  --name glas_RU_GBC_L_div0.1_sca0.1 \
  --arch GBC_Rolling_Unet_L \
  --loss 'BCEDiceWithGeometryLoss' \
  --div_weight 0.1 \
  --scale_weight 0.1
```
```
python train_GBC.py \
  --dataset cvc \
  --name cvc_RU_GBC_L_div0.1_sca0.1 \
  --arch GBC_Rolling_Unet_L \
  --loss 'BCEDiceWithGeometryLoss' \
  --div_weight 0.1 \
  --scale_weight 0.1 \
  --dataseed 6142
```

- Evaluate.
```
python val_GBC.py --name busi_RU_GBC_L_div0.01_sca0.1
```

## Citations
If this code is helpful for your study, please cite:
```
X. Shen, Q. Zhao, and L. Feng, “AD-GBC: Anisotropic granular-ball skip-connection refiner for UNet-based medical image segmentation,” Accepted to IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2026.
```
