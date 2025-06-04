# Rewind: Cloud Detection Challenge - Using TerraTorch
In this repository, I rewound the challenge '[On Cloud N](https://www.drivendata.org/competitions/83/cloud-cover/): Cloud Cover Detection Challenge' by using [TerraTorch](https://ibm.github.io/terratorch/stable/), 'Fine-tuning Framework for Geospatial Foundation Models'.

The original challenge was posted on DRIVENDATA and sponsored by Microsoft.

## Environmetal Setup
```
conda env create -f environment.yaml
```

## Dataset
The original dataset used from the challenge is quite large. Therefore, in this experiment, I reduced the dataset size by 5% by randomly selecting samples from each location. The reduced dataset is uploaded to HuggingFace.

- [Original](https://source.coop/repositories/radiantearth/cloud-cover-detection-challenge/description)
- [Small Size](https://huggingface.co/datasets/hk-kaden-kim/Small_S2_CloudCover_Seg)
- [Data Overview](./data_analysis.ipynb)

## Train / Test

| Encoder\Decoder    | FCN | UNET |
| -------- | ------- | ------- |
| ResNet 34       | r34_fcn                     | r34_unet  |
| ResNet 50       | r50_fcn                     | r50_unet  |
| Prithvi v2 100M | p100_fcn                    | p100_unet |
| Prithvi v2 300M | p300_fcn                    | p300_unet |

FCN: Fully Convolution Network

- [Trainig Script](./training.py)
- [Test Script](./test.py)
- Trained on Amazon EC2 G5 Instances
- Three different versions of each model were configured
  - No Fine-tuning
  - Decoder Fine-tuning
  - Encoder and Decoder Fine-tuning

## Analaysis
- [Model Analysis](./model_analysis.ipynb)
