# DBNAIC
This repository contains the reference code for the paper "Dual Branch Non-Autoregressive Image Captioning".

# Requirements
- Python 3.7
- Pytorch 1.12
- Torchvision 0.13
- timm 0.6.11
- numpy
- tqdm

# Data preparation
The necessary files in training and evaluation are saved in `mscoco` folder, which is organized as follows:
```python
mscoco/
|--feature/
  |--region/
  |--grid/
```
Download the image features from [GoogleDrive](https://drive.google.com/file/d/1EP9EB8OYoz7VT29g6ARwwi8uCZD8kvOS/view?usp=sharing) and put into `./mscoco/`. Download the files from [GoogleDrive](https://drive.google.com/file/d/1Z7iXdm602tEqz3fbqWO-bfamAy_Yxacz/view?usp=sharing) and put into `./evaluation/`.

# Training
```python
python train.py
```

# Evaluation
You can download the pre-trained model from [GoogleDrive](https://drive.google.com/file/d/1co3_Gvi-Vs64dkMhdTvFnWuuwYA3JjFO/view?usp=sharing) and put it into `./saved_models/`.
```python
python eval.py
```
| BLEU-1      | BLEU-4      | METEOR      | ROUGE-L     | CIDEr       |     
| ----------- | ----------- | ----------- | ----------- | ----------- |
| 81.7        | 39.5        | 28.8        | 59.4        | 128.8       |
