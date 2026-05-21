# Train PyTorch models on GTSRB dataset

This codebase is modified from https://github.com/huyvnphan/PyTorch_CIFAR10 to adapt Germany Traffic Sign Recognition Benchmark(GTSRB) dataset.

## Download GTSRB dataset.

- The official website is https://benchmark.ini.rub.de/gtsrb_news.html. 
- However, the connection to this website is not stable. Instead you can also download this dataset from [Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) or [HuggingFace](https://huggingface.co/datasets/tanganke/gtsrb)
- You can also load it directly from torchvision.dataset

## Install Environment

The running environment is validated on 4090 GPUs with CUDA 12.1
```bash
conda create -n gtsrb python=3.10 -y
conda activate gtsrb
pip install -r requirements.txt
```


## How to train models from scratch
Check the `train.py` to see all available hyper-parameter choices.
To reproduce the same accuracy use the default hyper-parameters

`python train.py --classifier resnet18 --data_dir ./data/GTSRB`