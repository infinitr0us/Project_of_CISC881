# Project_of_CISC881: Attention-guided Medical Imaging Classification
Code of project for CISC881: attention-guided medical imaging classification, Implemented by [Yuchuan Li](https://github.com/infinitr0us)

## Preparation
### Environmental setup
Anaconda is used for this implementation. Two enviromental set-up file could be found with the code.
1. environment.yaml: exported by the Anaconda directly
```
conda env create -n [env_name] -f environment.yaml
```
2. requirements.txt: exported by the pip directly 
```
pip install -r requirements.txt
```
All of the functions here is validated on my PC with: Windows 10 Pro 21H1 Version 19043.1586, Windows Feature Experience Pack 120.2212.4170.0;
CPU I9-10900K, GPU RTX 2080 Ti

### Dataset Preparation
2 datasets are used here. They are [MedMNIST v2](https://github.com/MedMNIST/MedMNIST) and [Chaoyang](https://github.com/bupt-ai-cz/HSA-NRL/tree/9d404dd671f675c3b3bd8f430c5708a7a35ae57d) dataset. Fortunately, we do not need data pre-processing here.
1. MedMNIST v2: It is easier here if you have done the environmental setup, the dataset would be downloaded automatically during the initial training process
2. Chaoyang dataset: You need to fill the form on their [official website](https://bupt-ai-cz.github.io/HSA-NRL/) and then download it.

## Training
Please carefully check the args in train.py and modify them for your needs. 
A general usage of it is:
```
set CUDA_VISIBLE_DEVICES=[CUDA_DEVICES] & python main.py --att [ATTENTION_MECHANISM] --dataset [DATASET] --batch-size [BATCH_SIZE] --learning-rate [LEARNING_RATE] --pretrained
```

## Visualization
The Visualization function is implemented based on the [cache-mechanism](https://github.com/luo3300612/Visualizer)
A general usage of it is:
```
set CUDA_VISIBLE_DEVICES=[CUDA_DEVICES] & python visualization.py --dataset [DATASET] --att [ATTENTION_MECHANISM] --load [CHECKPOINT_DIR]
```
