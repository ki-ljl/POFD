# Public Opinion Field Effect Fusion in Representation Learning for Trending Topics Diffusion (NeurIPS 2023)
Code for NeurIPS 2023 paper---**P**ublic **O**pinion **F**ield Effect Fusion in Representation Learning for Trending Topics **D**iffusion.
![poster](https://github.com/ki-ljl/POFD/assets/56509367/47253ff7-7c04-4fb8-904b-c64b945d25d8)

## Overview
```bash
POFD:.
│  args.py
│  get_data.py
│  pytorchtools.py
│  requirements.txt
│          
├─data
│  ├─BuzzFeed
│  │      
│  ├─DBLP
│  │              
│  ├─PolitiFact
│  │      
│  └─Twitter
│          
└─src
    │  infomax.py
    │  lp_main.py
    │  models.py
    │  nc_dblp_main.py
    │  nc_main.py
    │  util.py
    │  
    └─models   
```
1. **args.py**: args.py is the parameter configuration file, including model parameters and training parameters.
2. **get_data.py**: This file is used to load the data.
3. **pytorchtools**.py: This file is used to define the earlystopping mechanism.
4. **requirements.txt**: Dependencies file.
5. **data/**：Dataset folder.
6. **src/infomax.py**: This file is used to maximize the information, i.e., to calculate $L_p$.
7. **src/lp_main.py**: Public opinion concern prediction.
8. **src/models.py**: POFD implementation.
9. **src/nc_dblp_main**.py: Universality analysis.
10. **src/nc_main.py**: Event classification.
11. **src/util.py**: Defining various toolkits.

Due to the file size limitation, we only give the BuzzFeed dataset, and the complete dataset will be fully publicly available after the paper is accepted. You can also download the original dataset at [kaggle](https://www.kaggle.com/datasets/mdepak/fakenewsnet).
## Dependencies
Please install the following packages:
```
gensim==3.8.3
huggingface-hub==0.12.1
joblib==1.2.0
matplotlib==3.6.3
networkx==2.8.8
node2vec==0.3.3
numpy==1.22.4
pandas==1.3.3
scikit-learn==1.2.1
scipy==1.8.0
torch==1.12.1+cu113
torch-cluster==1.6.0+pt112cu113
torch-geometric==2.2.0
torch-scatter==2.1.0+pt112cu113
torch-sparse==0.6.16+pt112cu113
torch-spline-conv==1.2.1+pt112cu113
tqdm==4.62.3
transformers==4.26.1
```
You can also simply run:
```
pip install -r requirements.txt
```

## Usage
### Public Opinion Concern Prediction
```bash
cd src/
python lp_main.py --dataset BuzzFeed
python lp_main.py --dataset PolitiFact
python lp_main.py --dataset Twitter
```

### Event Classification
```bash
cd src/
python nc_main.py --dataset BuzzFeed
python nc_main.py --dataset PolitiFact
python nc_main.py --dataset Twitter
```

### Universality Analysis
```bash
cd src/
python nc_dblp_main.py
```
