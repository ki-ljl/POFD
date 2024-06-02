# POFD
Code for NeurIPS 2023 paper---**P**ublic **O**pinion **F**ield Effect Fusion in Representation Learning for Trending Topics **D**iffusion.

# Overview
```bash
POFD:.
│  get_data.py
│  pytorchtools.py
│  requirements.txt
│          
├─data
│  ├─BuzzFeed
│  │      
│  ├─DBLP
│  │      
│  └─PolitiFact
│          
└─src
    │  infomax.py
    │  lp_main.py
    │  models.py
    │  nc_dblp_main.py
    │  nc_main.py
    │  util.py
    │  
    └─checkpoints   
```
1**get_data.py**: This file is used to process the data.
2**pytorchtools**.py: This file is used to define the early_stopping mechanism.
3**requirements.txt**: Dependencies file.
4**data/**：Dataset folder.
5**src/infomax.py**: This file is used to maximize the information, i.e., to calculate $L_p$.
6**src/lp_main.py**: Public opinion concern prediction (**Section 4.2**).
7**src/models.py**: POFD implementation.
8**src/nc_dblp_main**.py: Universality analysis (**Section 4.4**).
9**src/nc_main.py**: Event classification (**Section 4.3**).
0**src/util.py**: Defining various toolkits.

Since github limits the size of uploaded files, you can get the full dataset from [Google Cloud Drive](https://www.kaggle.com/datasets/mdepak/fakenewsnet).

# Dependencies
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
# Public Opinion Concern Prediction
```bash
cd src/
python lp_main.py --dataset BuzzFeed
python lp_main.py --dataset PolitiFact
python lp_main.py --dataset Twitter
```

# Event Classification
```bash
cd src/
python nc_main.py --dataset BuzzFeed
python nc_main.py --dataset PolitiFact
python nc_main.py --dataset Twitter
```

# Universality Analysis
```bash
cd src/
python nc_dblp_main.py
```
