# Hierarchical graph attention network for miRNA-disease associations prediction

## Environments
- Python 3.6.13
- dgl 0.5.1
- numpy 1.19.5
- torch 1.9.0
- pandas 0.25.1

## Dataset
### M-d association
miRNA-disease associations  基因疾病关联
### Multi-types m-d association
wfy_multi_all_mirna_disease_pairs_without_negative  # multi-types associations
### Node Features
disease features : D_SSM1,D_SSM2,D_GSM
miRNA features :   M_FSM,        M_GSM  数据源自IMCMDA

D_SSM1,M_FSM  方法源自

@article{
   author = {Wang, D. and Wang, J. and Lu, M. and Song, F. and Cui, Q.},
   title = {Inferring the human microRNA functional similarity and functional network based on microRNA-associated diseases},
   journal = {Bioinformatics},
   volume = {26},
   number = {13},
   pages = {1644-50},
   year = {2010}
}
D_SSM2 方法源自

@misc{
   author = {Chen, Xing and Wang, Lei and Qu, Jia and Guan, Na-Na and Li, Jian-Qiang},
   title = {Predicting miRNA-disease association based on inductive matrix completion},
   volume = {34},
   number = {24},
   pages = {4256-4265},
   month = {Dec 15},
   year = {2018}
}

D_GSM M_GSM 方法源自 

@article{
   author = {Chen, Xing and Yan, Chenggang Clarence and Zhang, Xu and You, Zhu-Hong and Huang, Yu-An and Yan, Gui-Ying},
   title = {HGIMDA: Heterogeneous graph inference for miRNA-disease association prediction},
   journal = {Oncotarget},
   volume = {7},
   number = {40},
   pages = {65257-65269},
   year = {2016}
}



## How to run?
```
python main.py 
```
