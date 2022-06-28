# MSHGANMDA
MSHGANMDA is a new method for miRNA-disease association prediction.

## Environments
- Python 3.6.13
- dgl 0.5.1
- numpy 1.19.5
- torch 1.9.0
- pandas 0.25.1

## Dataset
### M-d association
miRNA-disease associations 
### Multi-types m-d association
wfy_multi_all_mirna_disease_pairs_without_negative 
### Node Features
disease features : D_SSM1,D_SSM2,D_GSM   
miRNA features :   M_FSM,M_GSM  


## How to run?
```
python main.py 
```

# If you refer to our study, please cite
@article{  
   author = {Wang, Shudong and Wang, Fuyu and Qiao, Sibo and Zhuang, Yu and Zhang, Kuijie and Pang, Shanchen and Nowak, Robert and Lv, Zhihan},  
   title = {MSHGANMDA: Meta-Subgraphs Heterogeneous Graph Attention Network for miRNA-Disease Association Prediction},  
   journal = {IEEE Journal of Biomedical and Health Informatics},  
   pages = {1-10},  
   year = {2022}  
}

