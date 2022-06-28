import warnings
from train import Train_multi
from utils import plot_auc_curves, plot_prc_curves

import argparse

if __name__ == '__main__':
    warnings.filterwarnings("ignore") # 忽略警告信息

    parser = argparse.ArgumentParser() #获取命令行参数
    parser.add_argument("--epochs", default=1000, type=int) #轮次
    parser.add_argument("--attn_size", default=128, type=int,help="") #特征注意力映射维度 #default 64
    parser.add_argument("--attn_heads", default=4,type=int) # 头 #default 8
    parser.add_argument("--out_dim", default=256, type=int) #GAT输出维度 # default 64
    parser.add_argument("--dropout", default=0.3,type=float) # 丢失率
    args = parser.parse_args() #读取命令行

    fprs, tprs, auc, precisions, recalls, prc = Train_multi(directory='data', # 数据路径,相对路径即可，当前文件同级下的文件直接开头
                                                      epochs=args.epochs, #原代码：1000
                                                      attn_size=args.attn_size, # feature_attn_size，元子图注意力层的特征输出维度
                                                      attn_heads=args.attn_heads, #多头注意力头数
                                                      out_dim=args.out_dim, # 模型的输出特征维度，元子图语意层的输出维度
                                                      dropout=args.dropout, # 丢失率，（还有一个元子图语意层的注意力向量q维度）
                                                      slope=0.2, # Leakeyrelu的参数
                                                      lr=0.001, #学习率
                                                      wd=5e-3,
                                                      random_seed=1000,#设置随机种子
                                                      cuda=True,
                                                      model_type='HGANMDA')
    # 参数说明
    # fprs{list:5 narray} [0.930340066240612, 0.9335017027743009, 0.9353000347332208, 0.937262300990015, 0.9325944658066214]
    # tprs{list:5 narray}
    # precisions{list:5 narray}  recalls{list:5 narray}
    # auc{lsit:5 float} prc {lsit:5 float} [0.9211309563122542, 0.927568090118527, 0.9393899679851035, 0.9310728717835131, 0.9264964732172374]


    #
    # fprs[5(五折)*narray(878测试集的样本数目)]
    # fprs[5*narray]

    # auc_result[5*1]
    # prc_result[5*1]

    # presion[5(五折)*narray(3885 个数随机，代码自动根据threshold划分)]
    # recalls[5*narray(3885)]

    # 绘制图片
    plot_auc_curves(fprs, tprs, auc, directory='roc_result', name='test_auc') # 绘制roc-auc,单分类
    plot_prc_curves(precisions, recalls, prc, directory='roc_result', name='test_prc') # 绘制pr-auc，单分类
