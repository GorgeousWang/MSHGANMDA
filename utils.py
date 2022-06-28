import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interp
from sklearn import metrics
import torch
import torch.nn as nn
import dgl

# 数据读取
def load_data(directory, random_seed):
    # 读特征
    D_SSM1 = np.loadtxt(directory + '/D_SSM1.txt') # 疾病相似度1ndarray[383,383] ,np.loadtxt默认dtype=float
    D_SSM2 = np.loadtxt(directory + '/D_SSM2.txt') # 疾病相似度2ndarray[383,383]
    D_GSM = np.loadtxt(directory + '/D_GSM.txt') # 疾病高斯相似度1ndarray[383,383]
    M_FSM = np.loadtxt(directory + '/M_FSM.txt')# mirna功能相似度[495,495]
    M_GSM = np.loadtxt(directory + '/M_GSM.txt')# mirna高斯相似度[495,495]

    # 读多分类的基因疾病关联数据
    multi_md_associations1 = pd.read_csv(directory + '/wfy_multi_all_mirna_disease_pairs_without_negative.csv', names=['miRNA', 'disease', 'label']) #mirna-disease关联 dataFrame【189585，3】正样本5430
    # multi_md_associations1['label'] = multi_md_associations1['label']
    # 读全部的基因疾病关联数据（负样本用0表示，正样本用1表示）
    all_associations = pd.read_csv(directory + '/all_mirna_disease_pairs.csv', names=['miRNA', 'disease', 'label']) #mirna-disease关联 dataFrame【189585，3】正样本5430

    D_SSM = (D_SSM1 + D_SSM2) / 2 # 疾病相似度2个特征汇总成1个
    ID = D_SSM #疾病语意相似度nadarry【383，383】，代表整合的疾病语意相似度，还没有用高斯进行填充
    IM = M_FSM #基因功能相似度nadarry【495，495】，代表整合的基因功能相似度，还没有用高斯核相似度进行填充
    for i in range(D_SSM.shape[0]): # 根据16、17公式进行相似度整合
        for j in range(D_SSM.shape[1]):
            if ID[i][j] == 0:
                ID[i][j] = D_GSM[i][j] # 为0的部分，用高斯相似度进行补充
    for i in range(M_FSM.shape[0]):
        for j in range(M_FSM.shape[1]):
            if IM[i][j] == 0:
                IM[i][j] = M_GSM[i][j]

# 筛选miRNA-disease正样本和与正样本数相同的负样本
#     known_associations = all_associations.loc[all_associations['label'] == 1] # df.loc按照行取，从189585得5430
    known_associations = all_associations.loc[all_associations['label'] != 0] # df.loc按照条件行取，从189585得5430 ,由于可能做多分类，所以md关联不能用==1了
    unknown_associations = all_associations.loc[all_associations['label'] == 0] # df.loc按条件行取所有负样本
    random_negative = unknown_associations.sample(n=known_associations.shape[0], random_state=random_seed, axis=0) # df.sample 按照行随机采样与正样本等量的负样本，axis=0表示按行取
    sample_df = known_associations.append(random_negative) # 拼接m-d关联的正样本和负样本10860

    #df索引重置，因为df随机采样后，会保留原本的index索引，即超过10860个数，而且有随机性，因此需要重置
    sample_df.reset_index(drop=True, inplace=True) # df.reset_index 打乱m-d样本，drop=True代表需要将重置的索引作为新列插入到dataframe当中，插入也是插入到index当中，不影响数据
    #wfy
    multi_md_associations1.reset_index(drop=True,inplace=True) # 打乱multi_md
    multi_md_associations = multi_md_associations1.values
    samples = sample_df.values      # 获得重新编号index的新样本的值，从df中取numpy数组
    return ID,IM,multi_md_associations,samples # 未知关联数量较多，选择和已知关联数目相同的未知关联组成样本，返回 整合的D,M，，和md的关联numpy数组

# miRNA-disease异质图和miRNA-disease-lncRNA异质图的构建
def build_graph(directory, random_seed):
    # 加载数据集返回 2个特征narray矩阵ID【383，383】,IM【495，495】和多关联样本【1451，3】和采样后的全部正负【10860，3】样本
    ID,IM,multi_md_associations, samples = load_data(directory,random_seed)
    # miRNA-disease二元异质图
    # 1.1 创建单一md关联二质图并载入节点特征
    g = dgl.DGLGraph()#先实例化一个dgl图，dgl.DGLGraph()是利用默认的构造器直接实例化一个DGLGraph类
    # g = dgl.graph([]) # dgl.graph([])是用dgl中的建图操作创建DGLGraph对象
    g.add_nodes(ID.shape[0] + IM.shape[0]) # 节点个数等于m+d的个数383+495
    node_type = torch.zeros(g.number_of_nodes(), dtype=torch.int64) #dgl推荐tensor作为其输入,节点类型首先设置为全0
    node_type[: ID.shape[0]] = 1 # 把前383个节点的类型设置为1，即disease，后面剩下的495个节点的类型为0，即mirna
    g.ndata['type'] = node_type # 节点的类型，用节点的属性g.ndata['type']表示
    # 传入前383个疾病节点特征
    d_sim = torch.zeros(g.number_of_nodes(), ID.shape[1])
    d_sim[: ID.shape[0], :] = torch.from_numpy(ID.astype('float32')) # 给不同类型的节点添加不同特征时，按照节点需要填充tensor矩阵，其余部分用0填充，根据g.ndata['']内的名字来进行划分
    g.ndata['d_sim'] = d_sim #传入疾病节点的特征，
    # 传后495个，383-877，miRNA节点特征
    m_sim = torch.zeros(g.number_of_nodes(), IM.shape[1])
    m_sim[ID.shape[0]: ID.shape[0]+IM.shape[0], :] = torch.from_numpy(IM.astype('float32'))
    g.ndata['m_sim'] = m_sim

    # 1.2 载入边字典
    # 让指针从0开始，原本边数据中的节点标签从1开始
    disease_ids = list(range(1, ID.shape[0]+1))#得到list【1-383】
    mirna_ids = list(range(1, IM.shape[0]+1))# 得到list【1-495】
    disease_ids_invmap = {id_: i for i, id_ in enumerate(disease_ids)} #{i：id_}，从1-383转0-382
    mirna_ids_invmap = {id_: i for i, id_ in enumerate(mirna_ids)} # #{i：id_}，从1-495转0-494
    # 提取边向量的顶点
    sample_disease_vertices = [disease_ids_invmap[id_] for id_ in samples[:, 1]] # 从m-d中取d，并将下标转为0-382 疾病list【】
    sample_mirna_vertices = [mirna_ids_invmap[id_] + ID.shape[0] for id_ in samples[:, 0]] # 从m-d中取m，并将下标转为383-877

    # 给图添加边，因为是边预测，需要给边打上标签数据data={'label':...}
    g.add_edges(sample_disease_vertices, sample_mirna_vertices,
                data={'label': torch.from_numpy(samples[:, 2].astype('float32'))}) #标签和特征都需要是tensor格式的
    g.add_edges(sample_mirna_vertices, sample_disease_vertices, #dgl构建无向图就是，添加相反的同一组边
                data={'label': torch.from_numpy(samples[:, 2].astype('float32'))})

    g.readonly() # 设置图不可变
    # g2.readonly()


# multi-types of miRNA-disease 多类关联异质图
    # 1.1建图加入节点，异质图设置节点类型
    g0 = dgl.DGLGraph()
    # g0.add_nodes(ID.shape[0] + IM.shape[0] + IL.shape[0])
    g0.add_nodes(ID.shape[0] + IM.shape[0])
    node_type = torch.zeros(g0.number_of_nodes(), dtype=torch.int64) # 返回一个878全为0的tensor 878+467=1345
    node_type[: ID.shape[0]] = 1            # disease383标记为1，miRNA标记为2
    node_type[ID.shape[0] + IM.shape[0]:] = 2
    g0.ndata['type'] = node_type             # 将图中疾病的节点记为1,miRNA标记为2
    # 1.2 异质图设置节点特征
    d_sim = torch.zeros(g0.number_of_nodes(), ID.shape[1])       # （1345，383）
    d_sim[: ID.shape[0], :] = torch.from_numpy(ID.astype('float32'))
    g0.ndata['d_sim'] = d_sim

    m_sim = torch.zeros(g0.number_of_nodes(), IM.shape[1])       # （1345,495）
    m_sim[ID.shape[0]: ID.shape[0]+IM.shape[0], :] = torch.from_numpy(IM.astype('float32'))
    g0.ndata['m_sim'] = m_sim        # 每一行表示一个miRNA的特征

    #wfy 提取边向量的顶点，提取边信息，注意边和节点的序号都是从0开始
    multi_disease_vertices = [disease_ids_invmap[id_] for id_ in multi_md_associations[:, 1]] #转为0-382的list
    multi_mirna_vertices = [mirna_ids_invmap[id_] + ID.shape[0] for id_ in multi_md_associations[:, 0]] # 从m-d中取m，并将下标转为383-877

    # 添加同质所有关联边，m-d 和 d-m
    g0.add_edges(sample_disease_vertices, sample_mirna_vertices,         # 0-10859 , 其中sample_disease_vertices 是list
                data={'dm': torch.from_numpy(samples[:, 2].astype('float32'))})
    g0.add_edges(sample_mirna_vertices, sample_disease_vertices,         # 10860-21719
                data={'md': torch.from_numpy(samples[:, 2].astype('float32'))})

    # wfy
    g0.add_edges(multi_disease_vertices, multi_mirna_vertices,
                data={'multi_label': torch.from_numpy(multi_md_associations[:, 2].astype('float32'))}) #标签和特征都需要是tensor格式的
    g0.add_edges(multi_mirna_vertices, multi_disease_vertices, #dgl构建无向图就是，添加相反的同一组边
                data={'multi_label': torch.from_numpy(multi_md_associations[:, 2].astype('float32'))})

    g0.readonly()
    # 返回二元异质图dgl，三元异质图dgl，所有样本的疾病节点向量list，所有样本的的基因节点向量list，ndarray疾病特征，基因特征，l基因特征，ndarray所有样本，ndarray [19,3]:ml样本，[677,3]:ld样本
    # return g, g0,g2, sample_disease_vertices, sample_mirna_vertices, ID, IM, samples
    return g, g0, sample_disease_vertices, sample_mirna_vertices, ID, IM, samples



def weight_reset(m):
    if isinstance(m, nn.Linear):# 返回一个对象是否是一个类或者子类的实例，
        m.reset_parameters()  # 参数初始化


# 参数说明
# fprs[5(五折)*narray(878测试集的样本数目)]
# fprs[5(五折)*narray]

# auc_result[5*1]
# prc_result[5*1]

# presion[5(五折)*narray(3885 个数随机，代码自动根据threshold划分)]
# recalls[5*narray(3885)]
# 计算roc-auc
def plot_auc_curves(fprs, tprs, auc, directory, name):
    mean_fpr = np.linspace(0, 1, 20000) #生成【0-1】之间，包含0和1的20000个数，作为线性差值法待插入数据的横坐标
    tpr = []

    for i in range(len(fprs)):
        tpr.append(interp(mean_fpr, fprs[i], tprs[i])) # 利用线性差值法，生成待插入数据的纵坐标，用于绘制五折平均的roc图
        tpr[-1][0] = 0.0 # 将第一个值设置为0
        plt.plot(fprs[i], tprs[i], alpha=0.4, linestyle='--', label='Fold %d AUC: %.4f'%(i + 1, auc[i]))

    mean_tpr = np.mean(tpr, axis=0) # 对五折的线性差值纵坐标 求平均
    mean_tpr[-1] = 1.0 #待插入数据的纵坐标的最后一个值设置为1
    # mean_auc = metrics.auc(mean_fpr, mean_tpr)
    mean_auc = np.mean(auc)
    auc_std = np.std(auc) #计算数组或者列表的标准差
    plt.plot(mean_fpr, mean_tpr, color='BlueViolet', alpha=0.9, label='Mean AUC: %.4f $\pm$ %.4f' % (mean_auc, auc_std))

    plt.plot([0, 1], [0, 1], linestyle='--', color='black', alpha=0.4) # 绘制中间线，alpha代表颜色的透视度，默认为1
    plt.xlim([-0.05, 1.05]) # 设置x,y轴的范围
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right') #显示图例
    plt.savefig(directory+'/%s.jpg' % name, dpi=1200, bbox_inches='tight')
    plt.close()

# 参数说明
# fprs[5(五折)*narray(878测试集的样本数目)]
# fprs[5*narray]

# auc_result[5*1]
# prc_result[5*1]

# presion[5(五折)*narray(3885 个数随机，代码自动根据threshold划分)]
# recalls[5*narray(3885)]
# 计算pr-auc
def plot_prc_curves(precisions, recalls, prc, directory, name):
    mean_recall = np.linspace(0, 1, 20000)
    precision = []

    for i in range(len(recalls)):
        precision.append(interp(1-mean_recall, 1-recalls[i], precisions[i]))
        precision[-1][0] = 1.0
        plt.plot(recalls[i], precisions[i], alpha=0.4, linestyle='--', label='Fold %d AP: %.4f' % (i + 1, prc[i]))

    mean_precision = np.mean(precision, axis=0)
    mean_precision[-1] = 0
    # mean_prc = metrics.auc(mean_recall, mean_precision)
    mean_prc = np.mean(prc)
    prc_std = np.std(prc)
    plt.plot(mean_recall, mean_precision, color='BlueViolet', alpha=0.9,
             label='Mean AP: %.4f $\pm$ %.4f' % (mean_prc, prc_std))  # AP: Average Precision

    plt.plot([1, 0], [0, 1], linestyle='--', color='black', alpha=0.4)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve')
    plt.legend(loc='lower left')
    plt.savefig(directory + '/%s.jpg' % name, dpi=1200, bbox_inches='tight')
    plt.close()