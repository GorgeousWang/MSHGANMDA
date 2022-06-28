import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from mxnet import ndarray as nd
from mxnet.gluon import nn as ng

if torch.cuda.is_available():
    context = torch.device('cuda')
else:
    context = torch.device('cpu')


from layers import MultiHeadGATLayer, Metasubgraph_semantic_gat # 从layers层导入


# 语义层注意力
# 主要用于修改数据，作比较模型，这里主要有两个地方：1.hidden_size; 2.beta
# 1是语义层向量q的维度变化对参数的影响；2是beta用于语义层贡献相同时的结果
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128): # in_size=out_dim * num_heads =512 8*64
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):                               # [x, 2, 512] ，X是节点个数
        w = self.project(z).mean(0)  # 权重矩阵，获得元路径的重要性 [2, 1]
        # beta = torch.softmax(w, dim=0)
        beta = torch.sigmoid(w)
        beta = beta.expand((z.shape[0],) + beta.shape)  # [x,2,1]
        # delta = beta * z
        return (beta * z).sum(1)                        # [x, 512]

##wfy
class HGANMDA_multi(nn.Module): # 传入的G是g0_train：三元异质图的训练子图
    def __init__(self, G, meta_paths_list, feature_attn_size, num_heads, num_diseases, num_mirnas,
                     d_sim_dim, m_sim_dim, out_dim, dropout, slope):
        super(HGANMDA_multi, self).__init__()

        self.G = G
        self.meta_paths = meta_paths_list
        self.num_heads = num_heads
        self.num_diseases = num_diseases
        self.num_mirnas = num_mirnas

        self.gat = MultiHeadGATLayer(G, feature_attn_size, num_heads, dropout, slope) #输入：图、特征注意力向量、头数、丢弃率 ，在layers中leckrelu
        self.heads = nn.ModuleList()

        self.metapath_layers = nn.ModuleList() # 四个元路径层，每个元路径需要attn_heads个元路径注意力层
        for i in range(self.num_heads):
            self.metapath_layers.append(Metasubgraph_semantic_gat(G, feature_attn_size, out_dim, dropout, slope))

        self.dropout = nn.Dropout(dropout)
        self.m_fc = nn.Linear(feature_attn_size * num_heads + m_sim_dim, out_dim) #多头miRNA特征降维映射矩阵
        self.d_fc = nn.Linear(feature_attn_size * num_heads + d_sim_dim, out_dim) #多头disease降维映射矩阵
        # self.semantic_attention = SemanticAttention(in_size=out_dim * num_heads) #语意注意力层,修改输出
        self.semantic_attention = SemanticAttention(in_size=feature_attn_size * num_heads) #语意注意力层

        self.h_fc = nn.Linear(out_dim , out_dim) #全链接
        self.predict = nn.Linear(out_dim * 2, 1)
        ###wfy：多分类的预测函数
        # self.predict = nn.Linear(out_dim * 2, 5) #有五类
        ###


    def forward(self, G, G0, diseases, mirnas): # 模型中的数据流动过程 G是g0_train,G0是G0,疾病和基因边的节点列表
        index1 = 0
        # multi_md = 1
        for meta_path in self.meta_paths: #对于四种元路径分别 ['md', 'dm', 'ml', 'dl'][c,e,t,g]
            if meta_path == 'md' or meta_path == 'dm':
                # 元路径为md和dm时，获得的聚合特征0-382是疾病特征。383-877是miRNA特征
                if index1 == 0: #只进行一次GAT，更新m和d的特征
                    h_agg0 = self.gat(G)# 得到1345*512的特征向量 1345*（feature_attn_size * 4+original dimsions） # 进入layers.py
                    index1 = 1
            elif meta_path == 'c':
                c_edges = G0.filter_edges(lambda edges: edges.data['multi_label']==1)#选元子图的边
                g_c = G0.edge_subgraph(c_edges, preserve_nodes=True)
                head_outsc = [attn_head(g_c, meta_path) for attn_head in self.metapath_layers]  # 得到4个list,代表对应个头注意力之后的结果
                h_aggc = torch.cat(head_outsc, dim=1)  # 得到【1345*512】的ml的矩阵矩阵
            elif meta_path == 'e':
                e_edges = G0.filter_edges(lambda edges: edges.data['multi_label']==2)
                g_e = G0.edge_subgraph(e_edges, preserve_nodes=True)
                head_outse = [attn_head(g_e, meta_path) for attn_head in self.metapath_layers]  # 得到4个list,代表对应个头注意力之后的结果
                h_agge = torch.cat(head_outse, dim=1)  # 得到【1345*512】的ml的矩阵矩阵
            elif meta_path == 't':
                t_edges = G0.filter_edges(lambda edges: edges.data['multi_label']==3)
                g_t = G0.edge_subgraph(t_edges, preserve_nodes=True)
                head_outst = [attn_head(g_t, meta_path) for attn_head in self.metapath_layers]  # 得到4个list,代表对应个头注意力之后的结果
                h_aggt = torch.cat(head_outst, dim=1)  # 得到【1345*512】的ml的矩阵矩阵
            elif meta_path == 'g':
                g_edges = G0.filter_edges(lambda edges: edges.data['multi_label']==4)
                g_g = G0.edge_subgraph(g_edges, preserve_nodes=True)
                head_outsg = [attn_head(g_g, meta_path) for attn_head in self.metapath_layers]  # 得到个4list,代表对应个头注意力之后的结果
                h_aggg = torch.cat(head_outsg, dim=1)  # 得到【1345*512】的ml的矩阵矩阵
# 不同元路径疾病特征和不同元路径mirna节点特征
        disease0 = h_agg0[:self.num_diseases] # 从dm这个meta-path中更新d的节点特征
        mirna0 = h_agg0[self.num_diseases:self.num_diseases + self.num_mirnas] #从md这个meta-path中更新d的节点特征

        # disease1 = h_agg2[:self.num_diseases] # 从dl中更新d
        # mirna1 = h_agg1[self.num_diseases:self.num_diseases + self.num_mirnas]#从ml中更新m
        #wfy
        disease_c = h_aggc[:self.num_diseases] # 从c_md中更新d
        mirna_c = h_aggc[self.num_diseases:self.num_diseases+self.num_mirnas]#从md中更新m

        disease_e = h_agge[:self.num_diseases] # 从c_md中更新d
        mirna_e = h_agge[self.num_diseases:self.num_diseases+self.num_mirnas]#从md中更新m

        disease_t = h_aggt[:self.num_diseases] # 从c_md中更新d
        mirna_t = h_aggt[self.num_diseases:self.num_diseases+self.num_mirnas]#从md中更新m

        disease_g = h_aggg[:self.num_diseases] # 从c_md中更新d
        mirna_g = h_aggg[self.num_diseases:self.num_diseases+self.num_mirnas]#从md中更新m

        # h_d:(383,895)  h_m:(495,1007)
        # semantic_embeddings1 = []
        # semantic_disease = torch.cat((disease, disease), dim=1)
# 特征在dim=1堆叠
        # 修改后，添加了disease_c和mirnac_c ，得到的语义嵌入变为383*3*512 和495*3*512
        semantic_embeddings1 = torch.stack((disease0,disease_c,disease_e,disease_t,disease_g), dim=1) #语意嵌入  tensor(383,2,512)
        h1 = self.semantic_attention(semantic_embeddings1) # (495*512)
        semantic_embeddings2 = torch.stack((mirna0,mirna_c,mirna_e,mirna_t,mirna_g), dim=1)
        h2 = self.semantic_attention(semantic_embeddings2) # （495*512)

# 将经过语义层注意力得到的疾病特征和miRNA特征，和原来的疾病特征和miRNA特征连接
        h_d = torch.cat((h1, self.G.ndata['d_sim'][:self.num_diseases]), dim=1)
        h_m = torch.cat((h2, self.G.ndata['m_sim'][self.num_diseases:878]), dim=1)

        h_m = self.dropout(F.elu(self.m_fc(h_m)))       # （495，1007）->（495,64）
        h_d = self.dropout(F.elu(self.d_fc(h_d)))       # (383,895)->（383,64）

        h = torch.cat((h_d, h_m), dim=0)    # （878,64）
        h = self.dropout(F.elu(self.h_fc(h))) # 做一个全链接 （878,64)-(878,64)

# 获取训练边或测试边的点的特征
        h_diseases = h[diseases]    # 样本边disease特征;(17376,64)
        h_mirnas = h[mirnas]        # 样本边mirna特征(17376,64)
# 全连接层得到结果
        h_concat = torch.cat((h_diseases, h_mirnas), 1)         # (17376,128)
        predict_score = torch.sigmoid(self.predict(h_concat))   # (17376,128)->(17376,128*2)->(17376,1)
        ##wfy多分类激活函数，应该为softmax
        # predict_score = torch.softmax(self.predict(h_concat), dim=1)   # (17376,128)->(17376,128*2)->(17376,num_classes)
        ##
        return predict_score
