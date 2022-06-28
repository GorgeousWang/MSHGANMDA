import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    context = torch.device('cuda')
else:
    context = torch.device('cpu')

# GATLayer为了方便求解只含miRNA-疾病的聚合特征
# layers用于对比实验，判断节点层主义力相同时的情况
class GATLayer(nn.Module): #G是g0_train  num_nodes=1345 num_edges=17376
    def __init__(self, G, feature_attn_size, dropout, slope):
        super(GATLayer, self).__init__()

        self.disease_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1) # 返回训练图疾病节点下标的tensor列表
        self.mirna_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 0) # 返回测试图基因节点下标的tensor列表

        self.G = G
        self.slope = slope

        self.m_fc = nn.Linear(G.ndata['m_sim'].shape[1], feature_attn_size, bias=False) # 特征变换
        self.d_fc = nn.Linear(G.ndata['d_sim'].shape[1], feature_attn_size, bias=False) # 特征变换
        self.dropout = nn.Dropout(dropout)
        # self.attn_fc = nn.Linear(feature_attn_size * 2, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self): # 参数重置
        gain = nn.init.calculate_gain('relu') # 返回给定非线性函数的增益值
        nn.init.xavier_normal_(self.m_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.d_fc.weight, gain=gain)
        # nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

# 边的注意力
    def edge_attention(self, edges): #
        # print('SRC size:', edges.src['z'].size())
        # print('DST size: ', edges.dst['z'].size())
        # z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        # a = self.attn_fc(z2)
        # return {'e': a}
        a = torch.sum(edges.src['z'].mul(edges.dst['z']), dim=1).unsqueeze(1) # mul是矩阵对应位置相乘，mm是矩阵乘法
        return {'e': F.leaky_relu(a, negative_slope=self.slope)}

    def message_func(self, edges): #消息函数，将边上源节点和边的注意力系数放到mailbox消息邮箱里面
        return {'z': edges.src['z'], 'e': edges.data['e']} #传播过去字典，z,e都返回在nodes.mailbox里

    def reduce_func(self, nodes):#聚合函数，从消息邮箱里面取数据，进行运算，对目标节点进行更新
        # alpha注意力系数
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)

        return {'h': F.elu(h)}

    def forward(self, G):
        self.G.apply_nodes(lambda nodes: {'z': self.dropout(self.d_fc(nodes.data['d_sim']))}, self.disease_nodes) # 通过提供的功能更新指定节点的特性。即更新disease_nodes节点的特征
        self.G.apply_nodes(lambda nodes: {'z': self.dropout(self.m_fc(nodes.data['m_sim']))}, self.mirna_nodes) # 映射基因节点的特征

        self.G.apply_edges(self.edge_attention) # apply_edges如果不指定边，默认载入所有的边进行更新，这里算了一个注意力系数放进边里
        self.G.update_all(self.message_func, self.reduce_func) # 消息函数、聚合函数、更新函数，不指定则更新所有目标节点，对于无向图，就是更新了所有节点

        return self.G.ndata.pop('h')

# 多头注意力
class MultiHeadGATLayer(nn.Module):
    def __init__(self, G, feature_attn_size, num_heads, dropout, slope, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()

        self.G = G
        self.dropout = dropout
        self.slope = slope
        self.merge = merge

        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(G, feature_attn_size, dropout, slope)) # heads内有num_heads个GAT

    def forward(self, G):
        head_outs = [attn_head(G) for attn_head in self.heads] # 得到八头注意力GAT之后的m和d的结果,是一个list
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1) #将attn_heads个头拼接起来
        else:
            return torch.mean(torch.stack(head_outs), dim=0)

# 元子图多头注意力 HAN_metapath_specific 用于获取元子图中，miRNA和疾病的特征
class Metasubgraph_semantic_gat(nn.Module):
    def __init__(self, G, feature_attn_size, out_dim, dropout, slope):
        super(Metasubgraph_semantic_gat, self).__init__()
        self.mirna_nodes = G.filter_nodes(lambda nodes:nodes.data['type']==0)
        self.disease_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1)

        self.G = G
        self.slope = slope

        self.m_fc = nn.Linear(G.ndata['m_sim'].shape[1], feature_attn_size, bias=False)
        self.d_fc = nn.Linear(G.ndata['d_sim'].shape[1], feature_attn_size, bias=False)

        self.m_fc1 = nn.Linear(feature_attn_size + 495, out_dim)   # 设置全连接层
        self.d_fc1 = nn.Linear(feature_attn_size + 383, out_dim)
        self.attn_fc = nn.Linear(feature_attn_size * 2, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.m_fc.weight, gain=gain)
        # nn.init.xavier_normal_(self.l_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.d_fc.weight, gain=gain)

    def edge_attention(self, edges):
        a = torch.sum(edges.src['z'].mul(edges.dst['z']), dim=1).unsqueeze(1)
        '''z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)'''
        return {'e': F.leaky_relu(a, negative_slope=self.slope)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)

        return {'h': F.elu(h)}

    def forward(self, new_g, meta_path):
# 这里的图为传过来的新构建的子图
        new_g = new_g.to(context)
        if meta_path == 'c':
            new_g.apply_nodes(lambda nodes: {'z': self.dropout(self.d_fc(nodes.data['d_sim']))}, self.disease_nodes)
            new_g.apply_nodes(lambda nodes: {'z': self.dropout(self.m_fc(nodes.data['m_sim']))}, self.mirna_nodes)
            new_g.apply_edges(self.edge_attention)
            new_g.update_all(self.message_func, self.reduce_func)

            h_c = new_g.ndata.pop('h')

            return h_c

        elif meta_path == 'e':
            new_g.apply_nodes(lambda nodes: {'z': self.dropout(self.d_fc(nodes.data['d_sim']))}, self.disease_nodes)
            new_g.apply_nodes(lambda nodes: {'z': self.dropout(self.m_fc(nodes.data['m_sim']))}, self.mirna_nodes)
            new_g.apply_edges(self.edge_attention)
            new_g.update_all(self.message_func, self.reduce_func)

            h_e = new_g.ndata.pop('h')

            return h_e

        elif meta_path == 't':
            new_g.apply_nodes(lambda nodes: {'z': self.dropout(self.d_fc(nodes.data['d_sim']))}, self.disease_nodes)
            new_g.apply_nodes(lambda nodes: {'z': self.dropout(self.m_fc(nodes.data['m_sim']))}, self.mirna_nodes)
            new_g.apply_edges(self.edge_attention)
            new_g.update_all(self.message_func, self.reduce_func)

            h_t = new_g.ndata.pop('h')

            return h_t

        elif meta_path == 'g':
            new_g.apply_nodes(lambda nodes: {'z': self.dropout(self.d_fc(nodes.data['d_sim']))}, self.disease_nodes)
            new_g.apply_nodes(lambda nodes: {'z': self.dropout(self.m_fc(nodes.data['m_sim']))}, self.mirna_nodes)
            new_g.apply_edges(self.edge_attention)
            new_g.update_all(self.message_func, self.reduce_func)

            h_g = new_g.ndata.pop('h')

            return h_g
