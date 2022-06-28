import time
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from sklearn.model_selection import KFold  # 用于划分k折
from sklearn import metrics

from utils import build_graph, weight_reset
from model import HGANMDA_multi
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1' #设置使用gpu01：cuda1
### wfy
def Train_multi(directory, epochs, attn_size, attn_heads, out_dim, dropout, slope, lr, wd, random_seed, cuda, model_type):
    #设置python\numpy\torch包的随机种子
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    # 设置模型运行的环境cpu还是gpu
    if torch.cuda.is_available():
        context = torch.device('cuda')
        print("driver", context)
    else:
        context = torch.device('cpu')
        print("driver", context)

    # 2数据准备
    # g为miRNA和disease单类关联图，g0为miRNA、disease异质关联图，所有样本的疾病节点向量list，所有样本的的基因节点向量
    # ndarray： ID【383，383】,IM【495，495】
    g, g0,disease_vertices, mirna_vertices, ID, IM, samples = build_graph(
        directory, random_seed) # 构建图像
    # DGLHeteroGraph:
    #g:Graph(
    #       num_nodes=878,
    #       num_edges=21720,
    #       ndata_schemes={'type': Scheme(shape=(), dtype=torch.int64),
    #                      'd_sim': Scheme(shape=(383,), dtype=torch.float32),
    #                      'm_sim': Scheme(shape=(495,), dtype=torch.float32)}
    #       edata_schemes = {'label': Scheme(shape=(), dtype=torch.float32)}
    #       )
    #g0:Graph(
    #       num_nodes=878, num_edges=25078,
    #       ndata_schemes={'type': Scheme(shape=(), dtype=torch.int64), 'd_sim': Scheme(shape=(383,), dtype=torch.float32), 'm_sim': Scheme(shape=(495,), dtype=torch.float32)}
    #       edata_schemes={'dm': Scheme(shape=(), dtype=torch.float32), 'md': Scheme(shape=(), dtype=torch.float32), 'multi_label': Scheme(shape=(), dtype=torch.float32)}
    #       )
    # list : disease_vertices[10860],mirna_vertices[10860]
    # narray : samples[10860*3]
    samples_df = pd.DataFrame(samples, columns=['miRNA', 'disease', 'label'])
    # 展示数据信息
    # HeteroNodeDataView g0.ndata: {'type': tensor([1, 1,...]), 'd_sim': tensor([[1.0,2.0]],[],[]) , '...'}
    # HeteroEdgeDataView  g0.edata: {'dm': tensor([1., 1., 1.,  ..., 0., 0., 0.]), 'md': tensor([0., 0., 0.,  ..., 0., 0., 0.]), 'multi_label[1*25078]': tensor([0., 0., 0.,  ..., 4., 4., 4.])}
    print('vertices:', g0.number_of_nodes()) # 打印二质图的节点数目 # g.number_of_nodes() == g.number_nodes()
    print('g edges number *2:', g.number_of_edges()) # 打印二质图的边的数目
    print('g0 edges number *2:', g0.number_of_edges()) # 打印二质图的边的数目
    print('disease nodes:', torch.sum(g0.ndata['type'] == 1).numpy()) #打印疾病节点的个数 #
    print('mirna nodes: ', torch.sum(g0.ndata['type'] == 0).numpy()) #打印mirna节点的个数

    #g0 = g0.to(context)
    #g = g.to(context) # 将基因、疾病二质图放入cpu中，用于加速训练和预测,dgl提供了这种功能
    # 用于记录结果数据
    prc_result = []  # pr-auc
    auc_result = [] # roc-auc

    acc_result = [] # acc
    pre_result = [] # precision
    recall_result = [] #recall
    f1_result = [] #f1

    fprs = []
    tprs = []
    precisions = []
    recalls = []

    # 设置五折交叉验证
    i = 0
    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)

    for train_idx, test_idx in kf.split(samples[:, 2]):  # 填入label numpy,返回五次划分结果的索引下标
        i += 1 # 标记折数目
        print('Training for Fold', i)
        # 将训练集的指针标记为1，其余为0
        samples_df['train'] = 0  # 先添加全为零的列
        samples_df['train'].iloc[train_idx] = 1  # 按行，把训练集下标的train列值全部变为1

        train_tensor = torch.from_numpy(samples_df['train'].values.astype('int64'))

        edge_data = {'train': train_tensor}  # 为参与训练的边添加一个字典'train' 标识训练集tensor类型的值1
        # 对两个异质图的训练边进行标记         g.edges g.edata g.etypes / g.nodes g.ndata g.ntypes
        g.edges[disease_vertices, mirna_vertices].data.update(edge_data)
        g.edges[mirna_vertices, disease_vertices].data.update(edge_data)
        g0.edges[disease_vertices, mirna_vertices].data.update(edge_data) # 对g0这个图也标记了
        g0.edges[mirna_vertices, disease_vertices].data.update(edge_data)  # g0.edges[mirna_vertices, disease_vertices].data
        # g0.edata.update(edge_data) #  Expect number of features to match number of edges. Got 10860 and 25078 instead. g0中有25078条边，不能用此方法来更新

        # 由于图的边的添加训练的原因，g0和g边的标记相同，然后分别根据训练边构建子图
        train_eid = g.filter_edges(lambda edges: edges.data['train'])  # 根据给的边，获取训练边的index，https://www.cnblogs.com/kaishirenshi/p/8611358.html
        g_train = g.edge_subgraph(train_eid, preserve_nodes=True) # 根据传入edge的index来构建子图，Preseve_nodes=True,在子图中保留没有边的节点
        g_train0 = g0.edge_subgraph(train_eid, preserve_nodes=True)

        g_train0 = g_train0.to(context)
        g_train = g_train.to(context)
        label_train = g_train.edata['label'].unsqueeze(1) #g.edata['']相当于边数据一个字典，通过'key'查边中含有的数据；等价于 g.edges[index].data['']获取某条边中的数据，利用字典；
        label_train.to(context)
        src_train, dst_train = g_train.all_edges()
        src_train = src_train.to(context)
        dst_train = dst_train.to(context)
        # 获取测试集的边信息
        test_eid = g.filter_edges(lambda edges: edges.data['train'] == 0)
        src_test, dst_test = g.find_edges(test_eid)
        # 获取测试集的标签
        label_test = g.edges[test_eid].data['label'].unsqueeze(1) #g.edata[''] 等价于 g.edges[index].data['']
        # label_test = g.edges[test_eid].data['label']
        # wfy 多分类
        # label_test = g.edges[test_eid].data['label'].long() # 将lalel改为long
        #
        label_test.to(context)
        # src_train, dst_train，label_train ；src_test, dst_test，label_test
        print('## Training edges:', len(train_eid))
        print('## Testing edges:', len(test_eid))
        # 2模型选择和开发
        if model_type == 'HGANMDA':  # in model1,传入子图g_train0
            model = HGANMDA_multi(G=g_train0,
                            meta_paths_list=['md', 'dm', 'c', 'e', 't', 'g'],  # 几条元路径列表
                            feature_attn_size=attn_size,  # HGAT层输出的结果
                            num_heads=attn_heads,  # 几头
                            num_diseases=ID.shape[0],
                            num_mirnas=IM.shape[0],
                            d_sim_dim=ID.shape[1],
                            m_sim_dim=IM.shape[1],
                            out_dim=out_dim,  # 元子图语层注意力层输出维度
                            dropout=dropout,  # 丢弃神经元
                            slope=slope,  #
                            )

        # 3、模型：实例、优化、损失
        model.apply(weight_reset)  # 对模型对象内所有的nn.的参数初始化
        model.to(context)  # 模型放入cpu或者cuda中
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
       # loss = nn.BCELoss()
        loss = nn.BCELoss()
        ### wfy:多分类损失
        # loss = nn.CrossEntropyLoss()
        ###
        # 4、训：轮次、训练、验证
        for epoch in range(epochs):
            start = time.time()
            '''
            https://blog.csdn.net/qq_38410428/article/details/101102075
            启用 Batch Normalization 和 Dropout。
            如果模型中有BN层(Batch Normalization）和 Dropout，需要在训练时添加model.train()。model.train()是保证BN层能够用到每一批数据的均值和方差。对于Dropout，model.train()是随机取一部分网络连接来训练更新参数。

            不启用 Batch Normalization 和 Dropout。
            如果模型中有BN层(Batch Normalization）和Dropout，在测试时添加model.eval()。model.eval()是保证BN层能够用全部训练数据的均值和方差，即测试过程中要保证BN层的均值和方差不变。对于Dropout，model.eval()是利用到了所有网络连接，即不进行随机舍弃神经元。
            训练完train样本后，生成的模型model要用来测试样本。在model(test)之前，需要加上model.eval()，否则的话，有输入数据，即使不训练，它也会改变权值。这是model中含有BN层和Dropout所带来的的性质。
            在做one classification的时候，训练集和测试集的样本分布是不一样的，尤其需要注意这一点。
            '''
            ###wfy:多分类测试label
            # label_train = torch.empty(len(np.array(label_train)), dtype=torch.long).random_(5)
            # label_test = torch.empty(len(np.array(label_test)), dtype=torch.long).random_(5)
            ##
            model.train()
            with torch.autograd.set_detect_anomaly(True):  # 设置autograd引擎异常检测的上下文管理器。
                score_train = model(g_train0, g0, src_train, dst_train)
                loss_train = loss(score_train, label_train)

                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

            model.eval()  # 这一部分验证的目的是便于观察模型当前的结果如何，用的是五折中的一折
            with torch.no_grad():
                score_val = model(g, g0, src_test, dst_test)
                score_val = score_val.to('cpu')
                loss_val = loss(score_val, label_test)

            score_train_cpu = np.squeeze(score_train.cpu().detach().numpy())  # 转到cpu上的训练集合得分
            score_val_cpu = np.squeeze(score_val.cpu().detach().numpy())  # 转到cpu上的验证集合得分
            label_train_cpu = np.squeeze(label_train.cpu().detach().numpy())  # 转到cpu上的训练集合标签
            label_val_cpu = np.squeeze(label_test.cpu().detach().numpy())  # 转到cpu上的验证集合标签
            ##wfy : auc指标，在多分类时需要变化，这里多分类时暂时不用该指标
            train_auc = metrics.roc_auc_score(label_train_cpu, score_train_cpu)
            val_auc = metrics.roc_auc_score(label_val_cpu, score_val_cpu)
            # ##
            ##wfy:多分类时，通过取最大值，获得多分类的结果
            # pred_val = np.argmax(score_val_cpu, axis=1)  # 预测值
            ##
            ## wfy：多分类测试指标
            pred_val = [0 if j < 0.5 else 1 for j in score_val_cpu] #预测值
            acc_val = metrics.accuracy_score(label_val_cpu, pred_val)  # acc值
            pre_val = metrics.precision_score(label_val_cpu, pred_val, average='macro')  # pre值
            recall_val = metrics.recall_score(label_val_cpu, pred_val,average='macro') #recall值
            f1_val = metrics.f1_score(label_val_cpu, pred_val,average='macro')#f1值
            #
            end = time.time()# 训练结束时间的位置可以上调一下
            #
            if (epoch + 1) % 10 == 0:
                print('Epoch:', epoch + 1, 'Train Loss: %.4f' % loss_train.item(),
                      'Val Loss: %.4f' % loss_val.cpu().detach().numpy(),
                      'Acc: %.4f' % acc_val, 'Pre: %.4f' % pre_val, 'Recall: %.4f' % recall_val, 'F1: %.4f' % f1_val,
                      'Train AUC: %.4f' % train_auc, 'Val AUC: %.4f' % val_auc, 'Time: %.2f' % (end - start))

#一轮run（这里是一折）所有epoch结束以后再进行一次eval，这次eval就不是观察模型的训练程度了，而是为了计算指标，并存储，最后用来模型评价
        model.eval()
        with torch.no_grad():
            score_test = model(g, g0, src_test, dst_test)

        score_test_cpu = np.squeeze(score_test.cpu().detach().numpy())
        label_test_cpu = np.squeeze(label_test.cpu().detach().numpy())

        fpr, tpr, thresholds = metrics.roc_curve(label_test_cpu, score_test_cpu)  # 一折中，所有epoch结束后，一组fpr,tpr，维度等于验证集的数量
        precision, recall, _ = metrics.precision_recall_curve(label_test_cpu, score_test_cpu) # 一组precision,recall
        test_auc = metrics.auc(fpr, tpr)
        test_prc = metrics.auc(recall, precision)
        ##wfy:多分类时，通过取最大值，获得多分类的结果
        # pred_test = np.argmax(score_val_cpu, axis=1)  # 预测值
        ##
        ## wfy：多分类测试指标
        pred_test = [0 if j < 0.5 else 1 for j in score_test_cpu]
        acc_test = metrics.accuracy_score(label_val_cpu, pred_test)  # acc值
        pre_test = metrics.precision_score(label_val_cpu, pred_test, average='macro')  # pre值
        recall_test = metrics.recall_score(label_val_cpu, pred_test, average='macro')  # recall值
        f1_test = metrics.f1_score(label_val_cpu, pred_test, average='macro')  # f1值

        print('Fold: ', i, 'Test acc: %.4f' % acc_test, 'Test Pre: %.4f' % pre_test,
              'Test Recall: %.4f' % recall_test, 'Test F1: %.4f' % f1_test, 'Test PRC: %.4f' % test_prc,
              'Test AUC: %.4f' % test_auc)
        # test auc:0.9 test prc:0.9  tpr[808] fpr[808]
        # 1 Fold: acc_test 0.8692449355432781  pre_test 0.8697805052827485
        # recal_test 0.8692801036466247 f1_test 0.8692049006178781
        # test_auc : 0.9404006471130888  test_prc : 0.9348773880521525
        # precision :narray[3624会变动]   recall :narray[3623]  #默认取3623个预值得出
        acc_result.append(acc_test) # 共五个值

        pre_result.append(pre_test) # 五个
        recall_result.append(recall_test) # 五个
        f1_result.append(f1_test) # 五个
        prc_result.append(test_prc) # 五个

        fprs.append(fpr) # 每折后，测试集的fpr，一共五个 fpr list
        tprs.append(tpr) # 每折后，测试集的tpr，一共五个 tpr list
        auc_result.append(test_auc)

        precisions.append(precision)
        recalls.append(recall)

    # 所有轮次结束之后，打印模型效果
    print('## Training Finished !')
    print('-----------------------------------------------------------------------------------------------')
    print('-AUC mean: %.4f, variance: %.4f \n' % (np.mean(auc_result), np.std(auc_result)),
          'Accuracy mean: %.4f, variance: %.4f \n' % (np.mean(acc_result), np.std(acc_result)),
          'Precision mean: %.4f, variance: %.4f \n' % (np.mean(pre_result), np.std(pre_result)),
          'Recall mean: %.4f, variance: %.4f \n' % (np.mean(recall_result), np.std(recall_result)),
          'F1-score mean: %.4f, variance: %.4f \n' % (np.mean(f1_result), np.std(f1_result)),
          'PRC mean: %.4f, variance: %.4f \n' % (np.mean(prc_result), np.std(prc_result)))

    return fprs, tprs, auc_result, precisions, recalls, prc_result
    # fprs[5*narray(878测试集的样本数目)]
    # fprs[5*narray]
    # auc_result[5*1]
    # prc_result[5*1]
    # presion[5*narray(3885代码自动根据threshold划分)]
    # recalls[5*narray(3885)]


