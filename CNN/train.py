# 主程序页面
from sklearn import metrics
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F
import numpy as np
import torch.optim.lr_scheduler
from scipy.io import arff

from model import Net
from data_utils import LineData
import deal_data
import evaluate
from pytorchtools import EarlyStopping


# 十折
fold_num = 10
# 文件数
file_num = 8
# 迭代次数
epoch_sum = 100
# 设置随机数种子
deal_data.setup_seed(20)

#lists = ['ant_selected', 'eclipse_selected', 'itextpdf_selected', 'jEdit_selected', 'liferay_selected', 'lucene_selected', 'struts_selected', 'tomcat_selected']
lists = ['ant', 'eclipse', 'itextpdf', 'jEdit', 'liferay', 'lucene', 'struts', 'tomcat']
headers = ['file', 'accuracy', 'gmean', 'recall0', 'recall1', 'precision0', 'precision1', 'fmeasure0', 'fmeasure1', 'balance0', 'balance1','auc','mcc']
rows = []
for j in range(8):#
    file_row = []
    gmeanes = []
    recall0es = []
    recall1es = []
    auces = []
    precision0es = []
    precision1es = []
    accuracyes = []
    fmeasure0es = []
    fmeasure1es = []
    balance0es = []
    balance1es = []
    mcces = []
    print("file:{}".format(str(lists[j])))
    for k in range(10):
        # 读取文件类
        sum_gmean = 0
        sum_recall0 = 0
        sum_recall1 = 0
        sum_auc = 0
        sum_precision0 = 0
        sum_precision1 = 0
        sum_loss = 0
        sum_accuracy = 0
        sum_fmeasure0 = 0
        sum_fmeasure1 = 0
        sum_balance0 = 0
        sum_balance1 = 0
        sum_mcc = 0
        for i in range(fold_num):
            print("_________________({})_____________________".format(i + 1))

            '''
            # 读取数据行列数
            # train_row = pd.read_csv("C:/Users/linan/PycharmProjects/datanorm/" + str(lists[j]) + "/train_" + str(i) + ".csv")
            #train_row = pd.read_csv("./data_10_norm/" + str(lists[j]) +"/"+ str(k) + "/train_" + str(i) + ".csv")#
            datas = arff.loadarff("./data_stan/" + str(lists[j])  +"/"+ str(k) + "/train_" + str(i) + ".arff")#实验1最终数据集
            train_row = pd.DataFrame(datas[0])
            # tp = train_row.shape[0]
            lens = train_row.shape[1]
            '''

            train_row = pd.read_csv("F:/beifen_torch_change_1018/torch_Change/8_files/" + str(lists[j]) +"/"+ str(k) + "/train_" + str(i) + ".csv")#实验2数据
            # datas = arff.loadarff("./data_stan/" + str(lists[j]) + "/" + str(k) + "/train_" + str(i) + ".arff")实验1
            # train_row = pd.read_csv("./data_stan/" + str(lists[j]) + "/" + str(k) + "/train_" + str(i) + ".csv")#实验1
            # train_row = pd.DataFrame(datas[0])   F:/2021newdata_10_norm/F:/beifen_torch_change_1018/torch_Change/8_files/
            # tp = train_row.shape[0]
            lens = train_row.shape[1]

            # 加载数据，设置batch size
            # trainset= LineData(data_file="C:/Users/17862/Desktop/no_f1_f32/" + str(lists[j]) + "/train_" + str(i) + ".csv")
            #trainset= LineData(data_file="./data_stan/" + str(lists[j])+ "/" + str(k)+"/train_" + str(i) + ".arff") #实验1
            trainset= LineData(data_file="F:/beifen_torch_change_1018/torch_Change/8_files/" + str(lists[j])+ "/" + str(k)+"/train_" + str(i) + ".csv") #实验1

            # trainset = LineData(
                # data_file="./addfeatures_10fold/" + str(lists[j]) + "/" + str(k) + "/train_" + str(i) + ".csv")  #
            #按照batch size封装成Tensor
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True)#1

            # testset = LineData(data_file="C:/Users/linan/PycharmProjects/datanorm/" + str(lists[j]) + "/test_" + str(i) + ".csv", train=False)
            #testset = LineData(data_file="./data_stan/" + str(lists[j]) +"/"+ str(k) +"/test_" + str(i) + ".arff", train=False)#实验1
            testset = LineData(data_file="F:/beifen_torch_change_1018/torch_Change/8_files/" + str(lists[j]) +"/"+ str(k) +"/test_" + str(i) + ".csv", train=False)#实验1
            # testset = LineData(data_file="./addfeatures_10fold/" + str(lists[j]) + "/" + str(k) + "/test_" + str(i) + ".csv",
                               # train=False)  #实验2
            testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False)#200True

            # 调用模型框架
            model = Net(len=lens)
            # 交叉熵损失函数
            criterion = nn.CrossEntropyLoss()
            # 优化器
            optimizer =torch.optim.RMSprop(model.parameters(), lr=1e-4, alpha=0.9)#0.00001 1e-4 500 well
            # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-3)#0.001
            # 早停
            early_stopping = EarlyStopping(patience=15, verbose=True)#30

            train_losses = []
            # avg_train_losses = []

            # 训练
            for epoch in range(epoch_sum):
                model.train()
                loss_sum = 0.0
                for data in trainloader:
                    inputs, labels = data
                    # 将梯度初始化为零
                    optimizer.zero_grad()
                    # 前向传播求出预测的值
                    outputs = model(inputs)
                    # CrossEntropyLoss() 是 softmax 和 负对数损失的结合,不需要再softmax
                    loss = criterion(outputs, labels)
                    # 反向传播求梯度
                    loss.backward()
                    # 更新所有参数
                    optimizer.step()
                    train_losses.append(loss.item())

                train_loss = np.average(train_losses)
                # avg_train_losses.append(train_loss)


                epoch_len = len(str(epoch_sum))
                print_msg = (f'[{epoch:>{epoch_len}}/{epoch_sum:>{epoch_len}}] ' +
                             f'train_loss: {train_loss:.10f} ' ) #8f  +f'recall: {recall:.5f}'+ f'valid_loss: {valid_loss:.5f}'
                print(print_msg)

                # clear lists to track next epoch
                train_losses = []

                early_stopping(train_loss, model)#valid_loss
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            #加载最优模型
            model.load_state_dict(torch.load('checkpoint.pt'))
            print('Finished Training')

            # 测试
            model.eval()

            TP = 0
            TN = 0
            FN = 0
            FP = 0

            with torch.no_grad():
                for data in testloader:
                    inputs, labels = data
                    outputs = model(inputs)

                    # 计算score
                    output = F.softmax(outputs, dim=1)
                    score = []
                    for item in output:
                        score.append(item[1])
                    score = torch.from_numpy(np.array(score))

                    _, predicted = torch.max(outputs.data, 1)

                    print(labels)
                    print(predicted)

                    TP += ((predicted == 1) & (labels == 1)).cpu().sum().item()
                    TN += ((predicted == 0) & (labels == 0)).cpu().sum().item()
                    FN += ((predicted == 0) & (labels == 1)).cpu().sum().item()
                    FP += ((predicted == 1) & (labels == 0)).cpu().sum().item()

            #计算评估指标
            precision_0, precision_1, recall_0, recall_1, fMeasure_0, fMeasure_1, accuracy, gmean, balance0, balance1, mcc = evaluate.result_evaluate(TP, TN, FN, FP)
            # auc = metrics.roc_auc_score(labels, out)
            auc = evaluate.calauc(score, labels)

            print("recall_1:{:.7g}".format(recall_1))
            print("precision_1:{:.7g}".format(precision_1))
            print("F1_1:{:.7g}".format(fMeasure_1))
            print("auc:{:.7g}".format(auc))
            print("mcc:{:.7g}".format(mcc))

            sum_accuracy += float(accuracy)
            sum_gmean += float(gmean)
            sum_recall0 += float(recall_0)
            sum_recall1 += float(recall_1)
            sum_precision0 += float(precision_0)
            sum_precision1 += float(precision_1)
            sum_fmeasure0 += float(fMeasure_0)
            sum_fmeasure1 += float(fMeasure_1)
            sum_balance0 += float(balance0)
            sum_balance1 += float(balance1)
            sum_auc += float(auc)
            sum_mcc += float(mcc)
            pass

        ave_accuracy = sum_accuracy / fold_num
        ave_gmean = sum_gmean / fold_num
        ave_recall0 = sum_recall0 / fold_num
        ave_recall1 = sum_recall1 / fold_num
        ave_precision0 = sum_precision0 / fold_num
        ave_precision1 = sum_precision1 / fold_num
        ave_fmeasure0 = sum_fmeasure0 / fold_num
        ave_fmeasure1 = sum_fmeasure1 / fold_num
        ave_balance0 = sum_balance0 / fold_num
        ave_balance1 = sum_balance1 / fold_num
        ave_auc = sum_auc / fold_num
        ave_mcc = sum_mcc / fold_num

        print("ave_recall_1:{:.7g}".format(ave_recall1))
        print("ave_precision_1:{:.7g}".format(ave_precision1))
        print("ave_F1_1:{:.7g}".format(ave_fmeasure1))
        print("ave_auc:{:.7g}".format(ave_auc))
        print("ave_mcc:{:.7g}".format(ave_mcc))
        print("~~~~~~~~~~~~~this program is:{:s}".format(str(lists[j])))

        file_row.append((lists[j],ave_accuracy,ave_gmean,ave_recall0,ave_recall1,ave_precision0,ave_precision1,ave_fmeasure0,ave_fmeasure1,ave_balance0,ave_balance1,ave_auc,ave_mcc))
        deal_data.outfile('./2021newResult/{}/new_result_2_26_smo.csv'.format(str(lists[j])), headers, file_row,k)
        # deal_data.outfile('./result.csv', headers, rows)
        file_row = []

        gmeanes.append(ave_gmean)
        recall0es.append(ave_recall0)
        recall1es.append(ave_recall1)
        auces.append(ave_auc)
        precision0es.append(ave_precision0)
        precision1es.append(ave_precision1)
        accuracyes.append(ave_accuracy)
        fmeasure0es.append(ave_fmeasure0)
        fmeasure1es.append(ave_fmeasure1)
        balance0es.append(ave_balance0)
        balance1es.append(ave_balance1)
        mcces.append(ave_mcc)

    gmean = np.average(gmeanes)
    recall0 = np.average(recall0es)
    recall1 = np.average(recall1es)
    auc = np.average(auces)
    precision0 = np.average(precision0es)
    precision1 = np.average(precision1es)
    accuracy = np.average(accuracyes)
    fmeasure0 = np.average(fmeasure0es)
    fmeasure1 = np.average(fmeasure1es)
    balance0 = np.average(balance0es)
    balance1 = np.average(balance1es)
    mcc = np.average(mcces)

    print("recall_1:{:.7g}".format(recall1))
    print("precision_1:{:.7g}".format(precision1))
    print("F1_1:{:.7g}".format(fmeasure1))
    print("auc:{:.7g}".format(auc))
    print("mcc:{:.7g}".format(mcc))
    print("~~~~~~~~~~~~~this program is:{:s}".format(str(lists[j])))

    rows.append((lists[j], accuracy, gmean, recall0, recall1, precision0, precision1, fmeasure0, fmeasure1, balance0, balance1, auc, mcc))
    # 将结果存入文件

    #deal_data.outfile('./stan_result_1111UNDERsample.csv',headers,rows)#实验1
    deal_data.outfile('./2021newResult/2021newResult_0226_smo.csv', headers, rows, j)  # 实验2
    # 清空rows
    rows = []

