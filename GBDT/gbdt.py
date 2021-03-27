import CART_regression_tree
import pandas as pd
import tensorflow as tf
import torch
import numpy as np
import deal_data
import read_data
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import evaluate
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import accuracy_score
#from sklearn import tree
import sys
import numpy as np
# 设置随机数种子
deal_data.setup_seed(20)
# 十折
fold_num = 10

def load_data(data_file):
    """导入训练数据
    :param data_file: {string} 保存训练数据的文件
    :return: {list} 训练数据
    """
    X, Y = [], []
    f = open(data_file)
    for line in f.readlines():
        sample = []
        lines = line.strip().split('\t')
        Y.append(lines[-1])
        for i in range(len(lines) - 1):
            sample.append(float(lines[i]))
        X.append(sample)
    return X, Y


# 二分类输出非线性映射
def sigmoid(x):
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))


class GBDT_RT(object):
    """
    GBDT回归算法类
    """
    def __init__(self):
        self.trees = None
        self.learn_rate = None
        self.init_val = None

    def get_init_value(self, y):
        """计算初始值的平均值
        :param y: {ndarray} 样本标签列表
        :return: average:{float} 样本标签的平均值
        """
        """
        初始化：
        F0(x)=log(p1/(1 - p1))
        """
        p = np.count_nonzero(y)
        n = np.shape(y)[0]
        return np.log(p / (n-p))

    def get_residuals(self, y, y_hat):
        """
        计算样本标签域预测列表的残差
        :param y: {ndarray} 样本标签列表
        :param y_hat: {ndarray} 预测标签列表
        :return: y_residuals {list} 样本标签与预测标签列表的残差
        """

        y_residuals = []
        for i in range(len(y)):
            y_residuals.append(y[i] - y_hat[i])
        return y_residuals

    def fit(self, X, Y, n_estimates, learn_rate, min_sample, min_err, max_height):
        """
        训练GDBT模型
        :param X: {list} 样本特征
        :param Y: {list} 样本标签
        :param n_estimates: {int} GBDT中CART树的个数
        :param learn_rate: {float} 学习率
        :param min_sample: {int} 学习CART时叶节点最小样本数
        :param min_err: {float} 学习CART时最小方差
        """

        # 初始化预测标签和残差
        self.init_val = self.get_init_value(Y)

        n = np.shape(Y)[0]
        F = np.array([self.init_val] * n)
        y_hat = np.array([sigmoid(self.init_val)] * n)
        y_residuals = Y - y_hat
        y_residuals = np.c_[Y, y_residuals]

        self.trees = []
        self.learn_rate = learn_rate
        # 迭代训练GBDT
        for j in range(n_estimates):
            tree = CART_regression_tree.CART_regression(X, y_residuals, min_sample, min_err, max_height).fit()
            for k in range(n):
                res_hat = CART_regression_tree.predict(X[k], tree)
                # 计算此时的预测值等于原预测值加残差预测值
                F[k] += self.learn_rate * res_hat
                y_hat[k] = sigmoid(F[k])
            y_residuals = Y - y_hat
            y_residuals = np.c_[Y, y_residuals]
            self.trees.append(tree)

    def GBDT_predicts(self, X_test):
        """
        预测多个样本
        :param X_test: {list} 测试集
        :return: predicts {list} 预测的结果
        """
        predicts = []
        for i in range(np.shape(X_test)[0]):
            pre_y = self.init_val
            for tree in self.trees:
                pre_y += self.learn_rate * CART_regression_tree.predict(X_test[i], tree)
            print(sigmoid(pre_y))
            if sigmoid(pre_y) >= 0.5:
                predicts.append(1)
            else:
                predicts.append(0)
        return predicts

    def cal_error(self, Y_test, predicts):
        """
        计算预测误差
        :param Y_test: {测试样本标签列表}
        :param predicts: {list} 测试样本预测列表
        :return: error {float} 均方误差
        """

        y_test = np.array(Y_test)
        y_predicts = np.array(predicts)
        error = np.square(y_test - y_predicts).sum() / len(Y_test)
        return error


if __name__ == '__main__':
    # name of features
    #featName = ['Number', 'Plasma', 'Diastolic', 'Triceps', '2-Hour', 'Body', 'Diabetes', 'Age', 'Class']
    # path_train = "F:\\2021newdata\\xg_train.csv"
    # path_test = "F:\\2021newdata\\xg_test.csv"
    # lists = ['ant', 'eclipse', 'itextpdf', 'jEdit', 'liferay', 'lucene', 'struts',
    #          'tomcat']
    #lists = ['ant_selected', 'eclipse_selected', 'itextpdf_selected', 'jEdit_selected', 'liferay_selected', 'lucene_selected', 'struts_selected', 'tomcat_selected']
    lists = ['ant', 'eclipse', 'itextpdf', 'jEdit', 'liferay',
             'lucene', 'struts', 'tomcat']
    #lists = [ 'jEdit_selected']
    headers = ['file', 'accuracy', 'gmean', 'recall0', 'recall1', 'precision0', 'precision1', 'fmeasure0', 'fmeasure1',
               'balance0', 'balance1', 'auc', 'mcc']
    rows = []
    for j in range(8):  #0,8
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
        for k in range(10):#10
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
            for i in range(fold_num):#fold_num
                print("_________________({})_____________________".format(i + 1))

                #X_test, y_test = read_data.rea_data( data_file="F:\\备份torch_change_1018\\torch_Change\\data_10_norm\\ant\\0\\test_2.csv")
                X_test, y_test = read_data.rea_data(data_file="F:/beifen_torch_change_1018/torch_Change/8_files_2/"+ str(lists[j]) + "/" + str(k) + "/test_" + str(i) + ".csv")
                #X_test, y_test = read_data.rea_data(data_file="F:/2021newdata_10_norm/" + str(lists[j]) + "/" + str(k) + "/test_" + str(i) + ".csv")
                    #X_train, y_train= read_data.rea_data(istrain=True, data_file="F:\\备份torch_change_1018\\torch_Change\\data_10_norm\\ant\\0\\train_2.csv")
                X_train, y_train = read_data.rea_data( istrain = True,data_file="F:/beifen_torch_change_1018/torch_Change/8_files_2/"+ str(lists[j]) +"/"+ str(k) +"/train_" + str(i) + ".csv")
                #X_train, y_train = read_data.rea_data(istrain=True, data_file="F:/2021newdata_10_norm/" + str(lists[j]) + "/" + str(k) + "/train_" + str(i) + ".csv")
                #
                X_train = np.array(np.array(X_train))
                y_train = np.array(np.array(y_train))

                X_test = np.array(np.array(X_test))
                y_test = np.array(np.array(y_test))

            #导入数据
                # path_train = "F:\\备份torch_change_1018\\torch_Change\\data_10_norm\\ant\\0\\train_0.csv"
                # path_test = "F:\\备份torch_change_1018\\torch_Change\\data_10_norm\\ant\\0\\test_0.csv"
                # # read data file
                # # data = pd.read_csv(path, sep=',', header=0, names=featName)
                # # data_train = pd.read_csv(path_train, sep=',', header=0, names=featName)
                # # data_test = pd.read_csv(path_test, sep=',', header=0, names=featName)
                #
                # data_train = pd.read_csv(path_train)
                # data_test = pd.read_csv(path_test)
                #
                #
                # tp_train = data_train.shape[0]  # 行
                # lens_train = data_train.shape[1]  # 列
                #
                #
                # tp_test = data_test.shape[0]  # 行
                # lens_test = data_test.shape[1]  # 列
                #
                # X_train = data_train.iloc[:, :-1].values  # train labels
                # X_test = data_test.iloc[:, :-1].values  # test labels
                #
                # print(X_train)
                # print(type(X_train))
                # print(type(X_test))
                #
                # y_train = data_train.iloc[:, -1].values  # 先取出想要的行数据 左闭右开
                # y_test = data_test.iloc[:, -1].values
                # print(y_train)
                # print(type(y_train))
            # 导入数据

                # set random seed
                np.random.seed(123)
                # split data into train and test
                #X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1].values, data.iloc[:, -1].values,
                                                                   # test_size=0.2, random_state=123)

                # normalize train
                # X_train = MinMaxScaler().fit_transform(X_train)
                # normalize test
                # X_test = MinMaxScaler().fit_transform(X_test)
                TP = 0
                TN = 0
                FN = 0
                FP = 0

                #gbdtTrees = GradientBoostingClassifier(n_estimators=100, learning_rate=0.4,max_depth=3,subsample=0.7,max_features=0.2) #n_estimators=100, learning_rate=0.1,subsample=0.5,max_depth=20, random_state=1
                gbdtTrees = GradientBoostingClassifier()
                gbdtTrees.fit(X_train, y_train)

                #a1 = accuracy_score(y_train, gbdtTrees.predict(X_train))
                #a2 = accuracy_score(y_test, gbdtTrees.predict(X_test))

                y_hat = gbdtTrees.predict(X_test)

                # gbdtTrees = GBDT_RT()
                # gbdtTrees.fit(X_train, y_train, n_estimates=10, learn_rate=0.4,min_sample=3, min_err=0.1, max_height=4)
                # for i in range(2):#2
                #     print(CART_regression_tree.numLeaf(gbdtTrees.trees[i]))
                #     print(CART_regression_tree.heightTree(gbdtTrees.trees[i]))
                #     CART_regression_tree.showTree(gbdtTrees.trees[i])
                #     print('--------------------------------------------')
                # y_hat = gbdtTrees.GBDT_predicts(X_test)


                y_hat = np.array(y_hat)

                y_test = torch.from_numpy(y_test)
                y_hat = torch.from_numpy(y_hat)

                # print(type(y_test))
                # print(type(y_hat))
                #
                print(y_test)
                print(y_hat)

                TP += ((y_hat == 1) & (y_test == 1)).cpu().sum().item()
                TN += ((y_hat == 0) & (y_test == 0)).cpu().sum().item()
                FN += ((y_hat == 0) & (y_test == 1)).cpu().sum().item()
                FP += ((y_hat == 1) & (y_test == 0)).cpu().sum().item()

                # y_hat = tf.convert_to_tensor(y_hat)
                # y_test = tf.convert_to_tensor(y_test)
                precision_0, precision_1, recall_0, recall_1, fMeasure_0, fMeasure_1, accuracy, gmean, balance0, balance1, mcc = evaluate.result_evaluate(
                    TP, TN, FN, FP)
                # auc = metrics.roc_auc_score(labels, out)
                auc = evaluate.calauc(y_hat, y_test)

                #acc = accuracy_score(y_test, y_hat)
                # print(accuracy)
                # print(recall_1)
                # print(precision_1)
                # print(fMeasure_1)
                # print(mcc)

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

            file_row.append((
                            lists[j], ave_accuracy, ave_gmean, ave_recall0, ave_recall1, ave_precision0, ave_precision1,
                            ave_fmeasure0, ave_fmeasure1, ave_balance0, ave_balance1, ave_auc, ave_mcc))
            deal_data.outfile('./GBDTResult/{}/gbdtresult2_225_SMO.csv'.format(str(lists[j])), headers, file_row,k)
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

        rows.append((
                    lists[j], accuracy, gmean, recall0, recall1, precision0, precision1, fmeasure0, fmeasure1, balance0,
                    balance1, auc, mcc))
        # 将结果存入文件

        # deal_data.outfile('./stan_result_1111UNDERsample.csv',headers,rows)#实验1
        deal_data.outfile('./GBDTResult/GBDTResult2_225_SMO.csv', headers, rows, j)  # 实验2
        # 清空rows
        rows = []
