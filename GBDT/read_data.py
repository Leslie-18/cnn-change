import torch.utils.data as data
import os.path as osp
import numpy as np
import pandas as pd
import sys
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import deal_data
from scipy.io import arff

def rea_data(istrain=False,data_file="",transform=deal_data.to_tensor):
    if istrain:
        train_datas = []
        train_labels = []
        data_train = pd.read_csv(data_file)

        tp_train = data_train.shape[0]  # 行
        lens_train = data_train.shape[1]  # 列


        # data_train = data_train.to_numpy()
        # data_test = data_test.to_numpy()

        X_train = data_train.iloc[:, :-1].values  # train labels
        y_train = data_train.iloc[:, -1].values  # 先取出想要的行数据 左闭右开

        if transform:
            for i in range(tp_train):

                if y_train[i] == str(b'TRUE'):#bool(True):
                    train_labels.append(1)
                else:
                    train_labels.append(0)


            #samples = RandomOverSampler()
            samples = SMOTE()

            X_train, train_labels = samples.fit_sample(X_train, train_labels)

            for item in X_train:
                train_datas.append(np.array(item))

            # 采样后 需要调整 label格式,否则有错误提
            train_labels = train_labels.astype(np.longlong)
        else:
            raise ValueError('We need tranform function!')

        return train_datas, train_labels

    if not istrain:
        test_datas = []
        test_labels = []
        data_test = pd.read_csv(data_file)

        tp_test = data_test.shape[0]  # 行
        lens_test = data_test.shape[1]  # 列

        # data_train = data_train.to_numpy()
        # data_test = data_test.to_numpy()

        X_test = data_test.iloc[:, :-1].values  # train labels
        y_test = data_test.iloc[:, -1].values  # 先取出想要的行数据 左闭右开

        if transform:
            for i in range(tp_test):
                if y_test[i] == str(b'TRUE'):#bool(True):
                    test_labels.append(1)
                else:
                    test_labels.append(0)

            # print(test_labels)
            # print(type(test_labels))

        else:
            raise ValueError('We need tranform function!')

        return X_test, test_labels



