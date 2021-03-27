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


# 读取数据类
class LineData(data.Dataset):

    def __init__(self, train=True, data_file="", transform=deal_data.to_tensor):
        self.data_file = data_file
        self.train = train
        self.transform = transform

        if not osp.exists(self.data_file):
            raise FileExistsError('Missing Datasets')

        # 如果是训练集 读取标签和特征并采样
        if self.train:
            self.train_datas = []
            self.train_labels = []

            '''
            data = arff.loadarff(self.data_file)
            train_row = pd.DataFrame(data[0])
            # train_row = pd.read_csv(self.data_file)
            # 读取行列数
            tp = train_row.shape[0]
            len = train_row.shape[1]
            train_row = train_row.values.tolist()
            '''
            #data = arff.loadarff(self.data_file)
            #train_row = pd.DataFrame(data[0])
            train_row = pd.read_csv(self.data_file)

            # 读取行列数
            tp = train_row.shape[0]
            len = train_row.shape[1]
            train_row = train_row.values.tolist()

            m=0
            n=0
            if self.transform:
                train = []
                for i in range(tp):
                    # print(train_row[i][len - 1])
                    # print(type(train_row[i][len - 1]))

                    if train_row[i][len - 1] == str(b'true'):#:#b'true'bool(True)
                        m=m+1
                        self.train_labels.append(1)
                        train.append(train_row[i][0:len - 1])

                    #elif train_row[i][len - 1] ==str(b'false') :#:#b'false'bool(False)
                    else:
                        n=n+1
                        self.train_labels.append(0)
                        train.append(train_row[i][0:len - 1])
                #sys.exit()
                #print(train)
                # print(self.train_labels)
                # 对训练集采样
                samples = SMOTE(kind='svm')
                print(m)
                print(n)

                # samples = RandomOverSampler()
                #print(train)
                #print(type(train))
                #sys.exit()
                #train = np.array(train).reshape(1, -1) #newnewnewnewnewnewnew!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


                train, self.train_labels = samples.fit_sample(train, self.train_labels)

                for item in train:
                    self.train_datas.append(transform(np.array(item)))

                # 采样后 需要调整 label格式,否则有错误提
                self.train_labels = self.train_labels.astype(np.longlong)
            else:
                raise ValueError('We need tranform function!')

        # 如果是测试集 只读取label 和 features 不采样
        if not self.train:
            self.test_datas = []
            self.test_labels = []

            '''
            # test_row = pd.read_csv(self.data_file)
            data = arff.loadarff(self.data_file)
            test_row = pd.DataFrame(data[0])
            tp = test_row.shape[0]
            len = test_row.shape[1]
            test_row = test_row.values.tolist()
            '''
            test_row = pd.read_csv(self.data_file)
            #data = arff.loadarff(self.data_file)
            #test_row = pd.DataFrame(data[0])
            tp = test_row.shape[0]
            len = test_row.shape[1]
            test_row = test_row.values.tolist()

            if self.transform:

                for i in range(tp):
                    if test_row[i][len - 1] == str(b'true'):#:#str(b'TRUE'):bool(True)
                        self.test_labels.append(1)
                        self.test_datas.append(transform(np.array(test_row[i][0:len - 1])))
                    #elif test_row[i][len - 1] == str(b'false'):#str(b'FALSE'):
                    else:
                        self.test_labels.append(0)
                        self.test_datas.append(transform(np.array(test_row[i][0:len - 1])))
            else:
                raise ValueError('We need tranform function!')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            data, target = self.train_datas[index], self.train_labels[index]
        else:
            data, target = self.test_datas[index], self.test_labels[index]

        return data, target

    def __len__(self):
        if self.train:
            return len(self.train_datas)
        else:
            return len(self.test_datas)
