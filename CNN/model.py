import torch.nn as nn
import tensorflow as tf
import torch.nn.functional as F
import math

class Net(nn.Module):

    def __init__(self,len = 0):
        self.len = len
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 5, 3)    # (168 - 3)/1 + 1 = 166 (* 10)143      1,10,3
        self.pool = nn.MaxPool1d(kernel_size=2,stride=2)      # (166 - 2)/2 + 1= 83 (* 10)141 71     2,2
        #self.pool = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding="SAME")
        self.conv2 = nn.Conv1d(5, 5, 3)   # (83 - 3)/1 + 1 = 81 (* 20) 69        10,20,3
        self.fc1 = nn.Linear(((self.len-5)//2 - 1) * 5, 2)

        # self.conv1 = nn.Conv1d(1,5, 3)  # (168 - 3)/1 + 1 = 166 (* 10)143      1,10,3
        #
        # self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # (166 - 2)/2 + 1= 83 (* 10)141 71     2,2
        # # self.pool = tf.layers.MaxPooling1D(pool_size=2, strides=2, padding="SAME")
        # self.conv2 = nn.Conv1d(5, 5, 3)  # (83 - 3)/1 + 1 = 81 (* 20) 69        10,20,3
        # self.poo2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # #self.dropout1 = nn.Dropout(p=0.1)
        # #self.dropout1 = nn.Dropout(p=0.3)  # dropout训练
        # self.fc1 = nn.Linear((((self.len-5)//2-1)//2) * 5, 2) # self.len - 5
        # self.dropout2 = nn.Dropout(p=0.1)  # dropout训练


        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                #nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')#He
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.1)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)

        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        #x = self.dropout1(x)
        x = F.relu(x)

        #  将前面多维度的tensor展平成一维self.drop = nn.Dropout(p=0.5)
        x = x.view(-1,((self.len-5)//2 - 1) * 5)
        #x = x.view(-1, ((self.len - 5) // 2 - 1) * 5)

        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))

        x = self.fc1(x)
        #x = self.dropout2(x)
        #x = self.fc3(x)
        return x