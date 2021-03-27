from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
import torch
import csv
import random
import numpy as np

# 将原始数据转化为训练需要的数据格式
# np.ndarray 转tensor 并增加一维
def to_tensor(data):
    data = torch.from_numpy(data).type(torch.float32)
    data = data.unsqueeze(0)

    return data

# 将结果数据存入文件
def outfile(file,headers,rows,n):
    if n==0:
        f = open(file, 'a+',newline='')
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(rows)
    else:
        f = open(file, 'a+',newline='')
        f_csv = csv.writer(f)
        #f_csv.writerow(headers)
        f_csv.writerows(rows)

# 设置随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True