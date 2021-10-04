# About CNN file and ChangePronePrediction file
## Intro
This CNN file is used to build CNN model and conduct model training and prediction. The ChangePronePrediction file is the source code in the paper [1].
## How to use
The data in the Data file is divided into data sets. Please refer to the code in PreMain.java for the division method. At the same time, please refer to the code in AttrSelect.java in ChangePronePrediction file for the code part of feature selection. After that, the data standardization file 0-1.py in CNN file is used to process the data. Run train.py code to train and test the model.

[1] Zhu X ,  He Y ,  Cheng L , et al. Software change﹑roneness prediction through combination of bagging and resampling methods[J]. Journal of Software Evolution & Process, 2018, 30(12).

# About GBDT file
## Intro
This file is used to build GBDT model and conduct model training and prediction.
## How to use
The data processing method is the same as the experimental process of CNN model. Run gbdt.py to train and predict the model.