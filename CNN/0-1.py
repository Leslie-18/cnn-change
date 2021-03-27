# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:04:49 2019

@author: snoopy
"""
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
import numpy as np

re_list = {'ant','eclipse','itextpdf','jEdit','liferay','lucene','struts','tomcat'}
#re_list = {'ant_selected','eclipse_selected','itextpdf_selected','jEdit_selected','liferay_selected','lucene_selected','struts_selected','tomcat_selected'}
shape = np.empty((0,2))
for re in re_list:
    for j in range(10):
        for i in range(0,10):
            data_train = np.genfromtxt('F:/beifen_torch_change_1018/torch_Change/8_files_2/{}/{}/train_{}.csv'.format(re, j, i), dtype=np.str, delimiter=',')
            data_test = np.genfromtxt('F:/beifen_torch_change_1018/torch_Change/8_files_2/{}/{}/test_{}.csv'.format(re, j, i), dtype=np.str, delimiter=',')

            data_te = data_train[1:, :-1]
            data_tr = data_test[1:, :-1]

            scaler = StandardScaler()
            #!!!!scaler = scale(data_te)
            my_matrix_normorlize_te=scale(data_te)
            scaler.fit(data_te)
            #print(scaler.mean_)
            #scaler.data_max_
            my_matrix_normorlize_te=scaler.transform(data_te)

            my_matrix_normorlize_te = my_matrix_normorlize_te.astype(np.float32)
            data_train[1:, :-1] = my_matrix_normorlize_te

            np.savetxt('F:/beifen_torch_change_1018/torch_Change/8_files_2/{}/{}/train_1_{}.csv'.format(re, j, i), data_train, fmt='%s', delimiter=',')
            print("OK")

            my_matrix_normorlize_tr = scaler.transform(data_tr)
            #my_matrix_normorlize_tr = scale(data_tr)
            my_matrix_normorlize_tr = my_matrix_normorlize_tr.astype(np.float32)
            data_test[1:, :-1] = my_matrix_normorlize_tr

            np.savetxt('F:/beifen_torch_change_1018/torch_Change/8_files_2/{}/{}/test_1_{}.csv'.format(re, j, i), data_test, delimiter=',', fmt='%s')
            print("OK")


# for re in re_list:
#     for j in range(10):
#         for i in range(0,10):
#             data_test = np.genfromtxt('F:/2021newdata/{}/{}/test_{}.csv'.format(re, j, i), dtype=np.str, delimiter=',')
#
#             data_tr = data_test[1:, :-1]
#
#             #scaler = MinMaxScaler()
#             scaler = StandardScaler()
#             scaler.fit(data_tr)
#             #scaler.data_max_
#             my_matrix_normorlize_tr = scaler.transform(data_tr)
#             my_matrix_normorlize_tr = my_matrix_normorlize_tr.astype(np.float32)
#             data_test[1:, :-1] = my_matrix_normorlize_tr
#
#             np.savetxt('F:/2021newdata_10_norm/{}/{}/test_{}.csv'.format(re, j, i), data_test, delimiter=',', fmt='%s')
#             print("OK")