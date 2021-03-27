import math

# 计算AUC(rank)
def calauc(pre,labels):
    f = list(zip(pre, labels))
    rank = [values2 for values1, values2 in sorted(f, key=lambda x: x[0])]
    rank_pre = [values1 for values1, values2 in sorted(f, key=lambda x: x[0])]
    posNum = 0
    negNum = 0
    sum = 0
    cnt = 0
    ranklist = []
    rank_list = []
    for i in range(len(labels)):
        ranklist.append(i+1)
    for i in range(len(labels)):
        rank_list.append(i+1)
    rank_list[len(labels)-1] = len(labels)

    for i in range(len(labels)):
        if (labels[i] == 1):
            posNum += 1
        else:
            negNum += 1

    i = len(labels) - 2
    while i >= 0:
        if rank_pre[i] == rank_pre[i+1]:
            cnt = cnt+1
            if i == 0:
                j = i
                for m in range(i, i + cnt + 1):
                    sum += ranklist[m]
                while j <= i + cnt:
                    rank_list[j] = sum / (cnt + 1)
                    j += 1
        else:
            if cnt > 0:
                j = i+1
                for m in range(i+1, i+1+cnt+1):
                    sum += ranklist[m]
                while j <= i+cnt+1:
                    rank_list[j] = sum/(cnt+1)
                    j += 1
                cnt = 0
                sum = 0
        i = i-1

    Sum = 0
    i = len(labels) - 1
    while i >= 0:
        if (rank[i] == 1):
            Sum += rank_list[i]
        i -= 1
    auc = (Sum - (posNum * (posNum + 1)) / 2) / (posNum * negNum)

    return auc

# 计算其他指标
def result_evaluate(TP, TN, FN, FP):

    # precision
    if (TN + FN) > 0:
        precision_0 = TN / (TN + FN)
    else:
        precision_0 = 0
    if (TP + FP) > 0:
        precision_1 = TP / (TP + FP)
    else:
        precision_1 = 0

    # recall
    if (FP + TN) > 0:
        recall_0 = TN / (FP + TN)
    else:
        recall_0 = 0
    if (TP + FN) > 0:
        recall_1 = TP / (TP + FN)
    else:
        recall_1 = 0

    # fmeasure
    if (recall_0 + precision_0) > 0:
        fMeasure_0 = 2 * recall_0 * precision_0 / (recall_0 + precision_0)
    else:
        fMeasure_0 = 0
    if (recall_1 + precision_1) > 0:
        fMeasure_1 = 2 * recall_1 * precision_1 / (recall_1 + precision_1)
    else:
        fMeasure_1 = 0

    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # false_alarm
    if (FP + TN) > 0:
        false_alarm1 = FP / (FP + TN)
    else:
        false_alarm1 = 0
    if (TP + FN) > 0:
        false_alarm0 = FN / (TP + FN)
    else:
        false_alarm0 = 0

    gmean = math.sqrt(recall_0 * recall_1)
    balance1 = 1 - math.sqrt((math.pow((1 - recall_1), 2) + math.pow((0 - false_alarm1), 2)) / 2)
    balance0 = 1 - math.sqrt((math.pow((1 - recall_0), 2) + math.pow((0 - false_alarm0), 2)) / 2)

    if math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) > 0:
        mcc = ((TP * TN) - (FP * FN)) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    else:
        mcc = 0


    return precision_0, precision_1, recall_0, recall_1, fMeasure_0, fMeasure_1, accuracy, gmean, balance0, balance1, mcc