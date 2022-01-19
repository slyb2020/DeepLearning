#!/usr/bin/env python
# encoding: utf-8
########################################################################################################################
# 本例程适用于 Python3
# 本例程用于产生batch数据
# 输入data中每个样本可以有多个特征，和一个标签，最好都是numpy.array格式。
# datas = [data1, data2, …, dataN ], labels = [label1, label2, …, labelN]，
# 其中data[i] = [feature1, feature2,…featureM], 表示每个样本数据有M个特征。
# 输入我们方法的数据，all_data = [datas, labels] 。
#
# 代码通过索引值来产生batch大小的数据，同时提供是否打乱顺序的选择，根据随机产生数据量范围类的索引值来打乱顺序。
########################################################################################################################

import numpy as np


def BatchGenerator(all_data, batch_size, shuffle=True):
    """
    :param all_data : all_data整个数据集，包含输入和输出标签
    :param batch_size: batch_size表示每个batch的大小
    :param shuffle: 是否打乱顺序
    :return:
    """
    # 输入all_datas的每一项必须是numpy数组，保证后面能按p所示取值
    all_data = [np.array(d) for d in all_data]
    # 获取样本大小
    data_size = all_data[0].shape[0]
    if shuffle:
        # 随机生成打乱的索引
        p = np.random.permutation(data_size)
        # 重新组织数据
        all_data = [d[p] for d in all_data]
    batch_count = 0
    while True:
        # 数据一轮循环(epoch)完成，打乱一次顺序
        if batch_count * batch_size + batch_size > data_size:
            batch_count = 0
            if shuffle:
                p = np.random.permutation(data_size)
                all_data = [d[p] for d in all_data]
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start: end] for d in all_data]


if __name__ == '__main__':
    # 输入x表示有23个样本，每个样本有两个特征
    # 输出y表示有23个标签，每个标签取值为0或1
    x = np.random.random(size=[23, 2])
    y = np.random.randint(2, size=[23, 1])
    count = x.shape[0]

    batch_size = 5
    epochs = 20
    batch_num = count // batch_size

    batch_gen = BatchGenerator([x, y], batch_size)

    for i in range(epochs):
        print("##### epoch %s ##### " % i)
        for j in range(batch_num):
            batch_x, batch_y = next(batch_gen)
            print("-----epoch=%s, batch=%s-----" % (i, j))
            print(batch_x, batch_y)
