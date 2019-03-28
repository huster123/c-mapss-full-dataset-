#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from matplotlib import pyplot as plt

GloUse = {}
GloUse['train_file'] = ['train_FD00' + str(i) for i in range(1, 5)]
GloUse['test_file'] = ['test_FD00' + str(i) for i in range(1, 5)]
GloUse['SL'] = [15, 15, 16, 15]  # [15, 21, 16, 21]
GloUse['train_units'] = [100, 260, 100, 249]
GloUse['test_units'] = [100, 259, 100, 248]


def mean_squared_error(x, y):
    sum = 0
    n = len(x)
    for i, j in zip(x, y):
        sum = sum + (i - j) ** 2
    return sum / n


def score(x, y):
    sum = 0
    for i, j in zip(x, y):
        z = i - j
        if z < 0:
            sum = sum + np.e ** (-z / 13) - 1
        else:
            sum = sum + np.e ** (z / 10) - 1
    return sum


if __name__ == '__main__':
    n = 4
    # 构建训练数据
    df = pd.read_pickle('Data/' + GloUse['train_file'][n - 1] + '.pickle')
    train_input = df.iloc[:, :-2].values.reshape(-1, 30, GloUse['SL'][n - 1], 1)
    train_output = df.iloc[:, -1].values.reshape(-1, )
    # 构建测试数据
    df = pd.read_pickle('Data/' + GloUse['test_file'][n - 1] + '.pickle')
    df1 = []
    df2 = []
    for i in range(GloUse['test_units'][n - 1]):
        if (i + 1) in (df.unit.values):
            df1.append(df[df.unit == i + 1].iloc[-1, :-2].values)
            df2.append(df[df.unit == i + 1].iloc[-1, -1])
    test_input = np.array(df1).reshape(-1, 30, GloUse['SL'][n - 1], 1)
    test_output = np.array(df2).reshape(-1, )
    # 构建模型
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(10, 1), strides=(1, 1), padding='same', activation='tanh',
                     input_shape=(30, GloUse['SL'][n - 1], 1)))
    # model.add(Dropout(0.5))
    # print('卷积1', model.output_shape)
    model.add(Conv2D(filters=10, kernel_size=(10, 1), strides=(1, 1), padding='same', activation='tanh'))
    # model.add(Dropout(0.5))
    # print('卷积2', model.output_shape)
    model.add(Conv2D(filters=10, kernel_size=(10, 1), strides=(1, 1), padding='same', activation='tanh'))
    model.add(Dropout(0.5))
    # print('卷积3', model.output_shape)
    model.add(Conv2D(filters=10, kernel_size=(10, 1), strides=(1, 1), padding='same', activation='tanh'))
    model.add(Dropout(0.5))
    # print('卷积4', model.output_shape)
    model.add(Conv2D(filters=1, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='tanh',
                     name='conv5'))  # 细节——第5层卷积层的卷积核为3*1，不是10*1
    model.add(Dropout(0.5))
    # print('卷积5', model.output_shape)
    model.add(Flatten())
    # print('平滑层', model.output_shape)
    model.add(Dropout(0.5))
    # print('dropout', model.output_shape)
    model.add(Dense(100, activation='tanh'))
    # print('连接层', model.output_shape)
    model.add(Dense(1, name='out'))
    # print('输出层', model.output_shape)

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mse', optimizer=adam)
    history1 = model.fit(train_input, train_output, batch_size=512, epochs=200, shuffle=True)

    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mse', optimizer=adam)
    history2 = model.fit(train_input, train_output, validation_split=0.33, batch_size=512, epochs=50, shuffle=True)
    # 保存迭代损失值
    np.savetxt(GloUse['train_file'][n - 1] + "iteration.txt", history1.history['loss'] + history2.history['loss'])
    # 绘制收敛曲线图
    plt.plot((history1.history['loss']) + (history2.history['loss']), label='DCNN train 0~250')
    plt.legend(loc='upper right')
    plt.show()
    # 保存模型
    model.save(GloUse['train_file'][n - 1] + '1dDCNNmodel.h5')
    # 保存预测值，RMSE，得分
    test_predict = model.predict(test_input)
    np.savetxt(GloUse['train_file'][n - 1] + "prediction result.txt", test_predict)
    RMSE = math.sqrt(mean_squared_error(test_output, test_predict))
    SCORE = score(test_output, test_predict)
    print("test rmse:", RMSE)
    print("test score:", SCORE)

    np.savetxt(GloUse['train_file'][n - 1] + "RMSE.txt", [RMSE])
    np.savetxt(GloUse['train_file'][n - 1] + "SCORE.txt", [SCORE])
