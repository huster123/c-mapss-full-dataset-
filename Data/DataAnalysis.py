import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def data_analysis_plot(j):
    # 读取原始训练集数据
    raw1 = np.loadtxt("train_FD00" + str(j) + ".txt")
    df1 = pd.DataFrame(raw1, columns=['unit', 'cycles', 'operational setting 1', 'operational setting 2',
                                      'operational setting 3'] + ['sensor measurement' + str(i) for i in
                                                                  range(1, 22)])
    # print(df1.iloc[:, 5:].describe())
    # 绘制传感器读数变化曲线
    plt.figure(1, figsize=(15, 9))
    plt.title('dataset' + str(j) + 'graph', fontsize=15)
    for i in range(1, 22):
        plt.subplot(5, 5, i)
        plt.title(i, fontsize=15)
        df1.iloc[:, i + 4].plot()
    plt.show()

    plt.figure(1, figsize=(15, 9))
    plt.title('dataset' + str(j) + 'histogram', fontsize=15)
    for i in range(1, 22):
        plt.subplot(5, 5, i)
        plt.title(i, fontsize=15)
        sns.distplot(df1.iloc[:, i + 4])
    plt.show()


if __name__ == '__main__':
    data_analysis_plot(1)
    data_analysis_plot(2)
    data_analysis_plot(3)
    data_analysis_plot(4)
