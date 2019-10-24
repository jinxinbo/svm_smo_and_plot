from math import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def selectJrand(i, m):
    j = i  # 排除i
    while (j == i):
        j = int(np.random.uniform(0, m))
    return j

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def smo(X, y, C, maxIter):

    alpha = np.zeros(shape=(X.shape[0]))
    m = len(X)
    b = -1

    iter = 0
    while (iter < maxIter):

        print('第%d次迭代'%iter)
        alphaPairsChanged = 0  # alpha是否已经进行了优化
        for i in range(m):
            gxi = 1.0 * np.multiply(alpha, y).T.dot((X.dot(X[i, :].T))) + b
            Ei= gxi - y[i]

            # 选择违反KKT的alpha_1
            if ((alpha[i] > 0) & (alpha[i] < C) &  (y[i] * gxi != 1)) or \
                    ((alpha[i] == 0) &  (y[i] * gxi < 1)) or ((alpha[i] == C) & (y[i] * gxi >= 1)):

                # 随机选个alpha_2
                j = selectJrand(i, m)
                gxj = 1.0 * np.multiply(alpha, y).T.dot((X.dot(X[j, :].T))) + b
                Ej = gxj - y[j]

                alphaIold = alpha[i].copy()
                alphaJold = alpha[j].copy()

                # 上下界
                if (y[i] != y[j]):
                    L = max(0, alpha[j] - alpha[i])
                    H = min(C, C + alpha[j] - alpha[i])
                else:
                    L = max(0, alpha[j] + alpha[i] - C)
                    H = min(C, alpha[j] + alpha[i])
                if L == H:  # print "L==H";
                    continue

                # eta = K11 + K12 - 2*K12
                eta = X[i, :].T.dot(X[i, :]) + X[j, :].T.dot(X[j, :]) - 2 * X[i, :].T.dot(X[j, :])
                if eta < 0:
                    continue
                alpha[j] = alphaJold + y[j] * (Ei - Ej) / eta

                alpha[j] = clipAlpha(alpha[j], H, L)
                # 如果内层循环通过以上方法选择的α_2不能使目标函数有足够的下降，那么放弃α_1
                if (abs(alpha[j] - alphaJold) < 0.00001):
                    continue

                alpha[i] = alphaIold + y[i]*y[j] * (alphaJold - alpha[j])
                b1 = b - Ei - y[i] * (alpha[i] - alphaIold) * (X[i, :].T.dot(X[i, :])) - y[j] * (alpha[j] - alphaJold) * (X[i, :].T.dot(X[j, :]))
                b2 = b - Ej - y[i] * (alpha[i] - alphaIold) * (X[i, :].T.dot(X[j, :])) - y[j] * (alpha[j] - alphaJold) * (X[j, :].T.dot(X[j, :]))

                if (0 < alpha[i]) & (C > alpha[i]):
                    b = b1
                elif (0 < alpha[j]) & (C > alpha[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                print(alpha, b)

                alphaPairsChanged += 1
        print('alphaPairsChanged:%d'%alphaPairsChanged)
        if (alphaPairsChanged == 0):
            iter += 1
        # else:
        #     iter = 0
    return b, alpha

def svm_predict(X_test, X, y, alpha, b):
    gx = []
    for i in range(len(X_test)):
        gxi = 1.0 * np.multiply(alpha, y).T.dot((X.dot(X_test[i, :].T))) + b
        # print(gxi)
        gx.append(gxi)

    return gx

def plot_svm(X, y, alpha, b, C):

    x_min = X[0,:].min() - 5
    x_max = X[0,:].max() + 5
    y_min = X[1,:].min() - 5
    y_max = X[1,:].max() + 5

    support_vevtor = X[(alpha < C) & (alpha > 0)] #支持向量
    
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    X_test = np.c_[XX.ravel(), YY.ravel()]

    Z= svm_predict(X_test, X, y, alpha, b)
    Z1 = np.array(Z)
    Z1 = Z1.reshape(XX.shape)

    plt.figure(figsize=(10, 10))
    plt.scatter(support_vevtor[:, 0], support_vevtor[:, 1], s=80,
                facecolors='none', zorder=10, edgecolors='k')
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, edgecolors='k') # zorder相同的数值，是表示一个画布上在同一图层
    plt.pcolormesh(XX, YY, Z1>0, cmap=plt.cm.Paired) #给半边颜色着色

    plt.contour(XX, YY, Z1, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])                 #绘制等高线

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())




# def foo(a, b, k1,k2):
#     print(a,b, k1,k2)
#
# A = (1, 2, 4)
# B = {'k1':'v1', 'k2':'v2'}
#
# foo(1, 2, **B)





if __name__ == '__main__':
    # X = np.random.ra(1,100, 40).reshape(20, 2)
    # y = np.array([1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1])
    X = np.c_[(.4, -.7),
              (-1.5, -1),
              (-1.4, -.9),
              (-1.3, -1.2),
              (-1.1, -.2),
              (-1.2, -.4),
              (-.5, 1.2),
              (-1.5, 2.1),
              (1, 1),
              # --
              (1.3, .8),
              (1.2, .5),
              (.2, -2),
              (.5, -2.4),
              (.2, -2.3),
              (0, -2.7),
              (1.3, 2.1)].T
    y = [0] * 8 + [1] * 8

    C = 1
    b, alpha  = smo(X, y, C, 50)

    # plot散点图及等高线
    coef = {'alpha':alpha, 'b': b, 'C': C}
    plot_svm(X, y, **coef)

