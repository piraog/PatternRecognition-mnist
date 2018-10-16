import os
import numpy as np
import matplotlib.pyplot as plt
from mnist import *
from sklearn.decomposition import PCA
from sklearn import svm,metrics

#import the datasets
X, image_shape, Y = get_training_set('mnist')
X_test, (_, _), Y_test = get_test_set('mnist')

#normalise and cropp the images (4 pixels margin)
X = X.astype(np.float64).reshape(60000,28,28)[:,4:24,4:24]
X = X.reshape(60000,400)
rawX = (X-X.mean(axis=0))/263

X_test = X_test.astype(np.float64).reshape(10000,28,28)[:,4:24,4:24]
X_test = X_test.reshape(10000,400)
rawX_test = (X_test-X_test.mean(axis=0))/263


def linRaw():
    #linear SVM on raw data
    acc = []
    for c in [0.01,0.1,10]:
        clf = svm.SVC(kernel='linear',C=c)
        clf.fit(rawX, Y)
        results = np.rint(clf.predict(rawX_test))
        print(results)
        accuracy = sum(results==Y_test)/len(Y_test)
        print('Linear SVM Accuracy with no dimension reduction, for C= ', c, ' ', accuracy)
        acc.append(accuracy)
    return acc
    '''Results:
            0.9404
            0.9449
            0.9377
            0.9281

            The parameter C, common to all SVM kernels, trades off misclassification of training examples against simplicity of the decision surface. A low C makes the decision surface smooth, while a high C aims at classifying all training examples correctly. gamma defines how much influence a single training example has. The larger gamma is, the closer other examples must be to be affected.
    '''

def plot_lin():
    '''Train and evaluate linear SVMs on the dataset reduced to different dimensions using PCA
    for different values of C, and plot the accuracies'''

    Acc = [[0.9404,0.9449,0.9377,0.9281]]      #linRaw values, they take a very long time to compute
    for dim in [40,80,200]:
        pca = PCA(n_components = dim)
        X = pca.fit_transform(rawX)
        X_test = pca.transform(rawX_test)
        acc = []
        for c in [0.01,0.1,1,10]:
            clf = svm.SVC(kernel='linear',C=c)
            clf.fit(X, Y)
            results = np.rint(clf.predict(X_test))
            accuracy = sum(results==Y_test)/len(Y_test)
            acc.append(accuracy)
            print('Linear SVM Accuracy with ',dim, ' dimensions, for C= ', c, ' ', accuracy)
        Acc.append(acc)
    print(Acc)
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('C', fontsize = 15)
    ax.set_ylabel('Accuracy', fontsize = 15)
    ax.set_title('Accuracy versus C for different dimension reductions using linear SVMs', fontsize = 15)
    C = [0.01,0.1,1,10]
    for acc in Acc:
        ax.plot(C,acc)
    ax.legend(['raw', 'd=40', 'd=80', 'd=200'])
    plt.show()
'''
Acc = [[0.9404, 0.9449, 0.9377, 0.9281],[0.9231, 0.933, 0.9333, 0.9325],[0.9384, 0.9405, 0.9404, 0.94],[0.9399, 0.944, 0.94 0.9356]]
'''


def RBF40():
    #RBF SVM on data reduced to dim 40 by PCA
    pca = PCA(n_components = 40)
    X = pca.fit_transform(rawX)
    X_test = pca.transform(rawX_test)
    acc = []
    for c in [0.01,0.1,1,10]:
        clf = svm.SVC(C=c,gamma='auto')
        print('ok')
        clf.fit(X, Y)
        print('ok2')
        results = np.rint(clf.predict(X_test))
        print(results)
        accuracy = sum(results==Y_test)/len(Y_test)
        print('Accuracy for C= ', c, ' ', accuracy)
        acc.append(accuracy)
    return acc
    '''Results for the range of C:
            0.9357
            0.9666
            0.9817
            0.9849

            The parameter C, common to all SVM kernels, trades off misclassification of training examples against simplicity of the decision surface. A low C makes the decision surface smooth, while a high C aims at classifying all training examples correctly. gamma defines how much influence a single training example has. The larger gamma is, the closer other examples must be to be affected.
    '''

if __name__ == '__main__':
    #linRaw()
    #plot_lin()
    #RBF40()
