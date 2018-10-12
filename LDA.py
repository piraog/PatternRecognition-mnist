import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la
from mnist import *
from collections import Counter
from mpl_toolkits.mplot3d import axes3d, Axes3D
from sklearn import neighbors


def LDA(data, target, dims_rescaled_data=2):
    '''main LDA function
    given the data, compute the LDA matrix and returns:
    - the data projected along the dims_rescaled_data most important eigenvectors
    - the eigenvalues
    - the eigenvectors
    '''

    n = len(data)
    mean = data.mean(axis=0)
    Sw = np.zeros((data.shape[1],data.shape[1]))
    Sb = np.zeros((data.shape[1],data.shape[1]))
    for cat in range(10):
        idx = target==cat
        data_cat = data[idx]
        n_cat = len(data_cat)
        mean_cat = data_cat.mean(axis=0)
        S = 1/n_cat * np.matmul((data_cat-mean_cat).T,data_cat-mean_cat)

        Sw += n_cat/n * S
        Sb += n_cat/n * np.matmul((mean_cat-mean).reshape(len(mean),1),(mean_cat-mean).reshape(1,len(mean)))


    LDA = la.inv(Sw)*Sb

    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric, 
    # the performance gain is substantial
    evals, evecs = la.eig(LDA)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return np.dot(data,evecs), evals, evecs

def plot_eig(data, target,image_shape, nb_vects=10):
    '''Plots the nb_vects most weighted eigenvectors of the LDA matrix obtained with data'''

    _, _, evecs = LDA(data, target, nb_vects)
    print(np.amax(evecs),np.amin(evecs))
    indices = np.array(range(nb_vects))
    images = data[indices].reshape((nb_vects, image_shape[0], image_shape[1]))
    for i, image in enumerate(images):
        plt.subplot(5, 5, i+1)
        plt.imshow(image, cmap='Greys', vmin=0, vmax=255, interpolation='none')
        plt.title(str(i))
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_lda(data,target):
    '''Plots the vectors of data projected in 2D using 2 dimensions LDA'''

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Projection direction 1', fontsize = 15)
    ax.set_ylabel('Projection direction 2', fontsize = 15)
    ax.set_title('2 Component LDA', fontsize = 20)

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    clf = LinearDiscriminantAnalysis()
    proj_data = clf.fit_transform(data, target)[:,:2]
    #proj_data,_,_ = LDA(data,target)
    targets = [0,1,2,3,4,5,6,7,8,9]
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for target, color in zip(targets,colors):
        indicesToKeep = Y == target
        ax.scatter(proj_data[indicesToKeep,0]
                   , proj_data[indicesToKeep,1]
                   , c = color
                   , s = 2)
    ax.legend(targets)
    ax.grid()

    plt.show()

def plot_lda_3D(data,target):
    '''Plots the vectors of data projected in 3D using 3 dimensions LDA'''

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(111, projection='3d') 
    ax.set_xlabel('Projection direction 1', fontsize = 15)
    ax.set_ylabel('Projection direction 2', fontsize = 15)
    ax.set_zlabel('Projection direction 3', fontsize = 15)
    ax.set_title('3 Component LDA', fontsize = 20)


    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    clf = LinearDiscriminantAnalysis()
    proj_data = clf.fit_transform(data, target)[:,:3]
    targets = [0,1,2,3,4,5,6,7,8,9]
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for target, color in zip(targets,colors):
        indicesToKeep = Y == target
        ax.scatter(proj_data[indicesToKeep,0]
                   , proj_data[indicesToKeep,1]
                   , proj_data[indicesToKeep,2]
                   , c = color
                   , s = 2)
    ax.legend(targets)
    ax.grid()

    plt.show()

def lda_1NN(train, Y, test, Y_test, dim_rescale):
    '''Trains the 1 nearest neighbour algorithm over data, projected along dim_rescale directions
    using LDA and print the accuracy on test projected along the same directions as data'''

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    clf = LinearDiscriminantAnalysis()
    proj_train = clf.fit_transform(train, Y)[:,:dim_rescale]

    proj_test = clf.transform(test)[:,:dim_rescale]

    #compute distance of the results with the training set and select the closest neighboor
    NN = neighbors.KNeighborsClassifier(1, weights='distance')
    NN.fit(proj_train,Y)

    results = NN.predict(proj_test)

    accuracy = sum(results==Y_test)/len(Y_test)
    print(accuracy)


if __name__=='__main__':
    #import the datasets
    X, image_shape, Y = get_training_set('mnist')
    X_test, (_, _), Y_test = get_test_set('mnist')

    #normalise and cropp the images (4 pixels margin)
    X = X.astype(np.float64).reshape(60000,28,28)[:,4:24,4:24]
    X = X.reshape(60000,400)
    X = (X-X.mean(axis=0))/263

    X_test = X_test.astype(np.float64).reshape(10000,28,28)[:,4:24,4:24]
    X_test = X_test.reshape(10000,400)
    X_test = (X_test-X_test.mean(axis=0))/263


    #LDA(X,Y)
    #plot_lda(X, Y)
    plot_lda_3D(X, Y)
    #plot_eig(X,Y,(20,20),25)
    #lda_1NN(X,Y,X_test,Y_test,2)           #0.440     #good for some classes (0.84 for class2), bad for others (0.26 for class3)
    #lda_1NN(X,Y,X_test,Y_test,3)           #0.667
    #lda_1NN(X,Y,X_test,Y_test,9)           #0.899
    #so not so good since we can't get a high enough number of dimensions to get enough precision

    #since 10 categories, the largest number of dim obtainable is 9
    #proof: the 11th eigenvalue is close to 0, ame for all the 11+ eigenvalues
    