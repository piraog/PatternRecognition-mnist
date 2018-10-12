import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la
from mnist import *
from collections import Counter
from mpl_toolkits.mplot3d import axes3d, Axes3D
from sklearn import neighbors

def PCA(data, dims_rescaled_data=2):
    '''main PCA function
    given the data, compute the covariance matrix and returns:
    - the data projected along the dims_rescaled_data most important eigenvectors
    - the eigenvalues
    - the eigenvectors
    '''
    #compute the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    evals, evecs = la.eig(R)
    # sort eigenvalue and eigenvectors in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]
    # select the first dims_rescaled_data most weighted vectors
    evecs = evecs[:, :dims_rescaled_data]
    # projects the data alongside the eigenvectors and return the values
    return np.dot(evecs.T, data.T).T, evals, evecs

def plot_eig(data, image_shape, nb_vects=10):
    '''Plots the nb_vects most weighted eigenvectors of the covariance matrix obtained with data'''

    _, _, evecs = PCA(data, nb_vects)
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


def plot_pca(data):
    '''Plots the vectors of data projected in 2D using 2 dimensions PCA'''

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 Component PCA', fontsize = 20)

    proj_data, _, _ = PCA(data)
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

def plot_pca_3D(data):
    '''Plots the vectors of data projected in 3D using 3 dimensions PCA'''

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(111, projection='3d') 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_zlabel('Principal Component 3', fontsize = 15)
    ax.set_title('3 Component PCA', fontsize = 20)


    proj_data, _, _ = PCA(data,3)
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


def plot_energy(data):
    '''Plot the cumulative variance energy of the ordered eigenvalues
    and print the smallest number of eigenvectors that conserve 95 percent of the total energy'''
    _, eigvals, _ = PCA(data)
    total_energy = sum(eigvals)
    energy = eigvals/total_energy
    energy = np.cumsum(energy)

    d = sum(energy<0.95)+1
    print(r'dimension for 95% energy: ',d)
    plt.plot(energy)
    plt.xlabel('number of dimensions')
    plt.ylabel('energy')
    plt.show()

def pca_1NN(train, test, dim_rescale):
    '''Trains the 1 nearest neighbour algorithm over data, projected along dim_rescale directions
    using PCA and print the accuracy on test projected along the same directions as data'''

    R = np.cov(train, rowvar=False)
    evals, evecs = la.eigh(R)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    evals = evals[idx]
    evecs = evecs[:, :dim_rescale]

    proj_train = np.dot(evecs.T, train.T).T
    proj_test = np.dot(evecs.T, test.T).T


    #compute distance of the results with the training set and select the closest neighboor
    clf = neighbors.KNeighborsClassifier(1, weights='distance')
    clf.fit(proj_train,Y)

    results = clf.predict(proj_test)

    accuracy = sum(results==Y_test)/len(Y_test)
    print(accuracy)

def raw1NN(train,test):
    '''Trains the 1 nearest neighbour algorithm over the raw data and print the accuracy on test set'''

    #compute distance of the results with the training set and select the closest neighboor
    clf = neighbors.KNeighborsClassifier(1, weights='distance')
    clf.fit(train,Y)

    results = clf.predict(test)

    accuracy = sum(results==Y_test)/len(Y_test)
    print(accuracy)


if __name__ == '__main__':
    X, image_shape, Y = get_training_set('mnist')
    X_test, (_, _), Y_test = get_test_set('mnist')

    #normalise and cropp the images (4 pixels margin)
    X = X.astype(np.float64).reshape(60000,28,28)[:,4:24,4:24]
    X = X.reshape(60000,400)
    X = (X-X.mean(axis=0))/263

    X_test = X_test.astype(np.float64).reshape(10000,28,28)[:,4:24,4:24]
    X_test = X_test.reshape(10000,400)
    X_test = (X_test-X_test.mean(axis=0))/263

    #plot_pca(X)
    plot_pca_3D(X)
    #plot_eig(X,image_shape,25)
    #plot_energy(X)              #d = 153
    #raw1NN(X,X_test)
    #pca_1NN(X,X_test,40)        #0.9735 / 0.9733 with normalised and cropped data
    #pca_1NN(X,X_test,80)        #0.9729 / 0.9716 with normalised and cropped data
    #pca_1NN(X,X_test,200)        #0.9691 / 0.969 with normalised data
    #pca_1NN(X,X_test,153)        #0.9694 / 0.9691 with normalized data
    #pca_1NN(X,X_test,2)            #0.392 with normalized data, comparable with LDA      0.87 for class 1, 0.26 for class 6