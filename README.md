A simple python script to try different ML techniques on MNIST.

All the code used for this project is in python. It uses general libraries (numpy, matplotlib, mpl_toolkits) as well as specialized ones (sklearn, tensorflow, keras).
Before running the code please install all the aforementioned packages.

The code is split in 6 files:
-	mnist.py is a toolbox to read the mnist files, it is then imported and used in every question
-	PCA.py tackles PCA and its applications
-	LDA.py tackles LDA and its applications
-	SVM.py implements SVMs with differents parameters and evaluate their performances 
-	network.py and train.py  implement a CNN and train it on mnist before evaluating its performances

You need to add a folder named mnist containing the full mnist dataset (4 parts) in the same folder as these files before running the code.

The files PCA.py, LDA.py and SVM.py each contain various functions and end with a section calling these functions to display the performances. Please uncomment the function you want to investigate to make it run, each function is fully independent of the other.

The file network.py describes the architecture of the CNN network we train and evaluate with train.py. Please run train.py to witness the training process, you can refer to the results in the report directly as the training takes some time to complete.

