__authors__ = '[1587634,1587646,1633426,1633623]'
__group__ = 'DL.11'

import numpy as np
from utils import rgb2gray
from scipy.spatial.distance import cdist
from collections import Counter


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        #Check if all array's components are floats
        if train_data.dtype != 'float64': 
            train_data = np.array(train_data,dtype=float)
        self.train_data = train_data
        
        if len(train_data.shape) == 2: # If the array has 2 dimensions, it means that it is already a vector, so we can return it
            return self.train_data
        elif len(train_data.shape) == 3: # If the array has 3 dimensions, it means that it is a grayscale image, so we have to turn it into a vector
            self.train_data = train_data.reshape(train_data.shape[0],train_data.shape[1]*train_data.shape[2])
            return self.train_data
        elif len(train_data.shape) == 4: #If the array has 4 dimensions, it means that it is a color image, so we have to turn it grayscale
            greyscale_train= rgb2gray(self.train_data)
            self.train_data = greyscale_train.reshape(train_data.shape[0],train_data.shape[1]*train_data.shape[2])
            return self.train_data
        else:
            raise Exception("X array has more than 4 dimensions")

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        if len(test_data.shape) == 2: # If the array has 2 dimensions, it means that it is already a vector, so we can return it
            test_data = test_data
        elif len(test_data.shape) == 3: # If the array has 3 dimensions, it means that it is a grayscale image, so we have to turn it into a vector
            test_data = test_data.reshape(test_data.shape[0],test_data.shape[1]*test_data.shape[2])
        elif len(test_data.shape) == 4: #If the array has 4 dimensions, it means that it is a color image, so we have to turn it grayscale
            greyscale_train= rgb2gray(test_data)
            test_data = greyscale_train.reshape(test_data.shape[0],test_data.shape[1]*test_data.shape[2])
        else:
            raise Exception("X array has more than 4 dimensions")
        
        distances = cdist(test_data, self.train_data, 'euclidean') # Returns a NxT distances matrix, where N is the number of rows in test_data and T is the number of rows in self.train_data
        
        argsort = np.argsort(distances, axis=1)[:, :k] # Sorts the distances and returns the indexes of the k nearest neighbors, from a NxT matrix, returns a Nxk matrix
        
        self.neighbors = self.labels[argsort] # Returns the labels of the k nearest neighbors

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 2 numpy array of Nx1 elements.
                1st array For each of the rows in self.neighbors gets the most voted value
                            (i.e. the class at which that row belongs)
                2nd array For each of the rows in self.neighbors gets the % of votes for the winning class
        """
        # Get the number of rows in self.neighbors
        num_rows = len(self.neighbors)

        # Initialize the most voted class and percentage of votes arrays
        most_voted = np.empty(num_rows, dtype=object)
        percent_votes = np.empty(num_rows, dtype=float)

        # Loop through the rows in self.neighbors
        for i, row in enumerate(self.neighbors):
            # Calculate the occurrences of each class
            counter = Counter(row)
            
            # Get the most voted class and its count
            most_voted_class, count = counter.most_common(1)[0] # To acces the most common tuple

            # Calculate the percentage of votes for the winning class
            percent_vote = count / len(row) * 100

            # Update the most voted and percentage of votes arrays
            most_voted[i] = most_voted_class
            percent_votes[i] = percent_vote
        
        return most_voted, percent_votes
        
    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class (2 Nx1 vector, 1st the class 2nd the  % of votes it got
        """
        self.get_k_neighbours(test_data, k)
        return self.get_class()