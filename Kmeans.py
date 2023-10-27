__authors__ = '[1633426,1633623,1587634,1587646]'
__group__ = 'DL.11'

import numpy as np
import utils


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        #Check if all array's components are floats
        if X.dtype != 'float64': 
            X = np.array(X,dtype=float)
        self.X = X
        #Check if X is a 2D matrix 
        #If len(X.shape) it is correct   OBS:"X.shape returns a list of the dimension of the matrix [2,3], 2 rows, 3 columns"
        if len(X.shape) == 2:
            return self.X
        elif len(X.shape) == 3: #case where de dims are F x D x 3, we have to do [FxD,3]
            self.X = X.reshape(X.shape[0]*X.shape[1], X.shape[2])
            return self.X
        else:
            raise Exception("X array has more than 3 dimensions")

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0.0001
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################


    def _init_centroids(self):
        """
        Initialization of centroids
        """
        if self.options['km_init'].lower() == 'first': # if the option is first
            self.centroids = np.zeros((self.K, self.X.shape[1]))
            self.old_centroids = np.zeros((self.K, self.X.shape[1]))
            different_points = []
            for i in range(self.X.shape[0]): # for each point in X
                if len(different_points) == self.K: # if we have K different points
                    break
                if not any((self.X[i] == x).all() for x in different_points): # check if point is already in list, any returns true if any element is true
                    different_points.append(self.X[i])
            self.centroids = np.array(different_points)
            self.old_centroids = np.array(different_points)
            
        elif self.options['km_init'].lower() == 'random':

            self.centroids = np.random.randint(0, 255, size=(self.K, self.X.shape[1]))
            self.old_centroids = np.random.randint(0, 255, size=(self.K, self.X.shape[1]))

        elif self.options['km_init'].lower() == 'equidistant':
            n = self.X.shape[0]  # number of points
            different_points = []
            self.centroids = np.zeros((self.K, self.X.shape[1]))
            self.old_centroids = np.zeros((self.K, self.X.shape[1]))
            for i in range(self.K):  # for each centroid
                step = int((i + 1) * n / (self.K + 1))  # interval length (number of points in each interval
                j = 0
                while j + step >= 0 and any((self.X[step + j] == x).all() for x in different_points):
                    j -= 1
                different_points.append(self.X[step + j])
            self.centroids = np.array(different_points)
            self.old_centroids = np.array(different_points)

        elif self.options['km_init'].lower() == 'kmeans++':
            self.centroids = np.zeros((self.K, self.X.shape[1]))
            self.old_centroids = np.zeros((self.K, self.X.shape[1]))
            self.centroids[0] = self.X[np.random.randint(self.X.shape[0])]
            for i in range(1, self.K):
                while True:
                    distances = np.array([min(np.linalg.norm(point - c) ** 2 for c in self.centroids[:i]) for point in self.X])
                    probabilities = distances / distances.sum() #The furthest points from centroids have higher probabilities.
                    new_centroid_idx = np.random.choice(len(self.X), p=probabilities)
                    new_centroid = self.X[new_centroid_idx]
                    if not any((new_centroid == c).all() for c in self.centroids[:i]): # check if new_centroid is already in the list
                        self.centroids[i] = new_centroid
                        break
            self.old_centroids = self.centroids.copy()

            # not necessary to check if centroids are different because we are using random.choice with probailities equal to 0
        else:
            raise Exception("Invalid initialization provided")


    def get_labels(self):
        """        
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """
        """
        # Old implementation:
        self.labels = np.random.randint(self.K, size=self.X.shape[0])
        for i in range(self.X.shape[0]): # for each point in X
            for j in range(self.K): # for each centroid
                if j == 0: # if it is the first centroid
                    min_dist = np.linalg.norm(self.X[i] - self.centroids[j]) 
                    self.labels[i] = j
                else: # if it is not the first centroid
                    dist = np.linalg.norm(self.X[i] - self.centroids[j])
                    if dist < min_dist:
                        min_dist = dist
                        self.labels[i] = j
        """
        distances = np.linalg.norm(self.X[:, np.newaxis] - self.centroids, axis=2)
        self.labels = np.argmin(distances, axis=1)


    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        
        # for i in range(self.K): # for each centroid
        #     self.old_centroids[i] = self.centroids[i]
        #     points = [] # list of points assigned to centroid
        #     for j in range(self.X.shape[0]): # for each point in X
        #         if self.labels[j] == i: # if point is assigned to centroid
        #             points.append(self.X[j])
        #     self.centroids[i] = np.mean(points, axis=0) # calculate mean of points assigned to centroid, axis=0 means that we calculate the mean of each column

        for i in range(self.K): # for each centroid
            self.old_centroids[i] = self.centroids[i] # save old centroid
            self.centroids[i] = np.mean(self.X[self.labels == i], axis=0) # calculate mean of points assigned to centroid, axis=0 means that we calculate the mean of each column
    
    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        #needed????
        if self.num_iter >= self.options['max_iter']: # if we have reached the maximum number of iterations
            return True
        
        for i in range(self.K): # for each centroid
            #Check distance between old and new centroid lower than tolerance
            if np.linalg.norm(self.centroids[i] - self.old_centroids[i]) > self.options['tolerance']: # if the distance is bigger than tolerance
                return False 
        return True

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        self._init_centroids()
        self.get_labels()
        self.get_centroids()
        while not self.converges():
            self.get_labels()
            self.get_centroids()
            self.num_iter += 1
    
    def withinClassDistance(self):
        """
        returns the within class distance of the current clustering
        """
        '''
        WCD = 0
        for i in range(self.K): # for each centroid
            WCD += np.linalg.norm(self.X[self.labels == i] - self.centroids[i], axis=1)/len(self.X[self.labels == i])
        return WCD
        '''
    
        WCD = 0
        for i in range(self.K): # for each centroid
            WCD += np.sum(np.linalg.norm(self.X[self.labels == i] - self.centroids[i], axis=1)**2)
        return WCD
        
    def interClassDistance(self):
        """
        Computes the interclass distance between each pair of centroids
        """
        '''
        # Old implementation (from the power point slides)
        ICD = 0.0
        for i in range(self.K):
            for j in range(i+1, self.K):
                ICD += np.linalg.norm(self.centroids[i] - self.centroids[j])
        return 2*ICD/(self.K*(self.K-1))
        '''
        ICD = 0
        for i in range(self.K):
            for j in range(i+1, self.K):
                ICD += np.linalg.norm(self.centroids[i] - self.centroids[j])**2 #sum of centroid distances to the power of two
        return ICD
    
    def fisherCriterion(self):
        """
        Computes the Fisher criterion of the current clustering
        """
        eps = 1e-10
        return self.withinClassDistance() / (self.interClassDistance()+eps)
    

    def find_bestK(self, max_K, parameter = 20):
        """
        sets the best k anlysing the results up to 'max_K' clusters
        """
        self.K = 2
        self.fit()
        #INTRA
        if (self.options['fitting'].upper() == 'WCD'):
            WCD_prev = self.withinClassDistance()
            for k in range(2, max_K): # we do not calculate kmeans(max_k)
                self.K = k+1
                self.fit()
                WCD = self.withinClassDistance()
                relation = 100 * (WCD / WCD_prev)
                if relation > 100 - parameter:  # paràmetre petit implica canvi petit (relació gran=>dif petita entre les dues)
                    self.K = k
                    return
                WCD_prev = WCD
        #INTER    
        elif self.options['fitting'].upper() == 'ICD':
            self.K = 3
            self.fit()
            ICD_prev = self.interClassDistance()
            for k in range(2, max_K):
                self.K=k+1
                self.fit()
                ICD = self.interClassDistance()
                relation = 100 * (ICD / ICD_prev)
                if relation > parameter + 100: #paràmetre gran implica canvi gran (relació gran => dif gran entre les dues)
                    self.K = k
                    return
                ICD_prev = ICD
        #FISHER
        elif self.options['fitting'].lower() =='fisher':
            fisher_prev = self.fisherCriterion()
            for k in range(2, max_K):
                self.K=k+1
                self.fit()
                fisher = self.fisherCriterion()
                relation = 100 * (fisher / fisher_prev)
                if relation > 100 - parameter:
                    self.K = k
                    return
                fisher_prev = fisher
        else:
            raise ValueError("Invalid fitting method")
        return
    
def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """
    '''
    # Old implementation
    dist = np.zeros((X.shape[0], C.shape[0]))
    for i in range(X.shape[0]): # for each pixel
        for j in range(C.shape[0]): # for each centroid
            dist[i, j] = np.linalg.norm(X[i, :] - C[j, :]) # Euclidean distance
            #for k in range(X.shape[1]): # for each dimension
                #dist[i, j] += (X[i, k] - C[j, k]) ** 2 #
            dist[i,j] = np.sqrt(dist[i,j])
    return dist
    '''
    
    dist = np.sqrt(np.sum((X[:, np.newaxis, :] - C) ** 2, axis=2))
    return dist

def distance_manhattan(X, C):
    """
    Calculates the manhattan distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """
    dist = np.sum(np.abs(X[:, np.newaxis, :] - C), axis=2)
    return dist
        
def get_colors(centroids, retrival = False):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K tuples each containing the most probable color and its corresponding probability
    """
    centroids_colors = []
    colors = utils.colors
    colors_matrix = utils.get_color_prob(centroids)
    for i in range(colors_matrix.shape[0]): 
        max_index = np.argmax(colors_matrix[i, :])
        if not retrival:
            centroids_colors.append(colors[max_index])
        else:
            centroids_colors.append((colors[max_index], colors_matrix[i, max_index]))

    if not retrival:
        return centroids_colors

    color_max_probs = {}
    for color, prob in centroids_colors:
        if not np.isnan(prob):
            if color not in color_max_probs or prob > color_max_probs[color]:
                color_max_probs[color] = prob

    color_labels, probabilities = zip(*color_max_probs.items())

    return list(color_labels), list(probabilities)
