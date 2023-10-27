__authors__ = '[1587634,1587646,1633426,1633623]'
__group__ = 'DL.11'

import time
import numpy as np
from Kmeans import *
from KNN import *
from utils_data import *
import random
from sklearn.decomposition import PCA

if __name__ == '__main__': 

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')
        
    train_imgs_gray, train_class_labels_gray, train_color_labels_gray, test_imgs_gray, test_class_labels_gray, \
        test_color_labels_gray = read_dataset(root_folder='./images/', gt_json='./images/gt.json', with_color=False)

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    # You can start coding your functions here
    
    def retrieval_by_color_probs(images, labels, probabilities, colors):
        '''
        Returns the images that have the specified colors, sorted by the sum of the probabilities of these colors.

        Args:
            images (list): List of images.
            labels (list): List of labels (colors) for the images.
            probabilities (list): List of probabilities corresponding to the labels.
            colors (list): List of colors to search for.

        Returns:
            found_images: List of found images sorted by the sum of the probabilities of the specified colors.
        '''
        # Create a list of tuples (image, total_probability) for the images that have the specified colors
        image_probs = [(img, sum(prob for color, prob in zip(label, probs) if color in colors))
                    for img, label, probs in zip(images, labels, probabilities) if all(color in label for color in colors)]
        
        # Sort the list by total_probability in descending order
        image_probs.sort(key=lambda x: x[1], reverse=True)

        # Get the sorted images
        found_images = [img for img, _ in image_probs]

        return found_images
    
    def retrieval_by_color(images, labels, colors):
        '''
        returns the images that have the specified color, without considering the probabilities
        '''
        found_images = []
        for img, label in zip(images, labels):
            if all(color in label for color in colors): # si volguéssim buscar algun any(color in label for color in colors)
                found_images.append(img)
        return found_images

    '''
    selected_images = retrieval_by_color(train_imgs, train_color_labels, ['Red'])
    visualize_retrieval(selected_images,10)
    '''
    '''
    img_labels=[]
    img_prob=[]
    for img in train_imgs:
        Kmeans = KMeans(img,4)
        Kmeans.fit()
        colors, prob = get_colors(Kmeans.centroids, True)
        img_labels.append(colors)
        img_prob.append(prob)
    
    selected_images = retrieval_by_color_probs(train_imgs, img_labels, img_prob, ["Red"])
    visualize_retrieval(selected_images,8)
    '''
    
    def retrieval_by_shape(images,labels,probabilities, shape):
        '''
        returns the images that have the specified shape
        '''

        founded_images = []
        for img, label, prob in zip(images, labels, probabilities):
            if shape in label:
                founded_images.append((img,prob))
        sorted_images = [img for img, prob in sorted(founded_images, key=lambda x: x[1], reverse=True)]
        
        return sorted_images
    
    '''
    Knn = KNN(train_imgs,train_class_labels)
    class_labels, probabilities = Knn.predict(test_imgs,15) #s'haurà de comparar amb test_class_labels
    
    selected_images = retrieval_by_shape(test_imgs,class_labels,probabilities,"Shorts")
    print(len(selected_images))
    visualize_retrieval(selected_images,10)
    '''

    def retrieval_combined(images, class_labels, shape_probs, color_labels, shape, colors):
        '''
        returns the images that have the specified shape and color
        '''
        im1=np.array(retrieval_by_shape(images, class_labels, shape_probs, shape))
        im2=np.array(retrieval_by_color(images, color_labels, colors))
        
        # Convert the image arrays to lists of tuples
        im1_list = [tuple(image.flatten()) for image in im1]
        print(len(im1_list))
        im2_list = [tuple(image.flatten()) for image in im2]
        print(len(im2_list))

        # Find the intersection of the lists
        intersection_list = [img for img in im1_list if img in im2_list]

        # Convert the intersection back to a numpy array
        intersection = np.array([np.array(image).reshape(images.shape[1:]) for image in intersection_list])

        return intersection


    Knn = KNN(train_imgs,train_class_labels)
    class_labels, shape_probs = Knn.predict(test_data=test_imgs,k=5) #s'haurà de comparar amb test_class_labels
    color_labels=[]
    for img in train_imgs:
        Kmeans = KMeans(img,4)
        Kmeans.fit()
        colors = get_colors(Kmeans.centroids)
        color_labels.append(colors)
    selected_images = retrieval_combined(test_imgs, class_labels, shape_probs, color_labels, "Flip Flops", ["Blue"])
    print(len(selected_images))
    visualize_retrieval(selected_images, 10)


    def Kmeans_statistics_time(image, fit, param):
        '''
        #returns list of time, fit and num_iter for each K
        '''
        start = time.time()
        kmeans = KMeans(image, options = {"fitting" : fit})
        kmeans.find_bestK(10, parameter = param)
        end = time.time()
        return end-start
    
    def Kmeans_statistics_kpredicted(image, fit, param):
        '''
        #returns list of time, fit and num_iter for each K
        '''
        kmeans = KMeans(image, options = {"fitting" : fit})
        kmeans.find_bestK(10, parameter = param)
        return kmeans.K
                    
    def Kmeans_statistics_avg(random_images, fit, param, parameter = "time"):
        data_list = []

        for image in random_images:
            if parameter == "time":
                data = Kmeans_statistics_time(image, fit, param)
            elif parameter == "k":
                data = Kmeans_statistics_kpredicted(image, fit, param)
            data_list.append(data)

        avg_data = np.mean(np.array(data_list)).tolist()

        return avg_data
    '''
    random_indices = np.random.choice(len(train_imgs), 20, replace=False)
    random_images = train_imgs[random_indices]
    
    criteria = ['WCD','Fisher','ICD']
    params = [10,20,30,50,60,90,95]

    avg_data = {}
    for param in params:
        avg_data[param] = []
        for criterion in criteria:
            data = Kmeans_statistics_avg(random_images, criterion, param, parameter = "k")
            avg_data[param].append(data)

    
    barWidth = 0.25
    bars = list(range(len(params)))

    # Set position of bar on X axis
    r1 = np.arange(len(params))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    plt.bar(r1, [avg_data[param][0] for param in params], color='b', width=barWidth, edgecolor='grey', label=criteria[0])
    plt.bar(r2, [avg_data[param][1] for param in params], color='r', width=barWidth, edgecolor='grey', label=criteria[1])
    plt.bar(r3, [avg_data[param][2] for param in params], color='g', width=barWidth, edgecolor='grey', label=criteria[2])

    plt.xlabel('Params')
    plt.ylabel('Predicted k') # plt.ylabel('Average Time (seconds)')
    plt.xticks([r + barWidth for r in range(len(params))], params)

    plt.legend()
    plt.show()
    
    '''

    def get_shape_accuracy(class_labels, ground_truth):
        '''
        returns the accuracy of the shape classification
        '''
        # calculate the accuracy
        accuracy = np.mean(np.array(ground_truth) == np.array(class_labels))
        
        return accuracy
    
    '''
    accuracy =[]
    Knn = KNN(train_imgs, train_class_labels)
    for k in range(1,30,2):
        class_labels = Knn.predict(test_imgs, k)[0] #To get only the labels
        accuracy.append(get_shape_accuracy(class_labels, test_class_labels))
        print("Accuracy for k = ", k, " is ", accuracy[-1])
    plt.plot(np.arange(1,30,2),accuracy)
    plt.xlabel('Number of neighbours')
    plt.ylabel('Accuracy')
    plt.title('KNN accuracy in function of the neighbours considered')
    plt.show()
    print("The maximum accuracy is ", max(accuracy), " for k = ", 2*accuracy.index(max(accuracy))+1)
    '''

    def get_color_accuracy(color_labels, ground_truth):
        '''
        returns the accuracy of the color classification
        '''
        accuracy = 0
        for label, gt in zip(color_labels, ground_truth):
            label = np.array(label)  # convert label to array
            gt = np.array(gt)  # convert gt to array
            accuracy += np.intersect1d(label, gt).size / np.union1d(label, gt).size
        accuracy = accuracy / len(ground_truth)
        return accuracy
    
    '''
    max_labels = 8
    
    accuracy = []
    for k in range(1, max_labels+1):
        labels=[]
        for image in cropped_images:
            Kmeans= KMeans(image,k)
            Kmeans.fit()
            labels.append(get_colors(Kmeans.centroids))
        accuracy.append(get_color_accuracy(labels, color_labels))

    plt.plot(range(1,max_labels+1),accuracy)
    print(f"The max accuracy for  is", max(accuracy), "with", accuracy.index(max(accuracy))+1, "centroids")


    plt.xlabel('Number of centroids')
    plt.xticks(np.arange(1, max_labels+1))
    plt.grid()
    plt.ylabel('Accuracy')
    plt.title('K-means accuracy in function of the number of centroids')
    plt.show()
    '''
    '''
    accuracy = []
    max_labels = 7 # perquè print(max(test_color_labels, key=len)) és 6
    for k in range(1,max_labels+1): 
        labels=[]
        for i in range(cropped_images.shape[0]):
            image = cropped_images[i]
            Kmeans= KMeans(image,k)
            Kmeans.fit()
            labels.append(get_colors(Kmeans.centroids))
        accuracy.append(get_color_accuracy(labels, color_labels))

    plt.plot(range(1,max_labels+1),accuracy)
    plt.xlabel('Number of centroids')
    plt.xticks(np.arange(1, max_labels+1))
    plt.grid()
    plt.ylabel('Accuracy')
    plt.title('K-means accuracy in function of the number of centroids')
    plt.show()
    print("The max accuracy is", max(accuracy), "with", accuracy.index(max(accuracy))+1, "centroids")
    '''
    
    #1 KNN PER VALOR MIG DELS PIXELES DE LES IMATGES
    #CALCULAR VALOR MIG DE LES IMATGES DEL TEST I DEL TRAIN

    '''
    train_imgs = train_imgs.reshape(train_imgs.shape[0],train_imgs.shape[1]*train_imgs.shape[2], train_imgs.shape[3])
    test_imgs = test_imgs.reshape(test_imgs.shape[0],test_imgs.shape[1]*test_imgs.shape[2], test_imgs.shape[3])

    train_mean_pix = np.mean(train_imgs, axis=2)
    test_mean_pix = np.mean(test_imgs, axis=2)

    Knn=KNN(train_mean_pix,train_class_labels)
    class_labels, probabilities = Knn.predict(test_mean_pix,15) #s'haurà de comparar amb test_class_labels

    selected_images = retrieval_by_shape(test_mean_pix,class_labels,probabilities,"Shorts")

    selected_images = [image.reshape(80,60) for image in selected_images]
    visualize_retrieval(selected_images,10)
    '''

    # 2 KNN PER VARIANÇA DELS PIXELES DE LES IMATGES
    # CALCULAR VARIANÇA DELS PIXELS DE LES IMATGES DEL TEST I DEL TRAIN

    '''
    train_imgs = train_imgs.reshape(train_imgs.shape[0],train_imgs.shape[1]*train_imgs.shape[2], train_imgs.shape[3])
    test_imgs = test_imgs.reshape(test_imgs.shape[0],test_imgs.shape[1]*test_imgs.shape[2], test_imgs.shape[3])

    train_var_pix = np.var(train_imgs, axis=2)
    test_var_pix = np.var(test_imgs, axis=2)

    Knn = KNN(train_var_pix, train_class_labels)
    class_labels, probabilities = Knn.predict(test_var_pix, 15) #s'haurà de comparar amb test_class_labels

    selected_images = retrieval_by_shape(test_var_pix,class_labels,probabilities,"Shorts")
    selected_images = [image.reshape(80,60) for image in selected_images]
    print(len(selected_images))
    visualize_retrieval(selected_images,10)
    '''

    def plot_accuracy_knn(k_values, other = False):
        accuracy_img =[]
        train_imgs_aux = train_imgs.reshape(train_imgs.shape[0],train_imgs.shape[1]*train_imgs.shape[2], train_imgs.shape[3])
        test_imgs_aux = test_imgs.reshape(test_imgs.shape[0],test_imgs.shape[1]*test_imgs.shape[2], test_imgs.shape[3])
        Knn_img = KNN(train_imgs, train_class_labels)

        if other:
            train_mean_pix = np.mean(train_imgs_aux, axis=2)
            test_mean_pix = np.mean(test_imgs_aux, axis=2)

            train_var_pix = np.var(train_imgs_aux, axis=2)
            test_var_pix = np.var(test_imgs_aux, axis=2)

            accuracy_mean = []
            accuracy_var = []

            Knn_mean = KNN(train_mean_pix, train_class_labels)
            Knn_var = KNN(train_var_pix, train_class_labels)

        for k in k_values:
            class_labels_img = Knn_img.predict(test_imgs, k)[0]
            accuracy_img.append(get_shape_accuracy(class_labels_img, test_class_labels))
            if other:
                class_labels_mean = Knn_mean.predict(test_mean_pix, k)[0]
                class_labels_var = Knn_var.predict(test_var_pix, k)[0]
                accuracy_mean.append(get_shape_accuracy(class_labels_mean, test_class_labels))
                accuracy_var.append(get_shape_accuracy(class_labels_var, test_class_labels))

        plt.plot(k_values ,accuracy_img, label = "Train images")
        if other:
            plt.plot(k_values ,accuracy_mean, label = "Mean pixels images")
            plt.plot(k_values ,accuracy_var, label = "Variance pixels images")
        plt.xlabel("k")
        plt.ylabel("Accuracy")
        plt.title("KNN accuracy in function of the neighbours considered")
        plt.legend()
        plt.show()
    
    #plot_accuracy_knn(np.arange(1,20,2))
    #plot_accuracy_knn(np.arange(1,20,2), other = True)
    
    def knn_nfold_cross_validation(data, labels, n, k , std = False):
        """
        Perform n-fold cross-validation on a given dataset and model.

        Parameters:
            data (numpy.ndarray): The input data.
            labels (numpy.ndarray): The corresponding labels for the input data.
            n (int): The number of folds.
            k (int): The number of neighbours to use in the KNN model.

        Returns:
            (float): The average accuracy across all folds.
        """
        indices = list(range(len(data)))
        random.shuffle(indices)
        split_indices = np.array_split(indices, n)
        accuracies = []

        for i in range(n):
            # Create train and test sets
            test_indices = split_indices[i]
            train_indices = np.concatenate(split_indices[:i] + split_indices[i+1:])
            train_data, train_labels = data[train_indices], labels[train_indices]
            test_data, test_labels = data[test_indices], labels[test_indices]

            # Train and predict with the model
            knn = KNN(train_data, train_labels)
            predictions = knn.predict(test_data, k)[0]

            # Calculate accuracy
            accuracy = (predictions == test_labels).mean()
            accuracies.append(accuracy)
            std_acurracy = np.std(accuracies)
            mean_accuracy = np.mean(accuracies)
        if std:
            return mean_accuracy, std_acurracy
        else:
            return mean_accuracy
        
    def plot_accuracy_cross_validation(k_values, n_folds, std = False, others = False):

        accuracy_img_cross =[]
        train_imgs_aux = train_imgs.reshape(train_imgs.shape[0],train_imgs.shape[1]*train_imgs.shape[2], train_imgs.shape[3])
        std_img_cross = []

        if others:  
            train_mean_pix = np.mean(train_imgs_aux, axis=2)
            train_var_pix = np.var(train_imgs_aux, axis=2)

            accuracy_mean_cross = []
            accuracy_var_cross = []

            std_mean_cross = []
            std_var_cross = []

        for k in k_values:
            if std:
                accuracy_img, std_img = knn_nfold_cross_validation(train_imgs, train_class_labels, n_folds, k, std = std)
                accuracy_img_cross.append(accuracy_img)
                std_img_cross.append(std_img)
                if others:
                    accuracy_mean, std_mean = knn_nfold_cross_validation(train_mean_pix, train_class_labels, n_folds, k, std = std)
                    accuracy_var, std_var = knn_nfold_cross_validation(train_var_pix, train_class_labels, n_folds, k, std = std)
                    accuracy_mean_cross.append(accuracy_mean)
                    accuracy_var_cross.append(accuracy_var)
                    std_mean_cross.append(std_mean)
                    std_var_cross.append(std_var)

            else:
                accuracy_img, std_img = knn_nfold_cross_validation(train_imgs, train_class_labels, n_folds, k, std = std)
                accuracy_img_cross.append(accuracy_img)
                if others:
                    accuracy_mean, std_mean = knn_nfold_cross_validation(train_mean_pix, train_class_labels, n_folds, k, std = std)
                    accuracy_var, std_var = knn_nfold_cross_validation(train_var_pix, train_class_labels, n_folds, k, std = std)
                    accuracy_mean_cross.append(accuracy_mean)
                    accuracy_var_cross.append(accuracy_var)


        plt.plot(k_values, accuracy_img_cross, label = "Train Img")

        if others:
            plt.plot(k_values, accuracy_mean_cross, label = "Train Mean Pix")
            plt.plot(k_values, accuracy_var_cross, label = "Train Var Pix")
        if std:
            plt.plot(k_values, np.array(accuracy_img_cross) + np.array(std_img_cross), linestyle = "dashed", color = "blue") 
            plt.plot(k_values, np.array(accuracy_img_cross) - np.array(std_img_cross), linestyle = "dashed", color = "blue", label = "Std")

        plt.xlabel('k')
        plt.ylabel('Accuracy')
        plt.title('Cross validation accuracy for KNN in function of the number of neighbours')
        plt.grid(True)
        plt.legend()
        plt.show()
        
    #plot_accuracy_cross_validation(np.arange(1,20 ,2), 5, std = True)

    def pca_sklearn(X_train, X_test, n_components):
        # Paso 1: Flattening de las imágenes
        X_train= X_train.reshape(X_train.shape[0], -1)
        X_test= X_test.reshape(X_test.shape[0], -1)
        # Paso 2: Normalizar los datos
        X_norm = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
        
        # Paso 3: Calcular los componentes principales
        pca = PCA(n_components=n_components)
        pca.fit(X_norm)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
        
        return X_train, X_test
    
    def explained_variance(X, n_components):
        # Paso 1: Flattening de las imágenes
        X = X.reshape(X.shape[0], -1)
        # Paso 2: Normalizar los datos
        X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        # Realizar PCA para diferentes números de componentes
        explained_variances = []
        for n in n_components:
            pca = PCA(n_components=n)
            pca.fit(X_norm)
            explained_variances.append(sum(pca.explained_variance_ratio_))

        return explained_variances

    
    def plot_accuracy_pca(n_components, evariance = False):
        if evariance:
            explained = explained_variance(train_imgs_gray, n_components)
            plt.plot(n_components, explained, 'o-')
            plt.xlabel('Number of components')
            plt.ylabel('Explained variance')
            plt.title('Explained variance in function of the number of components')
            plt.show()
        accuracies_pca = []
        accuracies_pca_cross = []
        for n in n_components:
            pca_train , pca_test = pca_sklearn(train_imgs_gray, test_imgs_gray, n)
            accuracy_cross = knn_nfold_cross_validation(pca_train, train_class_labels_gray, 5, 2)
            knn_pca = KNN(pca_train, train_class_labels_gray)
            predictions = knn_pca.predict(pca_test, 2)[0]
            accuracy = get_shape_accuracy(predictions, test_class_labels_gray) 
            accuracies_pca.append(accuracy)
            accuracies_pca_cross.append(accuracy_cross)
        
        #find elbow point
        dif = np.diff(accuracies_pca)
        for i in range(1,len(dif)):
            if dif[i]/dif[i-1] < 0.1 :
                elbow_point = n_components[i]
                arg = i
                break
        npac =  np.array(accuracies_pca)
        if evariance:
            print("Elbow point: ", elbow_point, "with accuracy: ", accuracies_pca[arg], "and explained variance: ", explained[arg])
            print("Max accuracy using PCA with", n_components[np.argmax(npac)], "and explained variance: ", explained[np.argmax(npac)])
        else:
            print("Elbow point: ", elbow_point, "with accuracy: ", accuracies_pca[arg])
            print("Max accuracy using PCA with", n_components[np.argmax(npac)])

        plt.plot(n_components, accuracies_pca, label = 'Accuracy')
        plt.plot(n_components, accuracies_pca_cross, label = 'Accuracy cross validation')
        plt.xlabel('N components')
        plt.ylabel('Accuracy')
        plt.title('Accuracy PCA')
        plt.legend()
        plt.show()
    
    #n_components = [i for i in range(1, 10)] + [i for i in range(10, 100, 10)] + [i for i in range(100, 500, 50)]
    #plot_accuracy_pca(n_components, evariance = True)

    def time_complexity(train_imgs, train_class_labels, test_imgs, type = "Normal", n_components = 20):
        '''
        returns the time complexity of the KNN algorithm
        '''
        
        if type == "Normal":
            start = time.time()
            Knn = KNN(train_imgs, train_class_labels)
        elif type == "PCA":
            start = time.time()
            train_imgs_gray = rgb2gray(train_imgs)
            pca_imgs, test_imgs = pca_sklearn(train_imgs_gray, test_imgs_gray, n_components)
            Knn = KNN(pca_imgs, train_class_labels)
        elif type == "Mean":
            start = time.time()
            train_imgs_aux = train_imgs.reshape(train_imgs.shape[0],train_imgs.shape[1]*train_imgs.shape[2], train_imgs.shape[3])
            test_imgs_aux = test_imgs.reshape(test_imgs.shape[0],test_imgs.shape[1]*test_imgs.shape[2], test_imgs.shape[3])
            train_mean_pix = np.mean(train_imgs_aux, axis=2)
            test_imgs = np.mean(test_imgs_aux, axis=2)
            Knn = KNN(train_mean_pix, train_class_labels)
        elif type == "Var":
            start = time.time()
            train_imgs_aux = train_imgs.reshape(train_imgs.shape[0],train_imgs.shape[1]*train_imgs.shape[2], train_imgs.shape[3])
            test_imgs_aux = test_imgs.reshape(test_imgs.shape[0],test_imgs.shape[1]*test_imgs.shape[2], test_imgs.shape[3])
            train_var_pix = np.var(train_imgs_aux, axis=2)
            test_imgs = np.var(test_imgs_aux, axis=2)
            Knn = KNN(train_var_pix, train_class_labels)
        else:
            raise ValueError("Type must be Normal, PCA, Mean or Var")
        
        Knn.predict(test_imgs, 2)
        end = time.time()
        train_time = end-start
        
        return train_time


    """
    time_normal = time_complexity(train_imgs, train_class_labels, test_imgs, type = "Normal")
    time_pca = time_complexity(train_imgs, train_class_labels, test_imgs, type = "PCA")
    time_max_pca = time_complexity(train_imgs, train_class_labels, test_imgs, type = "PCA", n_components = 350)
    time_mean = time_complexity(train_imgs, train_class_labels, test_imgs, type = "Mean")
    time_var = time_complexity(train_imgs, train_class_labels, test_imgs, type = "Var")

    plt.bar(["Normal", "PCA elbow", "PCA max", "Mean", "Var"], [time_normal, time_pca,time_max_pca , time_mean, time_var], color = ["blue", "orange", "green", "red", "purple"])
    plt.title("Time complexity")
    plt.xlabel("Type")
    plt.ylabel("Time (s)")
    plt.show()
    """