'''
A Deep learning tool kit with a Neural Network model implementation, in addition to various helper functions to ease the applications.
'''
# Importing necessary libraries:
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image
from PIL import Image
from os import listdir, getcwd
import pandas as pd
from datetime import datetime
datetime.now()

# 01 ________________________________________________________________________________________________________________________________________________________________

def prepare_image_data(images_path, resize = 64, label_tag = 1, show_rejected_images = False):
    '''
    A function to prepare one class of images into column arrays. Works only on images of RGB color mode.
    
    Arguments:
        images_path: A path containing the class of images.
        resize: The hight and width in pixels to get all images into the same square dimension of (resize * resize * 3), where 3 is for RGB channels.
        label_tag : A categorical label tag whether 0 or 1, 1 by default.
        show_rejected_images : Whether to show the rejected images or not, they are saved by default in a returned list 'rejected_pics', False by default.
    
    Returns:
        pics_array: A 4D array of shape (number of converted images, resize, resize, 3) containing the arrays of converted images.
        labels_array: A 2D array of shape (1, number of converted images) containing the lagel tage assigned.
        rejected_pics: A list containing the name of each rejected image.
    '''
    pics_list = list()
    rejected_pics = list()
      
    for picture in listdir(images_path):
        pic = Image.open(images_path + picture)
        
        try:
            pic_resized = pic.resize((resize, resize))
            pic_array = np.asarray(pic_resized)
            assert(np.shape(pic_array) == (resize, resize, 3))

        except:
            rejected_pics.append((picture, pic))
            if show_rejected_images:
                print(picture)
                print(pic)
                plt.imshow(pic)
                plt.show()
                print('-' * 50)
            continue
            
        pics_list.append(pic_array)
        
    pics_array = np.array(pics_list).reshape((len(pics_list), resize, resize, 3))
    labels_array = np.zeros((1, len(pics_list))) + label_tag
    
    print('Pics Array shape:', np.shape(pics_array))
    print('Labels Array shape:', np.shape(labels_array))
    
    return pics_array, labels_array, rejected_pics

# 02 ________________________________________________________________________________________________________________________________________________________________

def merge_shuffle_split(images_array_1, labels_array_1, images_array_2, labels_array_2, validation_split = 0.2, seed = 123):
    '''
    Given two pairs of images and thier labels, outputed by 'prepare_image_data' function, where each pair pertaines to a certain class, the function returns
    a merged, shuflled, and splited into train / test  arrays.
    
    Arguments:
        images_array_1: A 4D array of shape (number images, hight, width, 3) containing the images arrays of one class.
        labels_array_1: A 2D array of shape (1, number images) containing the labels of images in images_array_1.
        images_array_2: Same as images_array_1 for the other class of images.
        labels_array_2: Same as labels_array_1 for the ohter class of images.
        validation_split: Percentage of validation/test set out of all images (number of images in images_array_1 + number of images in images_array_2).
        seed: The seed to be set for the random shuffle of combained images.
    
    Returns:
        train_set_x_orig: The combianed images_array_1 and images_array_2 excluding the validation/test set (the "_orig" is to indicates that the array is yet to be standardized).
        train_set_y: The labels of train_set_x_orig. 
        test_set_x_orig: Same as train_set_x_orig for the validation / test set array.
        test_set_y: The labels of test_set_x_orig.
    '''
    np.random.seed(seed)
    # Merging the images and labels arrays (merge)
    images_array = np.concatenate((images_array_1, images_array_2), axis = 0)
    labels_array = np.concatenate((labels_array_1, labels_array_2), axis = 1)
    
    # Creating indices to shuffle (shuffle)
    indices = np.arange(images_array.shape[0])
    np.random.shuffle(indices)
    
    # Shuffling the merged arrays
    images_array = images_array[indices]
    labels_array = labels_array[:, indices]
    
    # Creating the train sets (split)
    train_set_x_orig = images_array[:int((1 - validation_split) * len(images_array))]
    train_set_y = labels_array[:, :int((1 - validation_split) * len(images_array))]
    
    # Creating the test sets
    test_set_x_orig = images_array[int((1 - validation_split) * len(images_array)):]
    test_set_y = labels_array[:, int((1 - validation_split) * len(images_array)):]
    
    print('Output Shapes:')
    print('train_set_x_orig:', np.shape(train_set_x_orig))
    print('train_set_y:', np.shape(train_set_y))
    print('test_set_x_orig:', np.shape(test_set_x_orig))
    print('test_set_y:', np.shape(test_set_y))
    
    return train_set_x_orig, train_set_y, test_set_x_orig, test_set_y

# 03 ________________________________________________________________________________________________________________________________________________________________

def prepare_image_arrays(set_x, stdr_method = 'pixel_max'):
    '''
    Given an images array, the function retunrs a flatten and standerdized version of it by dividing over the max pixel value of 255.
    
    Arguments:
        set_x: A 4D array of shape (number images, hight, width, 3), the output of merge_shuffle_split funtion.
        stdr_method: Indicator for the standardization method to be used. Currently only one method is availabel, dviding set_x by max pixel value of 255.
    
    Returns:
        set_x_flatten_stdr: The flattened and standardized set_x.
    '''
    set_x_flatten = set_x.reshape(set_x.shape[0], -1).T
#     if standardize == 'pixel_max':
    set_x_flatten_stdr = set_x_flatten / 255.
#     else:
#         set_x_flatten_stdr = (set_x_flatten - np.mean(set_x_flatten)) / np.std(set_x_flatten)
    print('Shape of Flatten and Standardized array:', np.shape(set_x_flatten_stdr))
    
    return set_x_flatten_stdr

# 04 ________________________________________________________________________________________________________________________________________________________________

def sigmoid(set_x):
    return (1 / (1 + np.exp(-set_x)))

# 05 ________________________________________________________________________________________________________________________________________________________________

def initialize_parameters(dim):
    b = 0
    w = np.random.randn(dim, 1) * 0
    
    return w, b

# 06 ________________________________________________________________________________________________________________________________________________________________

def cost_calc(a, set_y):
    return - np.sum(np.add(np.dot(set_y, np.log(a.T)), np.dot(1 - set_y, np.log(1 - a.T)))) / set_y.shape[1]

# 07 ________________________________________________________________________________________________________________________________________________________________

def forward_pass(set_x, set_y, w, b):
    costs = list()
    z = np.dot(w.T, set_x) + b
    a = sigmoid(z)
    cost = cost_calc(a, set_y)
    costs.append(cost)
    
    return w, b, z, a, costs

# 08 ________________________________________________________________________________________________________________________________________________________________

def optimize(set_x, set_y, num_iterations, learning_rate, print_cost):
    m = set_x.shape[1]
    w, b = initialize_parameters(set_x.shape[0])
    for i in range(num_iterations):
        w, b, z, a, costs = forward_pass(set_x, set_y, w, b)
        dz = a - set_y
        dw = np.dot(set_x, dz.T) / m
        db = np.sum(dz) / m
        w -= learning_rate * dw
        b -= learning_rate * db
        if i % 1000 == 0 and print_cost:
            print('Cost after iteration {}: {}'.format(i, costs[-1].round(4)))
        
    return w, b, z, a, costs

# 09 ________________________________________________________________________________________________________________________________________________________________

def predict(w, b, set_x, set_y):
    w, b, z, a, costs = forward_pass(set_x, set_y, w, b)
    yhat = a
    for i in range(a.shape[1]):
        if a[0, i] > 0.5:
            yhat[0, i] = 1
        else:
            yhat[0, i] = 0
            
    return yhat

# 10 ________________________________________________________________________________________________________________________________________________________________

def logistic_nn_model(set_x_train, set_y_train, set_x_test, set_y_test, num_iterations = 1000, learning_rate = 0.001, print_cost = False):
    
    w, b, z, a, costs = optimize(set_x_train, set_y_train, num_iterations, learning_rate, print_cost)
    
    yhat_train = predict(w, b, set_x_train, set_y_train)
    
    yhat_test = predict(w, b, set_x_test, set_y_test)
    
    train_acc = (100 - np.mean(np.abs(yhat_train - set_y_train)) * 100).round(4)
    test_acc = (100 - np.mean(np.abs(yhat_test - set_y_test)) * 100).round(4)
    
    print('Train Accuracy: {}%'.format(train_acc))
    print('Test Accuracy: {}%'.format(test_acc))
    
    model_summary = {'Model No. ': str(datetime.now()), 'Train X Shape': np.shape(set_x_train), 'Train Y Sahpe': np.shape(set_y_train),
                    'Test X Shape': np.shape(set_x_test), 'Test Y Sahpe': np.shape(set_y_test),
                     'Iterations': num_iterations, 'alpha': learning_rate,
                     'w': w, 'b': b, 'Costs': costs,
                    'Train Accuracy': train_acc, 'Test Accuracy': test_acc}
    
    return model_summary

# 11 ________________________________________________________________________________________________________________________________________________________________

def models_summary(models_list):
    '''
    Given a list of models resulting from the 'logistic_nn_model' or 'deep_nn_model' functions, it returns a pandas data frame and an index within that
    frame of the model with the highest test accuracy then train accuracy.
    
    Arguments:
        models_list: A list of models outputed by either 'logistic_nn_model' or 'deep_nn_model' functions.
    
    Retunrs:
        model_summary: A pandas data frame of the given models list.
    '''
    models_df = pd.DataFrame.from_dict(models_list)
    
    best_test_accuracy = models_df['Test Accuracy'].max()
    
    best_test_models = models_df[models_df['Test Accuracy'] == best_test_accuracy]
    
    local_best_train = best_test_models['Train Accuracy'].max()
    
    top_model_number = models_df.index[(models_df['Test Accuracy'] == best_test_accuracy) & (models_df['Train Accuracy'] == local_best_train)].tolist()
    print('Top models, based on Test then Train accuracies:', top_model_number)
    
    return models_df, top_model_number[0]

# 12 ________________________________________________________________________________________________________________________________________________________________

def predict_sample(sample_path, w, b):
    pics_array, labels_array, rejected_pics = prepare_image_data(sample_path, label_tag = 1, show_rejected_images = False)
    set_x_flatten_stdr = prepare_image_arrays(pics_array, standardize = 'pixel_max')
    yhat = predict(w, b, set_x_flatten_stdr, labels_array)
    
    return yhat

# 13 ________________________________________________________________________________________________________________________________________________________________

def deep_nn_model(X, Y, X_test, Y_test, mini_batch_size = 128, layer_structure = [5, 3, 1], iterations = 1000, alpha = 0.001,
                  lambd = 0, dropout_layers = [], keep_prob = 1, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8,
                  print_cost = True, print_every = 500, show_plots = True, seed = 0):
    '''
    An 'L' deep neural network model with regularization parameters for L2 and Dropout.
    
    Arguments:
        X: Features array of shape (number of features, number of examples).
        Y: Labels array of shape (1, number of examples).
        X_test: same as X used for testing.
        Y_test: same as Y used for testing.
        layer_structure: A list of the number of nodes per hidden layer, so len(layer_structure) = number of hidden layers.
        iterations: number of epochs.
        alpha: learning rate.
        lambd: L2 regularization parameter.
        dropout_layer: a list of the layer number to be masked.
        keep_prob: The probability of keeping a node active in the masked layers.
        print_cost: If Ture, prints the cost and training accuracy every specific number of iterations.
        print_every: The number of iterations before printing the cost and training accuracy (if print_cost == True)
        show_plots: If True, plots and shows costs over iterations.
        
    Returns:
        model_summary: A dictionary with varoius model information.
    '''
    np.random.seed(seed)
    start = datetime.now() # to measure training time (start).
    model_structure = layer_structure.copy() # to include the inpout layer of shape (X.shape[0], number of images / examples).
    model_structure.insert(0, X.shape[0]) # including the input layer and its dimensions.
    num_layers = len(model_structure) # total number of layers in the model including the input layer (layer 0).
    L = num_layers - 1 # number of hidden layers in the model.
    
    ## Initialize the parameters:
    P = dict() # parameters dictionary
    V = dict() # momentum parameters dictionary, used to store the exponentially moving averages of the gradients. 
    S = dict() # RMSProp parameters dictionary, used to store the exponentially moving averages of the squared gradients.
    
    for l in range(1, num_layers): # for every hidden layer in the model, create the set of parameters below.
        P['W' + str(l)] = np.random.randn(model_structure[l], model_structure[l - 1]) * np.sqrt(2 / model_structure[l - 1]) # random initialization with 'He' scaling.
        P['b' + str(l)] = np.zeros((model_structure[l], 1)) # zero inialization.
        
        V['dW' + str(l)] = np.zeros((P['W' + str(l)].shape)) # initializing momentum variables for the wieghtes.
        V['db' + str(l)] = np.zeros((P['b' + str(l)].shape)) # initializing momentum variables for the baises.
        
        S['dW' + str(l)] = np.zeros((P['W' + str(l)].shape)) # initializing RMSProp variables for the wieghtes.
        S['db' + str(l)] = np.zeros((P['b' + str(l)].shape)) # initializing RMSProp variables for the baises.
                
    # Dictionaries to run and save the forward and backward propagation results:
    Z = dict() # linear forward pass.
    A = dict() # forward activation.
    D = dict() # dropout mask.
    dZ = dict() # linear backward pass.
    dP = dict() # parameters gradients.
    dA = dict() # backward activation.
        
    costs = list() # to save cost values per iteration.
    
    ## Forward Propagation:
    X_train = X
    Y_train = Y
    adam_counter = 1
    for i in range(iterations): # over each iteration.
        X = X_train
        Y = Y_train
        ## Mini-Batches:
        m = Y.shape[1] # number of traning examples.
        
        indices = np.arange(m) # creating indices to shuffle X and Y for mini-batch creation.
        seed += 1 # to update the seed in order to get a different shuffle each iteration.
        np.random.shuffle(indices)
        X = X[:, indices] # shuffling X.
        Y = Y[:, indices].reshape((1, m)) # shuffling Y, reshaping added to make sure Y shape is maintained.
        
        num_full_mini_batches = int(m / mini_batch_size) # number of full (of size mini_batch_size) mini mini-batches.
        left_over_exampels = ((m / mini_batch_size) - num_full_mini_batches) * mini_batch_size # number of examples in the last mini-batch (less than mini_batch_size).
        mini_batches_list = list() # list to keep mini-batches for use in the training.
        
        for j in range(num_full_mini_batches): # creating the full mini-batches.
            mini_batch_X = X[:, j * mini_batch_size: (1 + j) * mini_batch_size]
            mini_batch_Y = Y[:, j * mini_batch_size: (1 + j) * mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches_list.append(mini_batch)
            
        if m % mini_batch_size != 0: # creating the last mini-batch, if any.
            batched_examples = int(m - left_over_exampels)
            mini_batch_X = X[:, batched_examples: m]
            mini_batch_Y = Y[:, batched_examples: m]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches_list.append(mini_batch)
        
        for mini_batch in mini_batches_list: # looping over mini-batches.
            X, Y = mini_batch # unpack first mini-batch into X and Y.
            A['A0'] = X # to intialize the forward pass.
            
            mini_batch_m = Y.shape[1]
            for l in range(1, num_layers): # for every layer in the model>
    #             if 0 in dropout_layers: # adding a dropout mask to the input layer.
    #                 D['D' + str(0)] = np.random.rand(A['A' + str(0)].shape[0], A['A' + str(0)].shape[1]) # generating mask D0.
    #                 D['D' + str(0)] = (D['D' + str(0)] < keep_prob).astype('int') # setting valuse to 0s and 1s based on keep_prob as a threshold.
    #                 A['A' + str(0)] *= D['D' + str(0)] / keep_prob # applying mask and scaling back A0 to maintain expected value (inverted dropout).
    
                Z['Z' + str(l)] = np.dot(P['W' + str(l)], A['A' + str(l - 1)]) + P['b' + str(l)] # linear forward pass.
        
                if l < L: # if this is not the last hidden layer in the model then:
                    A['A' + str(l)] = np.maximum(0, Z['Z' + str(l)]) # calculate the activation as a RelU function.
                    
                    if (len(dropout_layers) != 0) and (keep_prob < 1.0) and (l in dropout_layers):
                        D['D' + str(l)] = np.random.rand(A['A' + str(l)].shape[0], A['A' + str(l)].shape[1])
                        D['D' + str(l)] = (D['D' + str(l)] < keep_prob).astype('int')
                        A['A' + str(l)] *= D['D' + str(l)] / keep_prob
                else:
                    A['A' + str(l)] = 1 / (1 + np.exp(-(Z['Z' + str(l)]))) # else if it's the last hidden layer, then calculate the activation as a 
                                                                           # sigmoid fucntion (since the task is binary classification).

            cross_entropy_cost = - np.sum(np.add(np.dot(Y, np.log(A['A' + str(L)].T)), np.dot(1 - Y, np.log(1 - A['A' + str(L)].T)))) / mini_batch_m # calculates the cross entropy (first part of the cost).
            L2_regularization_cost = 0 # initialize the L2 regularization term.
            
            for l in range(1, num_layers): # to be applied on each of the W parameters.
                L2_regularization_cost += np.sum(np.square(P['W' + str(l)])) # calculating L2 regularization term (first part).
                
            L2_regularization_cost = L2_regularization_cost * lambd / (2 * mini_batch_m) # scaling regularization term by lambda over two m (second part).
            cost = cross_entropy_cost + L2_regularization_cost # calculating cost by adding L2 regularization term to the first part of cost (second part of the cost).
            cost = np.squeeze(cost) # insure it's not a rank one array.
            assert(cost.shape == ()) # raise error if it is not a scalar.
            costs.append(cost) # append it to the costs list.
            
            Yhat_train = A['A' + str(L)] # final output (Yhat).
            Yhat_train = np.array((Yhat_train > 0.5) * 1).reshape(1, mini_batch_m) # converting to 0s and 1s based on 0.5 threshold.
            train_acc = (100 - np.mean(np.abs(Yhat_train - Y)) * 100).round(4) # calculate accuracy using the final output.
                            
        ## Backward Propagation:   
            dA['dA' + str(L)] = - (np.divide(Y, A['A' + str(L)]) - np.divide(1 - Y, 1 - A['A' + str(L)])) # initializing backward propagation.
            dZ['dZ' + str(L)] = dA['dA' + str(L)] * A['A' + str(L)] * (1 - A['A' + str(L)]) # sigmoid activation backwared
            
            for l in reversed(range(1, num_layers)): # for every layer in the model, going last to first, calculate:
                dP['dW' + str(l)] = np.dot(dZ['dZ' + str(l)], A['A' + str(l - 1)].T) / mini_batch_m + (P['W' + str(l)] * lambd / mini_batch_m)# Ws gradients with regularization.
                dP['db' + str(l)] = np.sum(dZ['dZ' + str(l)], axis = 1, keepdims = True) / mini_batch_m # bs gradients.
                
                if l > 1: # As long as this is not the first layer, then calcualte:
                    dA['dA' + str(l - 1)] = np.dot(P['W' + str(l)].T, dZ['dZ' + str(l)]) # Relu activations gradients.
                    
                    if (len(dropout_layers) != 0) and (keep_prob < 1.0) and (l - 1 in dropout_layers):
                        dA['dA' + str(l - 1)] *= D['D' + str(l - 1)] / keep_prob # scaling back the activation gradients to maintaine the expected value 
                                                                                # of the hidden layers' output (Inverted Dropout).
                    dZ['dZ' + str(l - 1)] = np.array(dA['dA' + str(l - 1)], copy=True) # to calculate dZ_l-1.
                    dZ['dZ' + str(l - 1)][Z['Z' + str(l - 1)] <= 0] = 0 # the gradient of the linear activation at dZ_l-1.
                    
            Vc = dict() # corrected momentum parameters dictionary, used to store the exponentially moving averages of the gradients. 
            Sc = dict() # corrected RMSProp parameters dictionary, used to store the exponentially moving averages of the squared gradients.
        ## Updating the parameters:    
            for l in range(1, num_layers): # for every hidden layer in the model:
                V['dW' + str(l)] = beta1 * V['dW' + str(l)] + (1 - beta1) * dP['dW' + str(l)]
                V['db' + str(l)] = beta1 * V['db' + str(l)] + (1 - beta1) * dP['db' + str(l)]
                
                Vc['dW' + str(l)] = V['dW' + str(l)] / (1 - np.power(beta1, adam_counter))
                Vc['db' + str(l)] = V['db' + str(l)] / (1 - np.power(beta1, adam_counter))
                
                S['dW' + str(l)] = beta2 * S['dW' + str(l)] + (1 - beta2) * np.power(dP['dW' + str(l)], 2)
                S['db' + str(l)] = beta2 * S['db' + str(l)] + (1 - beta2) * np.power(dP['db' + str(l)], 2)
                
                Sc['dW' + str(l)] = S['dW' + str(l)] / (1 - np.power(beta2, adam_counter))
                Sc['db' + str(l)] = S['db' + str(l)] / (1 - np.power(beta2, adam_counter))
                
                P['W' + str(l)] = P['W' + str(l)] - alpha * (Vc['dW' + str(l)] / np.sqrt(Sc['dW' + str(l)] + epsilon)) # update Ws.
                P['b' + str(l)] = P['b' + str(l)] - alpha * (Vc['db' + str(l)] / np.sqrt(Sc['db' + str(l)] + epsilon)) # update bs.
                                
#                 P['W' + str(l)] -= alpha * dP['dW' + str(l)] # update Ws. Uncomment to discard the use of Adam Optimizer.
#                 P['b' + str(l)] -= alpha * dP['db' + str(l)] # update bs. Uncomment to discard the use of Adam Optimizer.
                
            adam_counter += 1 # used with the values of V & S to correct them and produce Vc & Sc.
            
        if print_cost and i % print_every == 0: # to print the cost and training accuracy if set to Ture, every number of iterations based on 'print_every' argument.
            print('Iteration {} : Cost: {}, Train Acc.: {}%'.format(i, cost.round(6), train_acc.round(4))) # round the cost and accuracy and print them.
            
    end = datetime.now() # to measure training time (end).
    
    ## Predictions on test set:
    m_test = Y_test.shape[1] # number of test examples.
    A_test = dict() # activations dictionary.
    A_test['A0'] = X_test # initializing to calculate the linear forward pass.
    
    for l in range(1, num_layers): # for every hidden layer in the model, calculate:
            Z['Z' + str(l)] = np.dot(P['W' + str(l)], A_test['A' + str(l - 1)]) + P['b' + str(l)] # linear forward pass.
            if l < L: # if this is not the last layer:
                A_test['A' + str(l)] = np.maximum(0, Z['Z' + str(l)]) # calculate the activations as ReLU functions.
            else: # otherwise:
                A_test['A' + str(l)] = 1 / (1 + np.exp(-(Z['Z' + str(l)]))) # calculate as sigmoid functions.
                
    Yhat_test = A_test['A' + str(L)] # final output (Yhat_test)
    Yhat_test = np.array((Yhat_test > 0.5) * 1).reshape(1, m_test) # converting to 0s and 1s based on 0.5 threshold.
    
    train_acc = (100 - np.mean(np.abs(Yhat_train - Y)) * 100).round(4) # calculating the training accuracy.
    test_acc = (100 - np.mean(np.abs(Yhat_test - Y_test)) * 100).round(4) # calculating the testing accuracy.
    
    print('Train Accuracy: {}%'.format(train_acc)) # printing train accuracy.
    print('Test Accuracy: {}%'.format(test_acc)) # printing test accuracy.
    
    if show_plots: # if 'show_plots' argument is set to True, show the costs plots:
        sub_costs = [costs[i] for i in range(len(costs)) if i % len(mini_batches_list) == 0] # list of costs resulting from full iterations.
        plt.plot(np.squeeze(sub_costs)) # plot costs resulting from full iterations.
#         plt.plot(np.squeeze(costs)) # ploting the costs over iterations.
        plt.ylabel('cost') # labeling the y axis.
        plt.xlabel('iterations') # labeling the x axis.
        #plt.xticks(np.arange(0, len(costs), step = iterations / 100))
        plt.title('model struc.: ' + str(model_structure) + '.' + ' alpha = ' + str(alpha)) # title of the plot showing the learning rate and model structure.
        plt.show() # to show the plot.
    
    ## Model Summary
    model_summary = {'Model No.': str(datetime.now()), 'Model Structure': tuple(model_structure),
                     'Training Time': str(end - start),
                     'Number of Parameters': len(P), 'Train X Shape': np.shape(X), 'Train Y Sahpe': np.shape(Y),
                     'Test X Shape': np.shape(X_test), 'Test Y Sahpe': np.shape(Y_test),
                     'Iterations': iterations, 'alpha': alpha,
                     'P': P, 'Costs': costs, 'Train Accuracy': train_acc, 'Test Accuracy': test_acc, 'Dropout Masks': D,
                     'Regularization Lambd': lambd, 'Keep Prob.': keep_prob, 'Dropout Layers': tuple(sorted(dropout_layers)),
                     'Mini Batch Size': mini_batch_size, 'beta1': beta1, 'beta2': beta2, 'epsilon': epsilon}
    
    return model_summary # the dictionary with model summary information returned.

# 14 ________________________________________________________________________________________________________________________________________________________________

def deep_nn_model_predict(sample_path = None, resize = 100, model = None):
    '''
    Given a path containing images, the function returns a prediction for its class using the provided model.
    
    Arguments:
        sample_path: A string, the path containing the images for which the calss so wished to be predicted.
        resize: An integer, the dimension to set the images to, must equal resize of the 'prepare_image_data' function.
        model: A dictionary, the model to be used for prediciton.
    '''
    pics_array = prepare_image_data(sample_path, resize)[0]
    set_x_flatten_stdr = prepare_image_arrays(pics_array)
    num_layers = len(model['Model Structure'])
    L = num_layers - 1
    P = model['P']
    Z = dict()
    A = dict()
    A['A0'] = set_x_flatten_stdr    
    for l in range(1, num_layers):
        Z['Z' + str(l)] = np.dot(P['W' + str(l)], A['A' + str(l - 1)]) + P['b' + str(l)]
        if l < L:
            A['A' + str(l)] = np.maximum(0, Z['Z' + str(l)])
        else:
            A['A' + str(l)] = 1 / (1 + np.exp(-(Z['Z' + str(l)])))
                
    Yhat = A['A' + str(L)]
    Yhat = np.array((Yhat > 0.5) * 1).reshape(1, len(pics_array))
    
    plt.figure(figsize = (15, 12))
    i = 1
    num_pics = len(listdir(sample_path))
    plt.subplot(num_pics, 6, i)
    for j in range(num_pics):
        pic = Image.open(sample_path + listdir(sample_path)[j])
        plt.subplot(5, 6, i + j)
        if Yhat[:, j] == 1:
            plt.title('Dog', c = 'r')
        else:
            plt.title('Monument', c = 'r')
        plt.tick_params(axis='both', which='both', labelleft = False, labelbottom = False)
        plt.imshow(pic)
        
    return Yhat

# 15 ________________________________________________________________________________________________________________________________________________________________

def random_image_check(num_images, set_x, set_y):
    '''
    Given set_x and its labels set_y, outputed by 'merge_shuffle_split' function, and a number for images to be checked, the function displays each with its label
    to be examined.
    '''
    num_images = 5 # number of images to check.
    plt.figure(figsize = (15, 15)) # setting the size of the figure to dispaly the images.
    i = 1
    plt.subplot(10, num_images, i)
    for j in range(num_images):
        random_test_image = np.random.randint(0, set_x.shape[0])
        plt.subplot(5, num_images, i + j)    
        plt.title(str(int(np.squeeze(set_y[:, random_test_image]))), c = 'r')
#         plt.tick_params(axis='both', which='both', labelleft = False, labelbottom = False) # uncomment to remove plot ticks x and y axses and white background.
        plt.imshow(set_x[random_test_image])
        plt.tight_layout()
    
# 16 ________________________________________________________________________________________________________________________________________________________________

def deep_nn_model_exp(train_set_x, train_set_y, test_set_x, test_set_y, mini_batch_size = 128,
                      layer_structures = [[1]], epochs_range = (1000, 3000), epochs_sets = 1, alpha_range = (0.001, 0.005), alpha_sets = 1,
                      lambd = 0.0, dropout_layers = [], keep_prob = 1.0, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8,
                      print_cost = True, print_every = 500, show_plots = True, seed = 0):
    '''
    The function performs iterative application of the 'deep_nn_model' funciton over the number of given epochs, for every given structure, for every given alpha
    and returns a list of the resulted models where each contains full information about the model parameters and hayperparameters...etc. For full details on the
    return ifo, refer to the output of 'deep_nn_model' function.
    
    Arguments:
        train_set_x: Features set to be used for training, outputed by 'prepare_image_arrays' function.
        train_set_y: Labels of 'train_set_x'.
        test_set_x: Same as 'train_set_x' for testing.
        test_set_y: Labels of 'test_set_x'.
        layer_structures: A list of lists of intergers such that each is one model structure with 'len()' equaling the number of hidden layers in the model,
                            and each element being the number of neurons for layer it is indexing. Last element must be 1 as it is for the output layer.
        epochs_range: A tuple that takes two elements, Min and Max, determining the interval to be divided into 'epochs_sets' using 'numpy.linspace(Min, Max, num = epochs_sets)'.
        epochs_sets: An integer specifying the number of differnt epochs to train the models on.
        alpha_range: A tuple that takes two elements, Min and Max, determining the interval to be divided into 'alpha_sets' using 'numpy.linspace(Min, Max, num = alpha_sets)'.
        lambd: A float from 0 to 1 inclusive determining the lambda parameter of the L2 frobenius norm to regularize the wights parameters.
        dropout_layers: A list of intergers numbering the layers to be masked (have thier neurons shut with 1 - keep_prob).
        keep_prob: The probability of keeping a node active in the masked layers.       
        alpha_sets: An integer specifying the number of differnt alphas (learning rates) to train the models with.
        print_cost: A boolean, True to print the cost and train accuracy.
        print_every: An interger specifying after how many epochs the cost and train accuracy to be printed.
        show_plots: A boolean, True to print the cost and train accuracy.
        
    Returns:
        model_summary: A dictionary with varoius model information, check 'deep_nn_funciton' output for details.        
    '''
    layer_structures = layer_structures
    num_iterations_list = list(np.linspace(epochs_range[0], epochs_range[1], num = epochs_sets))
    learning_rates_list = list(np.linspace(alpha_range[0], alpha_range[1], num = alpha_sets))

    models_list = list()
    count = 1
    np.random.seed(seed)
    for iteration in num_iterations_list:
        for alpha in learning_rates_list:
            for structure in layer_structures:
                print(count, 'of', len(num_iterations_list) * len(learning_rates_list) * len(layer_structures), '-' * 50, datetime.now())
                model = deep_nn_model(train_set_x, train_set_y, test_set_x, test_set_y, mini_batch_size = mini_batch_size,
                                      layer_structure = structure, iterations = int(iteration), alpha = alpha.round(6),
                                      lambd = lambd, dropout_layers = dropout_layers, keep_prob = keep_prob, beta1 = beta1, beta2 = beta2, epsilon = epsilon, 
                                      print_cost = print_cost, print_every = print_every, show_plots = show_plots, seed = seed)
                
                models_list.append(model)
                count += 1
                
    return models_list
# 17 ________________________________________________________________________________________________________________________________________________________________


# 18 ________________________________________________________________________________________________________________________________________________________________


# 19 ________________________________________________________________________________________________________________________________________________________________


# 20 ________________________________________________________________________________________________________________________________________________________________


# 21 ________________________________________________________________________________________________________________________________________________________________


# 22 ________________________________________________________________________________________________________________________________________________________________


# 23 ________________________________________________________________________________________________________________________________________________________________


# 24 ________________________________________________________________________________________________________________________________________________________________


# 25 ________________________________________________________________________________________________________________________________________________________________


# 26 ________________________________________________________________________________________________________________________________________________________________


# 27 ________________________________________________________________________________________________________________________________________________________________
