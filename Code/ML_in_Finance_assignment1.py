"""
ML_in_Finance_assignment1.py

Purpose:
   Use a machine learning model to make predictions on either a continuous outcome
   variable (set purpose to "regression", i.e. purposes[0]), a binary outcome
   variable (set purpose to "binary classification", i.e. purposes[1]) or a
   categorical variable (set purpose to "multilabel classification", i.e. 
   purposes[2]).
   
Version:
    1       Model constuction, fitting and predicting.
    2       Improved model architecture and added bce_tester function. 
    
            Next up: 
                - grid search function
                - cross validation option
                - classification evaluation function
                
            Later:
                - evaluation functions for regression and multilabel classification

Date:
    2023/10/10

Author:
    Julius de Clercq
"""


###########################################################
### Imports
import numpy as np
import pandas as pd
from itertools import product
from time import time as t 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import CategoricalCrossentropy
import tensorflow.keras.backend as K

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# check versions
# print(tf.__version__)               # 2.1.0
# print(tf.keras.__version__)         # 2.2.4-tf


#%%             Get data
###########################################################
### dfX_train, dfX_test, vY_train, vY_test, dfX_target = get_data(train_path, target_path, val_size, rseed)
def get_data(train_path, target_path, val_size, rseed):
    """
    Purpose:
        Extract the train and test datasets. Separate target variable (vY) from 
        feature space (dfX) if applicable. Also standardize (transform each variable
        s.t. it has mean = 0 and unit variance) the feature space of the training 
        data, and scale dfX of the test data using the same parameters. Separate
        a fraction (i.e. val_size) of the training data as validation set (i.e. test set).
        
    Inputs:
        train_path              Path to the training dataset.
        target_path             Path to the prediction target dataset.
        val_size                Fraction of train data to be used for validation.
        rseed                   Randomization seed.

    Return values:
        vY_train, dfX_train,    Dataframes with the train and test datasets, 
        vY_test, dfX_train      split into the vector of the outcome variable
                                (vY) and the feature space (dfX).  
                                
    """
    scalar = StandardScaler()                           # Initialize standardizer.
    
    df_train = pd.read_csv(train_path)
    
    vY_train  = df_train['target']
    vY_train = np.asarray(vY_train, dtype="float32")    
    
    dfX_train = df_train.drop(['target'],axis=1)
    dfX_train = scalar.fit_transform(dfX_train)         # Standardizing and saving parameters.
    
    # Saving 
    dfX_train, dfX_test, vY_train, vY_test = train_test_split(dfX_train, vY_train, 
                                            test_size = val_size, random_state = rseed)
    
    dfX_target = pd.read_csv(target_path)               # Test data has no target variable.
    dfX_target = scalar.transform(dfX_target)           # Standardizing using saved parameters.
    
    return dfX_train, dfX_test, vY_train, vY_test, dfX_target


#%%             Constructing neural network
###########################################################
### model = Model_architect(purpose, dfX_train, n_layers, activations, n_nodes, rseed, Gauss_noise, dropout_rate)
def Model_architect(purpose, dfX_train, n_layers, activations, n_nodes, rseed, Gauss_noise, dropout_rate):
    """
    Purpose:
        Create the deep learning architecture.
        
    Inputs:
        purpose                 Purpose of the algorithm
        dfX_train               Feature space of the training data
        n_layers                Number of layers of the neural network
        activations             Set of activation functions for each hidden layer
        n_nodes                 Hyperparameter defining the number of nodes of
                                each layer in the network
        rseed                   Randomization seed
        Gauss_noise             Hyperparameter defining standard deviation of the 
                                Gaussian noise added to the features
        dropout_rate            Hyperparameter of dropout regularization

    Return value:
        model                   model architecture that is to be fitted to the 
                                training data
    """
    if purpose == 'regression':
        final_activation = 'linear'
        loss_function = MeanSquaredLogarithmicError
    elif purpose == 'binary classification':
        final_activation = 'sigmoid'
        loss_function = BinaryCrossentropy
    elif purpose == 'multilabel classification':
        final_activation = 'softmax'
        loss_function = CategoricalCrossentropy
    else:
        return print(f"Invalid purpose: {purpose}")
    
    # Define sequential model structure and the input layer of the model.
    model = Sequential()
    model.add(Dense(n_nodes[0], activation = activations[0], input_shape=(dfX_train.shape[1],)))
    
    # Add regularization methods.
    model.add(GaussianNoise(Gauss_noise, seed=rseed))
    model.add(Dropout(dropout_rate))
    
    # Add depth to the network according to 
    if n_layers>1:
        model.add(Dense(n_nodes[1], activation = activations[1]))
    if n_layers>2:
        model.add(Dense(n_nodes[2], activation = activations[2]))
    if n_layers>3:
        model.add(Dense(n_nodes[3], activation = activations[3]))
    model.add(Dense(1, activation = final_activation))
    model.compile(optimizer="adam", loss= loss_function()) 
    
    return model

#%%             Binary cross entropy tester
###########################################################
### bce_mean, failrate_scores = bce_test()
def bce_test():
    """
    Purpose:
        Test the binary cross entropy (bce) loss function, by checking its 
        unconditional mean and its values for one mismatch in binary vectors of 
        length 1, 2, 5, 10, 20, 30, 50 and 100. The purpose of this is to get a 
        sense of how to interpret the values returned by the bce function.
        
    Inputs:
        N.A.                    

    Return value:
        bce_mean                Unconditional mean of the bce score, which is 
                                the mean score of comparing uncorrelated vectors
        failrate_scores         Dataframe with scores corresponding to fail rates.
        
    """
        
        
    bce = BinaryCrossentropy()
    
    rand_values = []
    for i in range(1000):
        length = 10000
        y_1 = np.asarray(np.random.randint(2, size=length), dtype="float32")
        y_2 = np.asarray(np.random.randint(2, size=length), dtype="float32")
        rand_values += [bce(y_1,y_2).numpy()]
    
    bce_mean = np.mean(rand_values)
    
    failrate_scores = []
    fail_rates = [1, 2, 5, 10, 20, 30, 50, 100]       
    for fail_rate in fail_rates:
        length = int(fail_rate)
        y_1 = np.asarray(np.random.randint(2, size=length), dtype="float32")
        y_2 = y_1 .copy()
        y_2[0] = 1.0 - y_1[0]
        failrate_scores += [{"Fail rate": f"1/{fail_rate}", "bce": bce(y_1, y_2).numpy()}]
    failrate_scores = pd.DataFrame(failrate_scores)
    
    
    return bce_mean, failrate_scores


#%%             Hyperparameter grid search completion time estimator
###########################################################
### completion_time = grid_search_length_estimator(t_epoch, hyperparameter_sets):
def grid_search_length_estimator(t_epoch, hyperparameter_sets):
    
    """
    Purpose:
        Calculate the total length of the hyperparameter grid search, which will
        evaluate all combinations of hyperparameter values in hyperparameter_sets.
        
        We need to multiply each element of n_epochs by t_epoch and combinations 
        of hyperparameters per element of n_epochs.
        
    Inputs:
        t_epoch                         scalar, time to complete one epoch upper bound
        hyperparameter_sets             dictionary containing the hyperparameters 
                                        as keys and lists of values as values
                                        
    Return value:
        completion_time                 Estimated upper bound for completing the 
                                        grid search.
    
    """
    
    # 
    
    # Find number of combinations of hyperparameters per element of n_epochs.
    non_epoch_sets = {key: value for key, value in hyperparameter_sets.items() if key != 'n_epochs'}
    hyperparameter_combinations = product(*non_epoch_sets.values())
    combinations_per_n_epochs   = [i for i in hyperparameter_combinations]
    n_combs_per_n_epochs        = len(combinations_per_n_epochs)   
    
    completion_time = 0
    for n_epochs in hyperparameter_sets['n_epochs']:
        time_per_n_epochs = n_epochs * t_epoch * n_combs_per_n_epochs
        completion_time += time_per_n_epochs
    
    return completion_time

#%%             Grid search
###########################################################
### routine
def Grid_search():
    """
    Purpose:
        Assess the model's accuracy for various configurations of hyperparameters.
        
    Inputs:
        XX                      ...

    Return value:
        df                      Dataframe showing the validation error rates
                                of various configurations of the hyperparameters
                                of the model.
    """
    
    df = pd.DataFrame()
    
    
    
    return df

#%%             Main
###########################################################
### main
def main():
    
    ################
    # Magic numbers    
    rseed = 98765               # Set randomization seed.
    val_size = 0.1              # Set fractional size of validation set.
    
    # Set paths to train and target data. And the path where to store the target predictions.
    train_path = "C:/Users/jlhde/Documents/BDS/ML in Finance/Assignment/train.csv"
    target_path  = "C:/Users/jlhde/Documents/BDS/ML in Finance/Assignment/test.csv"
    pred_path =  "C:/Users/jlhde/Documents/BDS/ML in Finance/Assignment/predictions.csv"
    
    purposes = ('regression', 'binary classification', 'multilabel classification')
    purpose = purposes[1]       # State the purpose of the neural network.

    # Define activation functions for hidden layers
    activations = ['relu','relu','relu','relu']

    # Define the number of nodes of each hidden layer
    # Note that powers of 2 make most efficient use of processor architecture,
    # best not to have max n_nodes to be either 32 or 64.
    n_nodes = [32,16,8,4]

    # Define the number of layers we will use (use any integer between 1 and 4)
    n_layers = 4
    # Number of epochs
    n_epochs = 100
    
    # batch size
    batch_size = 64
    # batch_size = math.floor(len(dfX_train)/8)
    
    # Regularization hyperparameters.
    Gauss_noise = 0.1
    dropout_rate = .15
    
    
    
    # Completion time of one epoch when n_layers = 4, which is the most complex NN
    # that is to be evaluated. So t_epoch is assumed to be an upper bound.
    t_epoch = 0.358


    # Hyperparameter options
    hyperparameter_sets = {'n_epochs'       : [100, 200, 300],     # , 750, 1000
                           'n_layers'       : [2, 3, 4],
                           'dropout_rates'  : [.1, .2, .3],
                           'Gauss_noise'    : [.1, .2, .25]
                           }
    
    ################
    # Initialisation
   
    dfX_train, dfX_test, vY_train, vY_test, dfX_target = get_data(train_path, target_path, val_size, rseed)
    
    # Reset the global state of Keras.
    K.clear_session()
    
    tf.random.set_seed(rseed)
    
    # Fit model to train data.
    model = Model_architect(purpose, dfX_train, n_layers, activations, n_nodes, rseed, Gauss_noise, dropout_rate)
    
    ################
    # Estimation
    start=t()
    model.fit(dfX_train, vY_train, epochs=n_epochs, batch_size= batch_size) # , validation_split = 0.01
    training_time = t() - start
    
    print(f"Training the model took {training_time} seconds, or {round(training_time/60,1)} minutes)")
    print(f"Time per epoch: {round(training_time/n_epochs,5)} seconds.")    
    
    # Provide summary of model architecture.
    model.summary()
    
    # Estimate completion time of grid search.
    completion_time = grid_search_length_estimator(t_epoch, hyperparameter_sets)
    
    # Testing the binary cross entropy loss function.
    # bce_mean, failrate_scores = bce_test()
    # print(f"\n\nbce_mean = {bce_mean}\n")
    # print(failrate_scores)
    
    
    ################
    # Validation
    vY_test_pred = model.predict(dfX_test)
    if purpose == "binary classification":
        vY_test_pred = (vY_test_pred > 0.5).astype(int)        # Converting sigmoid output to binary predictions.
    
        
    # Create the confusion matrix
    conf_matrix = confusion_matrix(vY_test, vY_test_pred)
    
    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(8, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Compute additional accuracy metrics.
    
    ######      WORK IN PROGRESS    ######
    # accuracy_scores = ['accuracy', 'precision', 'recall', 'f1']
    # scores = []
    # for score in accuracy_scores:
    #     scores += []
    
    accuracy = accuracy_score(vY_test, vY_test_pred)
    precision = precision_score(vY_test, vY_test_pred)
    recall = recall_score(vY_test, vY_test_pred)
    f1 = f1_score(vY_test, vY_test_pred)

    # Print metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    
    # Print classification report
    classification_rep = classification_report(vY_test, vY_test_pred, target_names=['Class 0', 'Class 1'])
    print('Classification Report:\n', classification_rep)
    

    
    
    ################
    # Output    
    print(f"\nEstimated grid search completion time = {round(completion_time, 2)} seconds, or {round(completion_time/60, 2)} minutes.\n")
    
    vY_target_pred = model.predict(dfX_target)
    if purpose == "binary classification":
        vY_target_pred = (vY_target_pred > 0.5).astype(int)    # Converting sigmoid output to binary predictions.
    
    
    # Save the predictions to a csv file.
    vY_target_pred = pd.DataFrame(vY_target_pred)
    vY_target_pred.to_csv(pred_path)
    

#%%             Start main
###########################################################
### start main
if __name__ == '__main__':          
    main()






























