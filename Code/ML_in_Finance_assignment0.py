"""
ML_in_Finance_assignment0.py

Purpose:
   Use a machine learning model to make predictions on a continuous outcome
   variable
   
Version:
    1       First start

Date:
    2023/10/07

Author:
    Julius de Clercq
"""


###########################################################
### Imports
import numpy as np
import pandas as pd
from time import time as t 

from sklearn.preprocessing import StandardScaler as scalar
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GaussianNoise
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import CategoricalCrossentropy
import tensorflow.keras.backend as K

# check versions
# print(tf.__version__)               # 2.1.0
# print(tf.keras.__version__)         # 2.2.4-tf


#%%             Get data
###########################################################
### vY_train, dfX_train, dfX_test = get_data(train_path, test_path)
def get_data(train_path, test_path):
    """
    Purpose:
        Extract the train and test datasets. Separate target variable from 
        feature space.
        
    Inputs:
        train_path              path to the training dataset
        test_path               path to the test dataset

    Return values:
        vY_train, dfX_train,    Dataframes with the train and test datasets, 
        vY_test, dfX_train      split into the vector of the outcome variable
                                (vY) and the feature space (dfX).  
                                
    """
    
    df_train = pd.read_csv(train_path)
    
    
    vY_train  = df_train['target']
    vY_train = np.asarray(vY_train, dtype="float32")
    
    dfX_train = df_train.drop(['target'],axis=1)
    
    dfX_test = pd.read_csv(test_path)
    
    return vY_train, dfX_train, dfX_test 


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
        Test the binary cross entropy loss function, by checking its unconditional
        mean, its values for one mistake in 5, 10, 20, 30, 50 and 100.
        
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
    fail_rates = [5, 10, 20, 30, 50, 100]       
    for fail_rate in fail_rates:
        length = int(fail_rate)
        y_1 = np.asarray(np.random.randint(2, size=length), dtype="float32")
        y_2 = y_1 .copy()
        y_2[0] = 1.0 - y_1[0]
        failrate_scores += [{"Fail rate": f"1/{fail_rate}", "bce": bce(y_1, y_2).numpy()}]
    
    
    return bce_mean, failrate_scores



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
    
    # Set randomization seed
    rseed = 98765
    
    # paths to train and test data
    train_path = "C:/Users/jlhde/Documents/BDS/ML in Finance/Assignment/train.csv"
    test_path  = "C:/Users/jlhde/Documents/BDS/ML in Finance/Assignment/test.csv"
    
    
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
    n_epochs = 1000
    
    # batch size
    batch_size = 64
    # batch_size = math.floor(len(dfX_train)/8)
    
    # Regularization hyperparameters.
    Gauss_noise = 0.1
    dropout_rate = .15
    
    
    
    ################
    # Initialisation
   
    vY_train, dfX_train, dfX_test = get_data(train_path, test_path)
    
    # Reset the global state of Keras.
    K.clear_session()
    
    tf.random.set_seed(rseed)
    
    # Fit model to train data
    model = Model_architect(purpose, dfX_train, n_layers, activations, n_nodes, rseed, Gauss_noise, dropout_rate)
    model.summary()
    
    ################
    # Estimation
    start=t()
    model.fit(dfX_train, vY_train, epochs=n_epochs, batch_size= batch_size, validation_split=0.05)
    training_time = round(t()-start,1)
    
    print(f"Training the model took {training_time} seconds, or {round(training_time/60,1)} minutes)")
    print(f"Time per epoch: {round(training_time/n_epochs,1)} seconds.")    
    
    
    # Testing the binary cross entropy loss function.
    bce_mean, failrate_scores = bce_test()
    print(f"bce_mean = {bce_mean}")
    print(failrate_scores)
    
    ################
    # Output
    
    # Predict test data
    y_pred = model.predict(dfX_test)
    print(type(y_pred))
    
    

#%%             Start main
###########################################################
### start main
if __name__ == '__main__':          
    main()






























