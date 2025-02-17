"""
ML_in_Finance_assignment2.py

Purpose:
   Use a machine learning model to make predictions on either a continuous outcome
   variable (set purpose to "regression", i.e. purposes[0]), a binary outcome
   variable (set purpose to "binary classification", i.e. purposes[1]) or a
   categorical variable (set purpose to "multilabel classification", i.e. 
   purposes[2]). Perform (automated) grid search to optimize hyperparameters, 
   and get a rough estimate of the completion time before executing grid search.
   
Version:
    1       Model constuction, fitting and predicting.
    2       Improved model architecture and added bce_tester function. 
    3       Added: 
                - grid search completion time estimation function
                - grid search function
                - classification evaluation function
                - cross validation option
            Next up: 
                - making target predictions and saving to .txt file
                
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
import os
import pathlib as Path

from sklearn.model_selection import KFold
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

import sklearn.metrics as skmetrics
import seaborn as sns
import matplotlib.pyplot as plt

# check versions
# print(tf.__version__)               # 2.1.0
# print(tf.keras.__version__)         # 2.2.4-tf


#%%             Get data
###########################################################
### df_train, dfX_target = get_data(train_path, target_path, rseed)
def get_data(train_path, target_path, rseed):
    """
    Purpose:
        Extract the train and target datasets. Standardize (transform each variable
        s.t. it has mean = 0 and unit variance) the feature space of the training 
        data, and scale dfX of the target data using the same parameters.
        
    Inputs:
        train_path              Path to the training dataset.
        target_path             Path to the prediction target dataset.
        rseed                   Randomization seed.

    Return values:
        df_train                Dataframe with the train data (feature space + outcome var).
        dfX_target              Dataframe with the taget data (feature space).
                                
    """
    scalar = StandardScaler()                           # Initialize standardizer.
    
    df_train = pd.read_csv(train_path)
    
    dfX_train = df_train.drop(['target'],axis=1)
    dfX_train = scalar.fit_transform(dfX_train)         # Standardizing and saving parameters.
    
    
    dfX_target = pd.read_csv(target_path)               # Test data has no target variable.
    dfX_target = scalar.transform(dfX_target)           # Standardizing using saved parameters.
    
    return df_train, dfX_target

#%%             Cross validation sets
###########################################################
### CV_sets = k_fold_CV_sets(df_train, k_folds, rseed)
def k_fold_CV_sets(df_train, k_folds, rseed):
    """
    Purpose:
        Split the data into k-folds for cross validation.
        
    Inputs:
        df_train                Dataframe containing the training data, 
                                including outcome variable.
        k_folds                 Number of cross validation sets to split the data into.
        rseed                   Randomization seed.

    Return value:
        CV_sets                 List of dictionaries, where each element in the
                                list corresponds to one of the k-folds, and each 
                                dictionary contains a train and test key, where 
                                each key corresponds to a dictionary with keys 
                                'dfX' with the feature space and 'vY' with the 
                                outcome variable.
    """
    scalar = StandardScaler()                               # Initialize standardizer.
    
    # Define the number of splits
    kf = KFold(n_splits = k_folds, shuffle = True, random_state = rseed)
    
    CV_sets = []
    
    # Split the data into cross-validation sets
    for train_index, test_index in kf.split(df_train):
        
        train_set = df_train.iloc[train_index]
        vY_train  = train_set['target']
        vY_train = np.asarray(vY_train, dtype="float32")
        dfX_train = train_set.drop(['target'],axis=1)
        dfX_train = scalar.fit_transform(dfX_train)         # Standardizing and saving parameters.
        
        test_set = df_train.iloc[test_index]
        vY_test  = test_set['target']
        vY_test = np.asarray(vY_test, dtype="float32")
        dfX_test = test_set.drop(['target'],axis=1)
        dfX_test = scalar.transform(dfX_test)               # Standardizing using saved parameters.
        
        
        CV_sets += [{'train' : {'dfX' : dfX_train, 'vY' : vY_train}, 
                     'test' : {'dfX' : dfX_test, 'vY' : vY_test}}]
    
    return CV_sets
    

#%%             Constructing neural network
###########################################################
### model = Model_architect(purpose, dfX_train, n_layers, activations, n_nodes, rseed, Gauss_noise, dropout_rate)
def Model_architect(purpose, dfX_train, n_layers, activations, n_nodes, rseed, Gauss_noise, dropout_rate):
    """
    Purpose:
        Create the deep learning architecture.
        
    Inputs:
        purpose                 Purpose of the algorithm.
        dfX_train               Feature space of the training data.
        n_layers                Number of layers of the neural network.
        activations             Set of activation functions for each hidden layer.
        n_nodes                 The number of nodes of each hidden layer.
        rseed                   Randomization seed.
        Gauss_noise             Standard deviation of the Gaussian noise added 
                                to the features.
        dropout_rate            Dropout rate regularization.

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
    
    """                 Note on number of nodes
    Note that only using powers of 2 for the number of nodes per layer makes 
    most efficient use of processor architecture, so this is best for computation 
    time. Depending on the processor (no idea how you can find this out) best 
    to have max n_nodes at most 64 or 128.
    """
    
    # Add first hidden layer to the network.
    model.add(Dense(n_nodes[0], activation = activations[0], input_shape=(dfX_train.shape[1],)))
    
    # Add regularization methods.
    model.add(GaussianNoise(Gauss_noise, seed=rseed))
    model.add(Dropout(dropout_rate))
    
    # Add n_layers - 1 more hidden layers to the neural network.
    if n_layers > 1:
        model.add(Dense(n_nodes[1], activation = activations[1]))
    if n_layers > 2:
        model.add(Dense(n_nodes[2], activation = activations[2]))
    if n_layers > 3:
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
        length = 50
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
### grid_search_length_estimator(t_epoch, gridsearch_hyp_sets, n_CV_sets)    :
def grid_search_length_estimator(t_epoch, gridsearch_hyp_sets, n_CV_sets)    :
    
    """
    Purpose:
        Calculate the total length of the hyperparameter grid search, which will
        evaluate all combinations of hyperparameter values in hyperparameter_sets.
        
        We need to multiply each element of n_epochs by t_epoch and combinations 
        of hyperparameters per element of n_epochs.
        
    Inputs:
        t_epoch                 Scalar showing estimated upper bound of time to 
                                complete one epoch.
        gridsearch_hyp_sets     Dictionary containing the hyperparameters 
                                as keys and lists of values as values.
        n_CV_sets               Number of cross validation sets.
                                
        
    Return value:
        None (void).            Prints completion time.
    
    """
    
    # Find number of combinations of hyperparameters per element of n_epochs.
    N_combs = len([i for i in product(*gridsearch_hyp_sets.values())])
    non_epoch_sets = {key: value for key, value in gridsearch_hyp_sets.items() if key != 'n_epochs'}
    non_epoch_combinations = product(*non_epoch_sets.values())
    combinations_per_n_epochs   = [i for i in non_epoch_combinations]
    n_combs_per_n_epochs        = len(combinations_per_n_epochs)   
    
    completion_time = 0
    for n_epochs in gridsearch_hyp_sets['n_epochs']:
        time_per_n_epochs = n_epochs * t_epoch * n_combs_per_n_epochs
        completion_time += time_per_n_epochs
    
    completion_time = completion_time * n_CV_sets
    completion_time_hours     = round(np.floor(completion_time/3600))
    completion_time_minutes   = round((completion_time/60) % 60)
    print(f"\nEstimated upper bound on grid search completion time for {N_combs} hyper parametersets = ")
    print(f"{round(completion_time, 1)} seconds, or {completion_time_hours} hour(s) and {completion_time_minutes} minutes.")

#%%             Plotting confusion matrix
###########################################################
### plot_confusion_matrix(conf_matrix, conf_matrix_path)
def plot_confusion_matrix(conf_matrix, conf_matrix_path):
    """
    Purpose:
        Plot the confusion table as a heatmap.
        
    Inputs:
        conf_matrix             Confusion matrix
        conf_matrix_path        path to where the plot will be stored

    Return value:
        None ()
    """

    # Plot the confusion matrix using seaborn
    plt.figure(figsize=(8, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    
    plt.savefig(conf_matrix_path, dpi = 900)


#%%             Evaluating binary classification
###########################################################
### eval_scores = eval_binary_classification(vY_true, vY_pred, eval_metrics, verbose = 0)
def eval_binary_classification(vY_true, vY_pred, bin_eval_metrics, verbose = 0):
    """
    Purpose:
        Evaluate the binary classification on the metrics, specified in bin_eval_metrics.
        
    Inputs:
        vY_test                 Vector with the true values of the binary
                                outcome variable.
        vY_test_pred            Vector with the predicted values of the binary
                                outcome variable.
        bin_eval_metrics        List of binary evaluation metrics to use.

    Return value:
        eval_scores             Dictionary containing evaluation metrics as keys
                                and their corresponding scores as values.
    """    
    
    eval_metric_fns = {metric : getattr(skmetrics, metric + "_score") for metric in bin_eval_metrics}
    eval_scores     = {metric : None for metric in bin_eval_metrics}
    
    for metric in bin_eval_metrics:
        eval_scores[metric] = eval_metric_fns[metric](vY_true, vY_pred)
        if verbose == 1:
            print(f"{metric} = {eval_scores[metric]:.3f}")
    
    # Print classification report
    # classification_rep = skmetrics.classification_report(vY_test, vY_test_pred, target_names=['Class 0', 'Class 1'])
    # print('Classification Report:\n', classification_rep)
    
    return eval_scores


#%%             Binary predictor
###########################################################
### vY_pred = bin_pred(purpose, model, dfX)
def bin_pred(purpose, model, dfX):
    """
    Purpose:
        Predicting binary outcome variable using the fitted model.
        
    Inputs:
        model                   The model fitted to training data.
        dfX_test                The feature space of the prediction target.

    Return value:
        vY_pred                 Vector of predicted outcome variable.
    """
    
    if purpose != "binary classification":
        return print("Trying to use binary prediction on non-binary outcome variable.")
    
    else:
        vY_pred = model.predict(dfX)
        vY_pred = (vY_pred > 0.5).astype(int)        # Converting sigmoid output to binary predictions.
        
        return vY_pred


#%%             Grid search
###########################################################
### df_hypset_eval_scores, df_best_scores = Grid_search(purpose, hyperparameters, gridsearch_hyp_sets, bin_eval_metrics, rseed, CV_sets, n_CV_sets)
def Grid_search(purpose, hyperparameters, gridsearch_hyp_sets, bin_eval_metrics, rseed, CV_sets, n_CV_sets):
    """
    Purpose:
        Assess the model's accuracy for various configurations of hyperparameters
        by taking the mean accuracy metrics of a specified number of cross 
        validation sets.
        
    Inputs:
        purpose                 Purpose of the algorithm.
        hyperparameters         Dictionary containing hyperparameters.
        gridsearch_hyp_sets     Dictionary containing the hyperparameters 
                                that are to be evaluated in the grid search as 
                                keys and lists of values that are to be tested
                                as values.
        bin_eval_metrics        List of binary evaluation metrics.
        rseed                   Randomization seed.
        CV_sets                 List of cross validation sets.
        n_CV_sets               Number of cross validation sets to evaluate on.

    Return value:
        df                      Dataframe showing the validation error rates
                                of various configurations of the hyperparameters
                                of the model.
    """
    start = t()
    
    non_gridsearch_hyperparameters = {key: value for key, value in hyperparameters.items() if key not in gridsearch_hyp_sets}
    hyperparameter_combinations = [i for i in product(*gridsearch_hyp_sets.values())]        
    
    hypset_eval_scores = []    
    gshs_keys = [i for i in gridsearch_hyp_sets.keys()]
    
    N_hyp_sets = len(hyperparameter_combinations)
    print(f"\nStarting evaluation of {N_hyp_sets} hyperparameter sets, with a total of {N_hyp_sets * n_CV_sets} iterations.")
    
    # CV_set = 0
    # hyp_set = hyperparameter_combinations[0]
    j = 0
    for hyp_set in hyperparameter_combinations:
        j += 1
        print(f"\nHyperparameter set: {j}")
        # Reconstructing hyperparameter set.
        # grid_search_hyp_set = {key: hyp for key, hyp in zip(gridsearch_hyp_sets, hyp_set)} # This line is the same as below
        grid_search_hyp_set = dict(zip(gshs_keys, hyp_set))
        hyps = {**non_gridsearch_hyperparameters, **grid_search_hyp_set}
        accumulated_scores = {}
        
        for CV_set in range(n_CV_sets):
            
            dfX_train, vY_train = CV_sets[CV_set]['train']['dfX'], CV_sets[CV_set]['train']['vY']
            dfX_test, vY_test   = CV_sets[CV_set]['test']['dfX'], CV_sets[CV_set]['test']['vY']
            
            # Defining model with specified hyperparameters and fitting it to the train data.
            model = Model_architect(purpose, dfX_train, hyps['n_layers'], 
                                    hyps['activations'], hyps['n_nodes'], rseed, 
                                    hyps['Gauss_noise'], hyps['dropout_rate'])
            model.fit(dfX_train, vY_train, epochs = hyps['n_epochs'], batch_size = hyps['batch_size'], verbose = 0)
            
            # Predicting outcome variable and evaluating prediction.
            if purpose == 'binary classification':
                vY_pred = bin_pred(purpose, model, dfX_test)
                
                temp_eval_scores = eval_binary_classification(vY_test, vY_pred, bin_eval_metrics, verbose = 0)
                for key, value in temp_eval_scores.items():
                    if key not in accumulated_scores:
                        accumulated_scores[key] = [value]
                    else:
                        accumulated_scores[key].append(value)
                        
            elif purpose == 'regression':
                return print("Grid search for non-binary prediction is not finished yet.")
            elif purpose == 'multilabel classification':
                return print("Grid search for non-binary prediction is not finished yet.")
    
        mean_scores = {key: sum(values) / len(values) for key, values in accumulated_scores.items()}
        eval_scores = {**hyps, **mean_scores}
        hypset_eval_scores.append(eval_scores)
    
    df_hypset_eval_scores = pd.DataFrame(hypset_eval_scores)
    
    best_scores = {}
    for score in temp_eval_scores.keys():
        best_score_index = df_hypset_eval_scores[score].idxmax()
        best_score = {score : df_hypset_eval_scores.loc[best_score_index].to_dict()}
        best_scores.update(best_score)
    df_best_scores = pd.DataFrame.from_dict(best_scores, orient='index')
    
    grid_search_seconds = t() - start
    grid_search_minutes   = round((grid_search_seconds/60) % 60)
    grid_search_hours     = round(np.floor(grid_search_seconds/3600))
    print(f"\n\nThe Grid search took {round(grid_search_seconds, 1)} seconds, or {grid_search_hours} hours {grid_search_minutes} minutes.\n\n")
    
    return df_hypset_eval_scores, df_best_scores


#%%             Time per epoch tester
###########################################################
### t_epoch = t_epoch_tester(CV_sets, hyperparameters, purpose, rseed)
def t_epoch_tester(CV_sets, hyperparameters, purpose, rseed):
    """
    Purpose:
        For now, this is just storage of incomplete code in a function such that 
        it does not run when the script is executed. 
        
        But this should become a function with which to test run fitting the 
        model to one CV train data set, with the purpose of sampling average
        time to complete one epoch (i.e. iteration) of the gradient descent on the
        loss function.
        
        
    Inputs:
        XX                      ...

    Return value:
        RetValue                ...
    """
    dfX_train, vY_train = CV_sets[0]['train']['dfX'], CV_sets[0]['train']['vY']
    dfX_test, vY_test   = CV_sets[0]['test']['dfX'], CV_sets[0]['test']['vY']
    
    
    # Fit model to train data.
    model = Model_architect(purpose, dfX_train, hyperparameters['n_layers'], 
                            hyperparameters['activations'], hyperparameters['n_nodes'], 
                            rseed, hyperparameters['Gauss_noise'], 
                            hyperparameters['dropout_rate'])
    # Provide summary of model architecture.
    model.summary()
    
    # Fit the model to 
    start = t()
    model.fit(dfX_train, vY_train, epochs = hyperparameters['n_epochs'], 
              batch_size = hyperparameters['batch_size'])
    
    training_time_seconds = t() - start
    training_time_minutes   = round((training_time_seconds/60) % 60)
    training_time_hours     = round(np.floor(training_time_seconds/3600))
    print(f"Training the model took {round(training_time_seconds, 1)} seconds, or {training_time_hours} hours {training_time_minutes} minutes.")
    print(f"\nTime per epoch: {round(training_time_seconds/hyperparameters['n_epochs'],5)} seconds.")    
    
    # Completion time of one epoch when n_layers = 4, which is the most complex NN
    # that is to be evaluated. So t_epoch is assumed to be an upper bound.
    t_epoch = training_time_seconds/hyperparameters['n_epochs']
    
    
    
    ################
    # Validation
    # vY_test_pred = bin_pred(purpose, model, dfX_test)
    
        
    # Create the confusion matrix
    # conf_matrix = skmetrics.confusion_matrix(vY_test, vY_test_pred)
    
    ################
    # Output    
    
    
    # vY_target_pred = model.predict(dfX_target)
    # if purpose == "binary classification":
    #     vY_target_pred = (vY_target_pred > 0.5).astype(int)    # Converting sigmoid output to binary predictions.
    
    
    # # Save the predictions to a csv file.
    # vY_target_pred = pd.DataFrame(vY_target_pred)
    # vY_target_pred.to_csv(pred_path)
    
    # plot_confusion_matrix(conf_matrix, conf_matrix_path)
    
    return t_epoch

#%%             Main
###########################################################
### main
def main():
    
    ################
    # Magic numbers    
    rseed = 98765               # Set randomization seed.
    
    # Input paths
    base_dir = Path.cwd().parent.parent     # This assumes this script is in a directory "Code" under the base_dir.
    train_path      = os.path.join(base_dir, "Input/train.csv")
    target_path     = os.path.join(base_dir, "Input/test.csv")
    
    # Output paths
    hypset_eval_scores_path = os.path.join(base_dir, "Output/hypset evaluations.csv") 
    best_scores_path        = os.path.join(base_dir, "Output/best evaluation scores.csv") 
    # pred_path       = "C:/Users/jlhde/Documents/BDS/ML in Finance/Assignment/Output/predictions.csv"
    # conf_matrix_path= "C:/Users/jlhde/Documents/BDS/ML in Finance/Assignment/Output/confusion matrix.pdf"
    
    # State the purpose of the neural network.
    purposes = ('regression', 'binary classification', 'multilabel classification')
    purpose = purposes[1]       

    
    hyperparameters =     {'n_epochs'       : 250,
                           'n_layers'       : 2,
                           'dropout_rate'   : 0.3,
                           'Gauss_noise'    : 0.5,         # Due to standardizing each variable has a standard deviation of 1.
                           'n_nodes'        : [32 ,16, 8, 4],
                           'batch_size'     : 64,
                           'activations'    : ['relu','relu','relu','relu'],
                           }


    # Hyperparameter options to be evaluated in grid search.
    gridsearch_hyp_sets = {
                            'n_epochs'       : [2000],
                            'n_layers'       : [1, 2],
                            # 'dropout_rate'   : [.1, .3],
                            # 'Gauss_noise'    : [.1,.5],
                            # 'batch_size'     : [32, 64, 128]
                           }
    
    
    
    # Set number of folds for cross validation.
    k_folds = 5
    
    # Number of k-fold cross validation sets to use for evaluating grid search.
    n_CV_sets = 3
    
    # Define the binary evaluation metrics to be used for evaluation.
    bin_eval_metrics    = ['accuracy', 'precision', 'recall', 'f1']
    
    ################
    # Initialisation
   
    df_train, dfX_target = get_data(train_path, target_path, rseed)
    CV_sets = k_fold_CV_sets(df_train, k_folds, rseed)
    
    
    
    # Reset the global state of Keras.
    K.clear_session()
    
    tf.random.set_seed(rseed)
    
    # Estimate completion time of grid search.
    # t_epoch = t_epoch_tester(CV_sets, hyperparameters, purpose, rseed)
    t_epoch = 0.4183832693099976
    grid_search_length_estimator(t_epoch, gridsearch_hyp_sets, n_CV_sets)  
    
    
    # Checking how many times each observation will be used to update the model parameters.
    n_obs = len(CV_sets[0]['train']['vY'])
    epochs_per_obs = (hyperparameters['n_epochs']* hyperparameters['batch_size'])/ n_obs
    print(f"\nOn average, each observation will be evaluated in the loss function {round(epochs_per_obs, 3)} times.")
    
    # Testing the binary cross entropy loss function.
    # bce_mean, failrate_scores = bce_test()
    # print(f"\n\nbce_mean = {bce_mean}\n")
    # print(failrate_scores)
    
    
    
    
    ################
    # Grid search    
    
    df_hypset_eval_scores, df_best_scores = Grid_search(purpose, hyperparameters, gridsearch_hyp_sets, bin_eval_metrics, rseed, CV_sets, n_CV_sets)
    
    df_hypset_eval_scores.to_csv(hypset_eval_scores_path)
    df_best_scores.to_csv(best_scores_path)
    
    
    
#%%             Start main
###########################################################
### start main
if __name__ == '__main__':          
    main()






























