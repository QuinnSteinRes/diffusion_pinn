#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing modules
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Make numpy values easier to read.
np.set_printoptions(precision=5, suppress=True)

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
#_AO from tensorflow.python.keras.layers import Dense, BatchNormalization, Dropout, Activation
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Activation
import os
import os.path
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Input
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
import matplotlib as mpl
import matplotlib
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.plots import plot_objective
from skopt.utils import use_named_args
from sklearn.model_selection import train_test_split
import OpenMORe.model_order_reduction as model_order_reduction
from OpenMORe.utilities import *
import tensorflow as tf

#################################################################################
# Dictionary with the instruction for the algorithm:                            #
settings ={ 

    ##### CASE ######                                    						#
    "inputFile"                 : "CFDAll.datAlpp",                             #         
    "resultFolder"          	: "trainingResults",                            #
                                                                                #
    ##### DATA PREPROCESSING SETTINGS ######                                    #
    # Centering and scaling options (string/string)                             #
    "centering_method"          : "mean",                                       #
    "scaling_method"            : "auto",                                       #
    # Percentage of input data to be used for training (int)                    #
    "training_ratio"            : 70, #%                                        #
                                                                                #
    ##### ANN AND OPTIMIZER SETTINGS #####                                      #
    # Number of epochs for the ANN (int)                                        #
    "network_epochs"            : 500,                                          #
    # Number of iterations for the optimizer (int)                              #
    "iterations_optimizer"      : 30,                                           #
    # Acquisition function to be utilized (string)                              #
    "acquisitionFunction"       : "EI",                                         #
    # Settings for the first iteration of the optimizer (int/int/string/float)  #
    "initial_neurons"           : 16,                                           #
    "initial_layers"            : 1,                                            #
    "initial_activation"        : "relu",                                       #
    "initial_learningRate"      : 1e-4,                                         #
    # Loss                                                                      #
    "lossFunction"              : "mae", #"msle",  #"mse",                                        #
    # Batch size																#
    "batchSize"					: 32,											#
                                                                                #                                                                                                                      
    ##### DESIGN SPACE SETTINGS #####                                           #
    # Lower and upper bound for number of layers (int/int)                      #
    "layers_lowerBound"         : 1,                                            #
    "layers_upperBound"         : 25,                                           #
    # Lower and upper bound for number of neurons (int/int)                     #
    "neurons_lowerBound"        : 5,                                            #
    "neurons_upperBound"        : 512,                                          #
    # Lower and upper bound for learning rate (float/float)                     #
    "learning_lowerBound"       : 1e-8, #1e-6,                                         #
    "learning_upperBound"       : 1e-3, #1e-2,                                         #
                                                                                #
    ##### OTHER UTILITIES ##### (bool/bool/int)                                 #
    "plot_results"              : True,                                         #
    "save_model"                : True,                                         #
    # Early stop to avoid overfit:don't touch unless you know what you're doing!#
    "earlyStop_patience"        : 5,                                            #
                                                                                #       
    ##### OTHER UTILITIES ##### (bool/bool/int)                                 #
    "dimension"                 : 3,                                            #
    "code_debugging"            : True,                                         #
    "enforce_realizability"     : False,                                        #
    "num_realizability_its"     : 5,                                            #      
    "capping"     				: False,                                        #
    "cappingValue"     			: 1e+6,                                         # 
	"timeScale"                 : 1                                             #                                             #                                                                                # 
}                                                                               #
################################################################################

# Casefolder
resultFolder = str(settings["resultFolder"]) 
if not os.path.exists(resultFolder):
   os.mkdir(resultFolder)
   print(f"Training results folder '{resultFolder}' created.")
# Input file
inputFile = str(settings["inputFile"]) 

def saveFigTensor(data, dataName, nSize=9):
   nColSqrt = int(nSize**0.5)
   fig, ax = plt.subplots(nColSqrt,nColSqrt)
   fig.suptitle(dataName)
   for i in range(nColSqrt):
       for j in range(nColSqrt):
           ax[i, j].plot(data[:,nColSqrt*i+j], "+")    
   plt.savefig(resultFolder+"/"+dataName+".png")
   np.savetxt(resultFolder+"/"+dataName+".dat", data, delimiter=" ")

def saveFigScalar(data, dataName):
   fig = plt.figure()
   plt.title(dataName)
   plt.plot(data[:], "+")    
   plt.savefig(resultFolder+"/"+dataName+".png")
   np.savetxt(resultFolder+"/"+dataName+".dat", data, delimiter=" ")

    
@staticmethod
def make_realizable(labels):
   """
   This function is specific to turbulence modeling.
   Given the anisotropy tensor, this function forces realizability
   by shifting values within acceptable ranges for Aii > -1/3 and 2|Aij| < Aii + Ajj + 2/3
   Then, if eigenvalues negative, shifts them to zero. Noteworthy that this step can undo
   constraints from first step, so this function should be called iteratively to get convergence
   to a realizable state.
   :param labels: the predicted anisotropy tensor (num_points X 9 array)
   """
   numPoints = labels.shape[0]
   A = np.zeros((3, 3))
   for i in range(numPoints):
       # Scales all on-diags to retain zero trace
       if np.min(labels[i, [0, 4, 8]]) < -1./3.:
           labels[i, [0, 4, 8]] *= -1./(3.*np.min(labels[i, [0, 4, 8]]))
       if 2.*np.abs(labels[i, 1]) > labels[i, 0] + labels[i, 4] + 2./3.:
           labels[i, 1] = (labels[i, 0] + labels[i, 4] + 2./3.)*.5*np.sign(labels[i, 1])
           labels[i, 3] = (labels[i, 0] + labels[i, 4] + 2./3.)*.5*np.sign(labels[i, 1])
       if 2.*np.abs(labels[i, 5]) > labels[i, 4] + labels[i, 8] + 2./3.:
           labels[i, 5] = (labels[i, 4] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 5])
           labels[i, 7] = (labels[i, 4] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 5])
       if 2.*np.abs(labels[i, 2]) > labels[i, 0] + labels[i, 8] + 2./3.:
           labels[i, 2] = (labels[i, 0] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 2])
           labels[i, 6] = (labels[i, 0] + labels[i, 8] + 2./3.)*.5*np.sign(labels[i, 2])

       # Enforce positive semidefinite by pushing evalues to non-negative
       A[0, 0] = labels[i, 0]
       A[1, 1] = labels[i, 4]
       A[2, 2] = labels[i, 8]
       A[0, 1] = labels[i, 1]
       A[1, 0] = labels[i, 1]
       A[1, 2] = labels[i, 5]
       A[2, 1] = labels[i, 5]
       A[0, 2] = labels[i, 2]
       A[2, 0] = labels[i, 2]
       evalues, evectors = np.linalg.eig(A)
       if np.max(evalues) < (3.*np.abs(np.sort(evalues)[1])-np.sort(evalues)[1])/2.:
           evalues = evalues*(3.*np.abs(np.sort(evalues)[1])-np.sort(evalues)[1])/(2.*np.max(evalues))
           A = np.dot(np.dot(evectors, np.diag(evalues)), np.linalg.inv(evectors))
           for j in range(3):
               labels[i, j] = A[j, j]
           labels[i, 1] = A[0, 1]
           labels[i, 5] = A[1, 2]
           labels[i, 2] = A[0, 2]
           labels[i, 3] = A[0, 1]
           labels[i, 7] = A[1, 2]
           labels[i, 6] = A[0, 2]
       if np.max(evalues) > 1./3. - np.sort(evalues)[1]:
           evalues = evalues*(1./3. - np.sort(evalues)[1])/np.max(evalues)
           A = np.dot(np.dot(evectors, np.diag(evalues)), np.linalg.inv(evectors))
           for j in range(3):
               labels[i, j] = A[j, j]
           labels[i, 1] = A[0, 1]
           labels[i, 5] = A[1, 2]
           labels[i, 2] = A[0, 2]
           labels[i, 3] = A[0, 1]
           labels[i, 7] = A[1, 2]
           labels[i, 6] = A[0, 2]

   return labels


# Dimension
nDim = int(settings["dimension"])  

# Debugging
debug = bool(settings["code_debugging"]) 

# Realizability constraint
enforce_realizability = bool(settings["enforce_realizability"])
num_realizability_its = int(settings["num_realizability_its"]) 

# Capping value
capping = bool(settings["capping"]) 
cappingValue = float(settings["cappingValue"]) 

# Read dataset
dataset = np.loadtxt(inputFile, skiprows=1)

# Number of data
num_points = dataset.shape[0]
if(debug):
	print("Number of points =",num_points)


# Alpp
alpp = dataset[:, 0]
# Granular pressure
Pp = dataset[:, 1]

# Timescale
timeScale = np.ones(dataset.shape[0])*float(settings["timeScale"])  

# GradUs
grad_u_flat = dataset[:, 2:11]
if(debug):
	saveFigTensor(grad_u_flat, "grad_u")

# Reshape grad_u and stresses to num_points X 3 X 3 arrays
grad_u = np.zeros((num_points, nDim, nDim))
for i in range(nDim):
   for j in range(nDim):
       grad_u[:, i, j] = grad_u_flat[:, i*nDim+j]

# Calculate Sij & Rij
Sij = np.zeros((num_points, nDim, nDim))
Rij = np.zeros((num_points, nDim, nDim))
for i in range(num_points):
   Sij[i, :, :] = timeScale[i] * ( 0.5*(grad_u[i, :, :] + np.transpose(grad_u[i, :, :])) - 1./nDim*np.eye(nDim)*np.trace(grad_u[i, :, :]) )
   #Sij[i, :, :] = timeScale[i] * ( 0.5*(grad_u[i, :, :] + np.transpose(grad_u[i, :, :])) ) 
   #Sij[i, :, :] = Sij[i, :, :]/tf.norm(Sij[i, :, :])
   Rij[i, :, :] = timeScale[i] * 0.5 * (grad_u[i, :, :] - np.transpose(grad_u[i, :, :]))
   #Rij[i, :, :] = Rij[i, :, :]/tf.norm(Rij[i, :, :])

# For debugging purposes
if(debug):
   flatSij = np.zeros((num_points, nDim*nDim))
   for i in range(nDim):
       for j in range(nDim):
           flatSij[:, nDim*i+j] = Sij[:, i, j]
   saveFigTensor(flatSij, "Sij")
   #
   flatRij = np.zeros((num_points, nDim*nDim))
   for i in range(nDim):
       for j in range(nDim):
           flatRij[:, nDim*i+j] = Rij[:, i, j]
   saveFigTensor(flatRij, "Rij")
#'''

if enforce_realizability:
   for i in range(num_realizability_its):
       flatSij = make_realizable(flatSij)
   for i in range(nDim):
       for j in range(nDim):
           Sij[:, i, j] = flatSij[:, nDim*i+j]    

#'''

'''
Given total particle stress tensor (num_points X 3 X 3), return flattened non-dimensional anisotropy tensor
:stresses: total particle stress tensor
:scaled_stresses: (num_points X 9) anisotropy tensor.  aij = (uiuj)/2Pp - 1./3. * delta_ij
'''
# Particle stresses
stresses_flat = dataset[:, 11:20]    
#stresses_flat = dataset[:, 21:30] 
stresses = np.zeros((num_points, nDim, nDim))

for i in range(nDim):
   for j in range(nDim):
       stresses[:, i, j] = stresses_flat[:, i*nDim+j]
anisotropy = np.zeros((num_points, nDim, nDim))
'''
for i in range(nDim):
   for j in range(nDim):
       trace = 0.5 * (stresses[:, 0, 0] + stresses[:, 1, 1] + stresses[:, 2, 2])
       trace = np.maximum(trace, 1e-8)
       anisotropy[:, i, j] = stresses[:, i, j]/(2.0 * trace)
   anisotropy[:, i, i] -= 1./3. 
'''
for i in range(num_points):
    trace = np.maximum(np.trace(stresses[i, :, :]), 1e-16)
    anisotropy[i, :, :] = (stresses[i, :, :] - 1./nDim*np.eye(nDim)*np.trace(stresses[i, :, :]))/trace
    
scaled_stresses = np.zeros((num_points, nDim*nDim))
for i in range(nDim):
   for j in range(nDim):
       scaled_stresses[:, nDim*i+j] = anisotropy[:, i, j]
#'''        
# Enforce realizability
if enforce_realizability:
   for i in range(num_realizability_its):
       scaled_stresses = make_realizable(scaled_stresses)        
#'''
if(debug):
	saveFigTensor(scaled_stresses, "scaled_stresses")

#'''

# Granular temperature
Theta = dataset[:, 20]   

# Calculate invariants
num_invariants = 3 #7 #6 #8 #7 #5
invariants_i = np.zeros((num_points, num_invariants))
for i in range(num_points):   
   sij = Sij[i, :, :]
   rij = Rij[i, :, :]   
   invariants_i[i, 0] = np.trace(np.dot(sij, sij))
   #invariants_i[i, 1] = np.trace(np.dot(rij, rij))
   #invariants_i[i, 2] = np.trace(np.dot(sij, np.dot(sij, sij)))
   #invariants_i[i, 3] = np.trace(np.dot(rij, np.dot(rij, sij)))
   #invariants_i[i, 4] = np.trace(np.dot(np.dot(rij, rij), np.dot(sij, sij)))
   invariants_i[i, 1] = np.log(tf.math.maximum(Pp[i], 1e-16))
   #invariants_i[i, 2] = np.log(tf.math.maximum(alpp[i]/0.64, 1e-16))
   invariants_i[i, 2] = alpp[i]/0.64
   #invariants_i[i, 3] = Theta[i]

if(debug):
	for i in range(num_invariants):
		saveFigScalar(invariants_i[:,i],  "invariants"+str(i))

# Normalization
mu = np.mean(invariants_i, axis=0)
std = np.std(invariants_i, axis=0)
scaled_invariants = (invariants_i - mu) / std
#scaled_invariants = np.zeros((num_points, num_invariants))
#for i in range(num_invariants): 
#    scaled_invariants[:, i] = tf.keras.layers.UnitNormalization()(invariants_i[:, i])
#scaled_invariants = invariants_i

if(debug):
	for i in range(num_invariants):
		saveFigScalar(scaled_invariants[:,i],  "scaled_invariants"+str(i))

# Calculate tensor bases#    
num_tensor_basis = 1 #5 #4
Tb_i = np.zeros((num_points, num_tensor_basis, nDim, nDim))
for i in range(num_points):
   sij = Sij[i, :, :]
   rij = Rij[i, :, :]
   Tb_i[i, 0, :, :] = sij
   #Tb_i[i, 1, :, :] = np.dot(sij, rij) - np.dot(rij, sij)
   #Tb_i[i, 2, :, :] = np.dot(sij, sij) - 1./nDim*np.eye(nDim)*np.trace(np.dot(sij, sij))
   #Tb_i[i, 3, :, :] = np.dot(rij, rij) - 1./nDim*np.eye(nDim)*np.trace(np.dot(rij, rij))
   for j in range(num_tensor_basis):
       Tb_i[i, j, :, :] = Tb_i[i, j, :, :] - 1./nDim*np.eye(nDim)*np.trace(Tb_i[i, j, :, :])    

Tb = np.zeros((num_points, num_tensor_basis, nDim*nDim))
for i in range(nDim):
   for j in range(nDim):
       Tb[:, :, nDim*i+j] = Tb_i[:, :, i, j]

#muTb = center(Tb, settings["centering_method"])
#sigmaTb = scale(Tb, settings["scaling_method"])
#scaled_Tb = center_scale(Tb, muTb, sigmaTb)
scaled_Tb = Tb

if(debug):
	for i in range(num_tensor_basis):
		saveFigTensor(scaled_Tb[:, i, :], "Tb"+str(i))

#
#
# Initialize other quantitites and create a folder to store the trained model
path_best_model = '19_best_model.hdf5'
best_errorPrediction = 10000000000000000
newDirName =   "optimizeNetworkAvalancheStresses" + "_epochs=" + str(settings["iterations_optimizer"]) + "_evals=" + str(settings["network_epochs"]) + "scaling=" + str(settings["scaling_method"])
os.mkdir(newDirName)
os.chdir(newDirName)

# Define the design space with the values that were set in the dictionary above
numberLayers = Integer(low=int(settings["layers_lowerBound"]), high=int(settings["layers_upperBound"]), name='layers')
numberNeurons = Integer(low=int(settings["neurons_lowerBound"]), high=int(settings["neurons_upperBound"]), name='neurons')
dim_activation = Categorical(categories=['relu', 'elu', 'selu', 'leaky_relu'], name='activation')
dim_learning_rate = Real(low=float(settings["learning_lowerBound"]), high=float(settings["learning_upperBound"]), prior='log-uniform', name='alpha')

# Initialize the optimizer with the values chosen above
dimensions = [numberLayers, numberNeurons, dim_activation, dim_learning_rate]
default_parameters = [int(settings["initial_layers"]), int(settings["initial_neurons"]), str(settings["initial_activation"]), float(settings["initial_learningRate"])]

def log_dir_name(layers, neurons, activation, alpha):
    '''
    This function creates a folder for each training.
    Input vars:
        layers: number of layers for the i-th training.
        neurons: number of neurons for the i-th training.
        activation: activation function utilized for the i-th training.
        alpha: LR utilized for the i-th training.
    '''

    s = "./Logs/layers_{0}_nodes_{1}_activation_{2}_alpha_{3}/"
    log_dir = s.format(layers, neurons, activation, alpha)

    return log_dir

# Compute how much is to be used for test:
# Tranining ratio (in settings) is in percentage
test_ratio = 1-float(settings["training_ratio"])*0.01
x_train, x_test, tb_train, tb_test, y_train, y_test = train_test_split(scaled_invariants, scaled_Tb, scaled_stresses, test_size=test_ratio, random_state=42)  

if(debug):
	print("x_train.shape =", x_train.shape)
	print("x_test.shape =", x_test.shape)
	print("tb_train.shape =", tb_train.shape)
	print("tb_test.shape =", tb_test.shape)
	print("y_train.shape =", y_train.shape)
	print("y_test.shape =", y_test.shape)
'''    
	for i in range(num_tensor_basis):
		saveFigTensor(tb_train[:, i, :], "tb_train"+str(i))
		saveFigTensor(tb_test[:, i, :], "tb_test"+str(i))           
'''
def create_model(layers, neurons, activation, alpha):
    '''
    This function creates the ANN model. It starts with splitting the dataset in
    train/test, and then it creates the architecture given the input variables.
    Input vars:
        layers: number of layers for the i-th training.
        neurons: number of neurons for the i-th training.
        activation: activation function utilized for the i-th training.
        alpha: LR utilized for the i-th training.
    Output vars:
        model: model for the ANN.
    '''
    # Initializer
    initializer = 'normal'
    #initializer = tf.keras.initializers.HeNormal()
    #initializer = tf.keras.initializers.HeUniform()

    # The first branch operates on the scaled invariants
    inputX = Input(shape = (num_invariants,)) 

    # Add all the hidden layers and activate each of them
    for i in range(layers+1):
        if i == 0:
            layerX = Dense(neurons, kernel_initializer=initializer, activation=tf.keras.activations.get(activation))(inputX)
        layerX = Dense(neurons, kernel_initializer=initializer, activation=tf.keras.activations.get(activation))(layerX)
    #layerX = Dense(num_tensor_basis, activation=tf.keras.activations.get(activation))(layerX)
    layerX = Dense(num_tensor_basis, activation='linear')(layerX)
    layerX = Model(inputs=inputX, outputs=layerX)

    # Scaled tensor basis inputs
    inputTbs = Input(shape = (num_tensor_basis,nDim*nDim,))

    # Multiply g by Tb          
    var = tf.keras.layers.Dot(axes=1)([layerX.output, inputTbs])

    # Create model
    model = Model(inputs=[layerX.input, inputTbs], outputs=var)

	# Model summary
	#model.summary()
    
    # Define the optimizer with the given initial LR
    optimizer = Adam(learning_rate=float(alpha))
    lossF = settings["lossFunction"]
    model.compile(optimizer=optimizer, loss=lossF, metrics=[lossF])
    
    return model

@use_named_args(dimensions=dimensions)
def fitness(layers, neurons, activation, alpha):
    '''
    This function trains the ANN model, and evaluates the error that corresponds
    to that particular architecture.
    Input vars:
        layers: number of layers for the i-th training.
        neurons: number of neurons for the i-th training.
        activation: activation function utilized for the i-th training.
        alpha: LR utilized for the i-th training.
    Output vars:
        errorPrediction: error which is observed for the i-th training.
    '''
    try:
        # Create model
        model = create_model(layers=layers, neurons=neurons, activation=activation, alpha=alpha)        
        batchS = settings["batchSize"]
                
        # Create a folder to store the training
        log_dir = log_dir_name(layers, neurons, activation, alpha)
        callback_log = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_grads=False, write_images=False)
        earlyStopping = EarlyStopping(monitor='val_loss', patience=int(settings["earlyStop_patience"]), verbose=0, mode='min')
        
        # Fit model
        history = model.fit(x=[x_train, tb_train], y=y_train, validation_data=([x_test, tb_test],y_test), 
        				    epochs=int(settings["network_epochs"]), batch_size=batchS, callbacks=[callback_log, earlyStopping])
                
        # Error prediction
        lossF = settings["lossFunction"]
        errorPrediction = history.history[lossF][-1]
        print("Regression error (",lossF,"): {}".format(errorPrediction))
        # Save the model if it improves on the best performance.
        global best_errorPrediction

        # If the errorPrediction is improved, then this is the best model so far
        if errorPrediction < best_errorPrediction:
            model.save(path_best_model)
            best_errorPrediction = errorPrediction
            print("********* NEW BEST MODEL FOUND *********")
            print('Number of layers:', layers)
            print('Number of neurons:', neurons)
            print('Activation function:', activation)
            print('Learning rate:', alpha)
            print("******************")

        del model
        K.clear_session()

    except:
        print("I am in Except")
        exit()
        pass

    return errorPrediction

# This is the function of skopt to train the optimizer
fitness(x= default_parameters)
search_result = gp_minimize(func=fitness, dimensions=dimensions, acq_func=str(settings["acquisitionFunction"]), n_calls=int(settings["iterations_optimizer"]), x0=default_parameters)


##### PART 2: plotting the results and saving the model #####
if settings["plot_results"]:
    #_AO plot_convergence(search_result, y_scale="log")
    plot_convergence(search_result, yscale="log")
    plt.savefig("Converge.png", dpi=400)
    #plt.show()
    plt.close()


print(search_result.x)
print(search_result.fun)

# Load optimised model
best_model= load_model(path_best_model)
opt_par = search_result.x

# use hyper-parameters from optimization
num_layers = opt_par[0]
num_nodes = opt_par[1]
opti_acti = opt_par[2]
opti_LR = opt_par[3]

text_file = open("best_training.txt", "wt")
neurons_number = text_file.write("The optimal number of neurons is equal to: {} \n".format(num_nodes))
layers_number = text_file.write("The optimal number of layers is equal to: {} \n".format(num_layers))
acti_opt = text_file.write("The optimal acquisition function is: {} \n".format(str(opti_acti)))
layers_number = text_file.write("The optimal LR is equal to: {} \n".format(opti_LR))
text_file.close()

#
#test_results = {}
#test_results['model'] = best_model.evaluate((x_test,tb_test), y_test, verbose=0)
#pd.DataFrame(test_results, index=['Mean absolute error']).T

# Predicted stresses
pred_all = best_model.predict((x_test,tb_test))
    
if settings["save_model"]:
    np.save("prediction_trainData.npy", pred_all)
    best_model.save("bestModel.hdf5")

    # save model structure
    model_json = best_model.to_json()
    with open('bayesModel.json', 'w', encoding = 'utf8') as json_file:
        json_file.write(model_json)


# save ground truth and prediction to local disk as CSV file
filename="../"+resultFolder+"/"+"truePredictionsBayesian.dat"
oof = pd.DataFrame(dict(
           pd = pred_all[:,0],
           gt = y_test[:,0]
     ))
oof.to_csv(filename, index=False)
oof.head(20)

#a = plt.axes(aspect='equal')
nPlt = 3
fig, ax = plt.subplots(nPlt, nPlt)	
for axi in ax.flat:
   axi.axline((0,0), slope=1, color='r', linewidth=2, linestyle='--')  
fig.set_size_inches(18.5, 10.5)
for i in range(nPlt):
   for j in range(nPlt):
       ax[i, j].scatter(y_test[:,nPlt*i+j], pred_all[:,nPlt*i+j])
       ax[i, j].set_box_aspect(1)
       ax[i, j].set_xlim([np.min(y_test[:,nPlt*i+j]), np.max(y_test[:,nPlt*i+j])])
       ax[i, j].set_ylim([np.min(y_test[:,nPlt*i+j]), np.max(y_test[:,nPlt*i+j])])   
plt.show()

