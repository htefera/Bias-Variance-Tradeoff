from __future__ import division
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt

# load and save model
import pickle

#%% fix seed for reproducibility
SEED = 888
np.random.seed(SEED)

#%% number of foldes for cross validation

numFolds = 5

#result_path = ['degree1' , 'degree2', 'degree3', 'degree4']
poly_degree  = [1,5,6,7]

#%% load data
colnames = ["Sex", "Length", "Diameter", "Height", "Whole weight", 
                "Shucked weight", "Viscera weight", "Shell weight", "Rings"]
dataset = pd.read_csv("abalone.csv", names=colnames)

Y = dataset["Rings"]
Y = Y.values
# Some data preprocessing
X = dataset.drop(["Rings","Sex", "Shell weight", "Shucked weight"],axis=1)
X  = X.values

#%% separate test and training
test_indices = [5,60,100]
X_test = X[test_indices,:]
Y_test = Y[test_indices]


X_train = np.delete(X,test_indices,axis=0)
#axis 0 is row 
Y_train = np.delete(Y,test_indices,axis=0)
Y_train  = np.reshape(Y_train,(len(Y_train),1))

kf = KFold(len(Y_train), numFolds, shuffle=True)

#%% for bias and variance of the data
bias = np.zeros((len(test_indices),len(poly_degree)),dtype='float64')
variance = np.zeros((len(test_indices),len(poly_degree)),dtype='float64')
#bias variable row 3 column

#%% polynomial evaluation
P1_result  = np.zeros((len(test_indices),numFolds),dtype='float64')
P2_result  = np.zeros((len(test_indices),numFolds),dtype='float64')
P3_result  = np.zeros((len(test_indices),numFolds),dtype='float64')
P4_result  = np.zeros((len(test_indices),numFolds),dtype='float64')

def compute_bias(predicted_values, actual):
    #compute  expectation
    mean_val=np.mean(predicted_values,axis=1)
    #axis =1 is column wise addition 
    mean_val = np.reshape(mean_val, (len(mean_val), 1))
    actual = np.reshape(actual, (len(mean_val), 1))
    
    bias = mean_val - actual;
    bias  = np.square(bias)
    bias  = np.reshape(bias, (len(actual), 1))
    
    return bias

def compute_variance(predicted_values):
    # compute expectation
    mean_val = np.mean(predicted_values, axis=1)
    mean_val = np.reshape(mean_val, (len(mean_val), 1))
    diff_sqr = np.square(predicted_values - mean_val )
    #compute variance
    var=np.mean(diff_sqr,axis=1)
    var = np.reshape(var, (len(mean_val), 1))
    
    return var

def main():
    """Following code is for calculating the age of the abalone plant"""

    Model = LinearRegression
    i = 0
    for train_indices, test_indices in kf:
        #split data
        X_train_ = X_train[train_indices, :]; Y_train_ = Y[train_indices]
#        X_test = training_data.ix[test_indices, :]; Y_test = Y[test_indices]

        # Testing out on the linear reg_modelression
        for d,degree in enumerate(poly_degree):
            reg_model = Model()
            if degree == 1:
                 X_tr = X_train_
                 X_tst = X_test
                
            else:
                # PolynomialFeatures (prepreprocessing)
                poly = PolynomialFeatures(degree=degree)
                X_tr = poly.fit_transform(X_train_)
                X_tst = poly.fit_transform(X_test)
            
            # fit regressor
            reg_model.fit(X_tr, Y_train_)
            
            # make prediction
            predictions = reg_model.predict(X_tst)
            predictions = np.reshape(predictions, (len(predictions), 1))

            # store result
            if d==1:
                P1_result[:,[i]] = predictions
            elif d==2:
                P2_result[:,[i]] = predictions
            elif d == 3:
                P3_result[:,[i]] = predictions
            elif d == 4:
                P4_result[:,[i]] = predictions
        i += 1
    #%% compute bias and variance for each complexity
    bias[:,[0]]  = compute_bias (P1_result, Y_test)
    variance[:,[0]]  = compute_variance(P1_result)
    
    bias[:,[1]]  = compute_bias (P2_result, Y_test)
    variance[:,[1]]  = compute_variance(P2_result)
    
    bias[:,[2]]  = compute_bias (P3_result, Y_test)
    variance[:,[2]]  = compute_variance(P3_result)
    
    bias[:,[3]]  = compute_bias (P4_result, Y_test)
    variance[:,[3]]  = compute_variance(P4_result)

def plot_bias_var(bias=None, var=None):
#    x  = range(1,len(poly_degree)+1)
    x = np.linspace(1, len(poly_degree), len(poly_degree), endpoint=True)
    
    # plot bias and varinave for each data point
    for data_num in range(bias.shape[0]):
        y_bias  = bias[data_num,:]
        y_var  = var[data_num,:]
        plt.plot(x,y_bias, 'r', label='bias') 
        plt.plot(x, y_var, 'b', label='variance') 
        plt.xlabel("Complexity")
        plt.ylabel("Bias Variance")
        plt.legend()
        plt.show()
    
if __name__ == "__main__":
    main()
    plot_bias_var(bias, variance)