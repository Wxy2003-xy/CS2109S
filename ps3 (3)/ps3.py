# Inital imports and setup

import os
import numpy as np

###################
# Helper function #
###################
def load_data(filepath):
    '''
    Load in the given csv filepath as a numpy array

    Parameters
    ----------
    filepath (string) : path to csv file

    Returns
    -------
        X, y (np.ndarray, np.ndarray) : (m, num_features), (m,) numpy matrices
    '''
    *X, y = np.genfromtxt(
        filepath,
        delimiter=',',
        skip_header=True,
        unpack=True,
    ) # default dtype: float
    X = np.array(X, dtype=float).T # cast features to int type
    return X, y.reshape((-1, 1))

data_filepath = 'housing_data.csv'
X, y = load_data(data_filepath)

### Task 1.1: Mean Squared Error (MSE)

def mean_squared_error(y_true, y_pred):
    '''
    Calculate mean squared error between y_pred and y_true.

    Parameters
    ----------
    y_true (np.ndarray) : (m, 1) numpy matrix consists of true values
    y_pred (np.ndarray)   : (m, 1) numpy matrix consists of predictions
    
    Returns
    -------
        The mean squared error value.
    '''
    
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_1_1():
    y_true, y_pred = np.array([[3], [5]]), np.array([[12], [15]])
    
    assert mean_squared_error(y_true, y_pred) in [45.25, 90.5]

### Task 1.2: Mean Absolute Error (MAE)

def mean_absolute_error(y_true, y_pred):
    '''
    Calculate mean absolute error between y_pred and y_true.

    Parameters
    ----------
    y_true (np.ndarray) : (m, 1) numpy matrix consists of true values
    y_pred (np.ndarray)   : (m, 1) numpy matrix consists of predictions
    
    Returns
    -------
        The mean absolute error value.
    '''
  
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_1_2():
    y_true, y_pred = np.array([[3], [5]]), np.array([[12], [15]])
    
    assert mean_absolute_error(y_true, y_pred) == 9.5

### Task 2.1: Adding a bias column

def add_bias_column(X):
    '''
    Create a bias column and combine it with X.

    Parameters
    ----------
    X : (m, n) numpy matrix representing a feature matrix
    
    Returns
    -------
        new_X (np.ndarray):
            A (m, n + 1) numpy matrix with the first column consisting of all 1s
    '''
  
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_2_1():
    without_bias = np.array([[1, 2], [3, 4]])
    expected = np.array([[1, 1, 2], [1, 3, 4]])
    
    assert np.array_equal(add_bias_column(without_bias), expected)

### Task 2.2: Get best fitting bias and weights

def get_bias_and_weight(X, y, include_bias = True):
    '''
    Calculate bias and weights that give the best fitting line.

    Parameters
    ----------
    X (np.ndarray) : (m, n) numpy matrix representing feature matrix
    y (np.ndarray) : (m, 1) numpy matrix representing target values
    include_bias (boolean) : Specify whether the model should include a bias term
    
    Returns
    -------
        bias (float):
            If include_bias = True, return the bias constant. Else,
            return 0
        weights (np.ndarray):
            A (n, 1) numpy matrix representing the weight constant(s).
    '''
    Z = add_bias_column(X)
    return np.linalg.inv(np.transpose(Z) @ Z) @ np.transpose(Z) @ y
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_2_2():
    public_X, public_y = np.array([[1, 3], [2, 3], [3, 4]]), np.arange(4, 7).reshape((-1, 1))
    
    test_1 = (round(get_bias_and_weight(public_X, public_y)[0], 5) == 3)
    test_2 = np.array_equal(np.round(get_bias_and_weight(public_X, public_y)[1], 1), np.array([[1.0], [0.0]]))
    test_3 = np.array_equal(np.round(get_bias_and_weight(public_X, public_y, False)[1], 2), np.round(np.array([[0.49], [1.20]]), 2))
    
    assert test_1 and test_2 and test_3

### Task 2.3: Get the prediction line

def get_prediction_linear_regression(X, y, include_bias = True):
    '''
    Calculate the best fitting line.

    Parameters
    ----------
    X (np.ndarray) : (m, n) numpy matrix representing feature matrix
    y (np.ndarray) : (m, 1) numpy matrix representing target values
    include_bias (boolean) : Specify whether the model should include a bias term

    Returns
    -------
        y_pred (np.ndarray):
            A (m, 1) numpy matrix representing prediction values.
    '''
  
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_2_3():
    test_X, test_y = np.array([[1, 3], [2, 3], [3, 4]]), np.arange(4, 7).reshape((-1, 1))
    
    assert round(mean_squared_error(test_y, get_prediction_linear_regression(test_X, test_y)), 5) == 0

import matplotlib.pyplot as plt

area = X[:, 0].reshape((-1, 1))
predicted = get_prediction_linear_regression(area, y)
plt.scatter(area, y)
plt.plot(area, predicted, color = 'r')
plt.xlabel("Size in square meter")
plt.ylabel("Price in SGD")
plt.show()

### Task 2.4: Gradient Descent on multiple features

def gradient_descent_multi_variable(X, y, lr = 1e-5, number_of_epochs = 250):
    '''
    Approximate bias and weight that gave the best fitting line.

    Parameters
    ----------
    X (np.ndarray) : (m, n) numpy matrix representing feature matrix
    y (np.ndarray) : (m, 1) numpy matrix representing target values
    lr (float) : Learning rate
    number_of_epochs (int) : Number of gradient descent epochs
    
    Returns
    -------
        bias (float):
            The bias constant
        weights (np.ndarray):
            A (n, 1) numpy matrix that specifies the weight constants.
        loss (list):
            A list where the i-th element denotes the MSE score at i-th epoch.
    '''
    # Do not change
    bias = 0
    weights = np.full((X.shape[1], 1), 0).astype(float)
    loss = []
    
    m = X.shape[0]
    pred = X @ weights + bias
    for _ in range(number_of_epochs):
        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """
    
    return bias, weights, loss

def test_task_2_4():
    _, _, loss = gradient_descent_multi_variable(X, y, lr = 1e-5, number_of_epochs = 250)
    loss_initial = loss[0]
    loss_final = loss[-1]
    
    assert loss_initial > loss_final

### Task 2.5: Which algorithm should we use for Linear Regression?

### Task 3.1 : Create Polynomial Matrix

def create_polynomial_matrix(X, power = 2):
    '''
    Create a polynomial matrix.
    
    Parameters
    ----------
    X: (m, 1) numpy matrix

    Returns
    -------
        A (m, power) numpy matrix where the i-th column denotes
            X raised to the power of i.
    '''
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_3_1():
    vector = np.array([[1], [2], [3]])
    poly_matrix = np.array([[1, 1, 1], [2, 4, 8], [3, 9, 27]])
    
    assert np.array_equal(create_polynomial_matrix(vector, 3), poly_matrix)

### Task 3.2: Get the prediction line

def get_prediction_poly_regression(X, y, power = 2, include_bias = True):
    '''
    Calculate the best polynomial line.

    Parameters
    ----------
    X (np.ndarray) : (m, 1) numpy matrix representing feature matrix
    y (np.ndarray) : (m, 1) numpy matrix representing target values
    power (int) : Specify the degree of the polynomial
    include_bias (boolean) : Specify whether the model should include a bias term

    Returns
    -------
        A (m, 1) numpy matrix representing prediction values.
    '''
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_3_2():
    test_X, test_y = np.arange(3).reshape((-1, 1)), np.arange(4, 7).reshape((-1, 1))
    pred_y = get_prediction_poly_regression(test_X, test_y, 2)
    
    assert round(mean_squared_error(test_y, pred_y), 5) == 0

import matplotlib.pyplot as plt

schools = X[:, 2].reshape((-1, 1))
predicted = get_prediction_poly_regression(schools, y, 3)
plt.scatter(schools, y)
plt.scatter(schools, predicted, color = 'r', s = 100)
plt.xlabel("Number of schools within 1km")
plt.ylabel("Price in SGD")
plt.show()

### Task 3.3: Feature Scaling

def feature_scaling(X):
    '''
    Mean normalized each feature column.

    Parameters
    ----------
    X (np.ndarray) : (m, n) numpy matrix representing feature matrix

    Returns
    -------
        A (m, n) numpy matrix where each column has been mean-normalized.
    '''
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_3_3():
    public_X = np.array([[1, 133], [4, 700], [5, 133], [8, 700]])
    expected = np.array([[-1.4, -1], [-0.2, 1], [0.2, -1], [1.4, 1]])
    
    assert np.array_equal(feature_scaling(public_X), expected)

### Task 3.4: Find number of epochs to converge

def find_number_of_epochs(X, y, lr, delta_loss):
    '''
    Do gradient descent until convergence and return number of epochs
    required.

    Parameters
    ----------
    X (np.ndarray) : (m, n) numpy matrix representing feature matrix
    y (np.ndarray) : (m, 1) numpy matrix representing target values
    lr (float) : Learning rate
    delta_loss (float) : Termination criterion
    
    Returns
    -------
        bias (float):
            The bias constant
        weights (np.ndarray):
            A (n, 1) numpy matrix that specifies the weight constants.
        num_of_epochs (int):
            Number of epochs to reach convergence.
        current_loss (float):
            The loss value obtained after convergence.
    '''
    # Do not change
    bias = 0
    weights = np.full((X.shape[1], 1), 0).astype(float)
    num_of_epochs = 0
    previous_loss = 1e14
    current_loss = -1e14

    m = X.shape[0]
    pred = X @ weights + bias
    current_loss = mean_squared_error(y, pred)
    while abs(previous_loss - current_loss) >= delta_loss:
        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """
    
    return bias, weights, num_of_epochs, current_loss

def test_task_3_4():
    poly_X = create_polynomial_matrix(X[:, 2].reshape((-1, 1)), 3)
    _, _, num_of_epochs, _ = find_number_of_epochs(poly_X, y, 1e-5, 1e7)
    
    assert num_of_epochs > 0

### Task 3.5: Analyze the effects of feature scaling on Gradient Descent

# Initial imports and setup

import numpy as np
import os
import pandas as pd

from sklearn import svm
from sklearn import model_selection

# Read credit card data into a Pandas dataframe for large tests

dirname = os.getcwd()
credit_card_data_filepath = os.path.join(dirname, 'credit_card.csv')

credit_df = pd.read_csv(credit_card_data_filepath)
X_task5 = credit_df.values[:, :-1]
y_task5 = credit_df.values[:, -1:]

credit_df.head()

# Inspect the number of fraudulent and non-fraudulent transactions.
credit_df['Class'].value_counts()

# Select the 'Class' column in the credit dataframe
credit_df['Class']

# Obtain the first 2 rows
credit_df[0:2]

credit_df['Class'] == 0

# Obtain the credit dataframe where the 'Class' field is 0
credit_df[credit_df['Class'] == 0]

pd.concat([credit_df[:2], credit_df[-2:]], axis=0)

### Task 4.1: Problem with imbalanced data

### Task 5.1: Cost function

def cost_function(X: np.ndarray, y: np.ndarray, weight_vector: np.ndarray):
    '''
    Cross entropy error for logistic regression

    Parameters
    ----------
    X: np.ndarray
        (m, n) training dataset (features).
    y: np.ndarray
        (m,) training dataset (corresponding targets).
    weight_vector: np.ndarray
        (n,) weight parameters.

    Returns
    -------
    Cost
    '''
    
    # Machine epsilon for numpy `float64` type
    eps = np.finfo(np.float64).eps

    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_5_1():
    data1 = [[111.1, 10, 0], [111.2, 20, 0], [111.3, 10, 0], [111.4, 10, 0], [111.5, 10, 0], [111.6, 10, 1], [111.4, 10, 0], [111.5, 10, 1], [111.6, 10, 1]]
    df1 = pd.DataFrame(data1, columns = ['V1', 'V2', 'Class'])
    X1 = df1.iloc[:, :-1].to_numpy()
    y1 = df1.iloc[:, -1].to_numpy()
    w1 = np.transpose([0.002, 0.1220])
    
    assert np.round(cost_function(X1, y1, w1), 5) == np.round(1.29333, 5)

### Task 5.2: Weight update

def weight_update(X: np.ndarray, y: np.ndarray, alpha: np.float64, weight_vector: np.ndarray) -> np.ndarray:
    '''
    Do the weight update for one step in gradient descent

    Parameters
    ----------
    X: np.ndarray
        (m, n) training dataset (features).
    y: np.ndarray
        (m,) training dataset (corresponding targets).
    alpha: np.float64
        logistic regression learning rate.1
    weight_vector: np.ndarray
        (n,) weight parameters.

    Returns
    -------
    New weight vector after one round of update.
    '''

    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_5_2():
    data1 = [[111.1, 10, 0], [111.2, 20, 0], [111.3, 10, 0], [111.4, 10, 0], [111.5, 10, 0], [111.6, 10, 1],[111.4, 10, 0], [111.5, 10, 1], [111.6, 10, 1]]
    df1 = pd.DataFrame(data1, columns = ['V1', 'V2', 'Class'])
    X1 = df1.iloc[:, :-1].to_numpy()
    y1 = df1.iloc[:, -1].to_numpy()
    w1 = np.transpose([2.2000, 12.20000])
    a1 = 1e-5
    nw1 = np.array([2.199,12.2])
    
    assert np.array_equal(np.round(weight_update(X1, y1, a1, w1), 3), nw1)

### Task 5.3: Logistic regression classification

def logistic_regression_classification(X: np.ndarray, weight_vector: np.ndarray, prob_threshold: np.float64=0.5):
    '''
    Do classification task using logistic regression.

    Parameters
    ----------
    X: np.ndarray
        (m, n) training dataset (features).
    weight_vector: np.ndarray
        (n,) weight parameters.
    prob_threshold: np.float64
        the threshold for a prediction to be considered fraudulent.

    Returns
    -------
    Classification result as an (m,) np.ndarray
    '''

    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_5_3():
    data1 = [[111.1, 10, 0], [111.2, 20, 0], [111.3, 10, 0], [111.4, 10, 0], [111.5, 10, 0], [211.6, 80, 1],[111.4, 10, 0], [111.5, 80, 1], [211.6, 80, 1]]
    df1 = pd.DataFrame(data1, columns = ['V1', 'V2', 'Class'])
    X1 = df1.iloc[:, :-1].to_numpy()
    y1 = df1.iloc[:, -1].to_numpy()
    w1 = np.transpose([-0.000002, 0.000003])
    expected1 = np.transpose([0, 0, 0, 0, 0, 0, 0, 1, 0])
    result1 = logistic_regression_classification(X1, w1)
    
    assert result1.shape == expected1.shape and (result1 == expected1).all()

### Task 5.4: Logistic regression using stochastic gradient descent

def logistic_regression_stochastic_gradient_descent(X_train: np.ndarray, y_train: np.ndarray, max_num_iterations: int=250, threshold: np.float64=0.05, alpha: np.float64=1e-5, seed: int=43) -> np.ndarray:
    '''
    Initialize your weight to zeros. Write a terminating condition, and run the weight update for some iterations.
    Get the resulting weight vector.

    Parameters
    ----------
    X_train: np.ndarray
        (m, n) training dataset (features).
    y_train: np.ndarray
        (m,) training dataset (corresponding targets).
    max_num_iterations: int
        this should be one of the terminating conditions. 
        The gradient descent step should happen at most max_num_iterations times.
    threshold: np.float64
        terminating when error <= threshold value, or if you reach the max number of update rounds first.
    alpha: np.float64
        logistic regression learning rate.
    seed: int
        seed for random number generation.

    Returns
    -------
    The final (n,) weight parameters
    '''

    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_5_4():
    data1 = [[111.1, 10, 0], [111.2, 20, 0], [111.3, 10, 0], [111.4, 10, 0], [111.5, 10, 0], [211.6, 80, 1],[111.4, 10, 0], [111.5, 80, 1], [211.6, 80, 1]]
    df1 = pd.DataFrame(data1, columns = ['V1', 'V2', 'Class'])
    X1 = df1.iloc[:, :-1].to_numpy()
    y1 = df1.iloc[:, -1].to_numpy()
    expected1 = cost_function(X1, y1, np.transpose(np.zeros(X1.shape[1])))
    
    assert cost_function(X1, y1, logistic_regression_stochastic_gradient_descent(X1, y1)) < expected1

### Task 5.5: Stochastic gradient descent vs batch gradient descent

### Task 6.1: Linear SVM vs Gaussian Kernel SVM

def linear_svm(X: np.ndarray, y: np.ndarray):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.3, random_state=42)

    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)

    clf_predictions = clf.predict(X_test)

    return clf_predictions, clf.score(X_test, y_test) * 100


# small data
# Do note that y values for data1 are either 0 or 1 for this half of the task, but typically are 
#   either -1 or 1 for SVMs. You do not have to change this data for this task.
data1 = [[111.1, 10, 0], [111.2, 20, 0], [111.3, 10, 0], [111.4, 10, 0], [111.5, 10, 0], [211.6, 80, 1],
        [111.4, 10, 0], [111.5, 80, 1], [211.6, 80, 1]]
df1 = pd.DataFrame(data1, columns = ['V1', 'V2', 'Class'])
X1 = df1.iloc[:, :-1].to_numpy()
y1 = df1.iloc[:, -1].to_numpy()
result1 = linear_svm(X1, y1)

# subset of credit card data
class_0 = credit_df[credit_df['Class'] == 0]
class_1 = credit_df[credit_df['Class'] == 1]

data_0 = class_0.sample(n=15, random_state=42)
data_1 = class_1.sample(n=50, random_state=42)
data_100 = pd.concat([data_1, data_0], axis=0)
X_task6 = data_100.iloc[:, :-1].to_numpy()
y_task6 = data_100.iloc[:, -1].to_numpy()

result = linear_svm(X_task6, y_task6.ravel())


def gaussian_kernel_svm(X: np.ndarray, y: np.ndarray):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.3, random_state=42)

    gaussian_kernel_classifier = svm.SVC(kernel='rbf')
    gaussian_kernel_classifier.fit(X_train, y_train)

    gaussian_kernel_classifier_predictions = gaussian_kernel_classifier.predict(X_test)

    return gaussian_kernel_classifier_predictions, gaussian_kernel_classifier.score(X_test, y_test) * 100


# small data
data1 = [[111.1, 10, -1], [111.2, 20, -1], [111.3, 10, -1], [111.4, 10, -1], [111.5, 10, -1], [211.6, 80, 1],
        [111.4, 10, -1], [111.5, 80, 1], [211.6, 80, 1]]
df1 = pd.DataFrame(data1, columns = ['V1', 'V2', 'Class'])
X1 = df1.iloc[:, :-1].to_numpy()
y1 = df1.iloc[:, -1].to_numpy()
result1 = gaussian_kernel_svm(X1, y1)

# subset of credit card data
class_0 = credit_df[credit_df['Class'] == 0]
class_1 = credit_df[credit_df['Class'] == 1]

data_0 = class_0.sample(n=15, random_state=42)
data_1 = class_1.sample(n=50, random_state=42)
data_100 = pd.concat([data_1, data_0], axis=0)
X_task6 = data_100.iloc[:, :-1].to_numpy()
y_task6 = data_100.iloc[:, -1].to_numpy()

result = gaussian_kernel_svm(X_task6, y_task6.ravel())



if __name__ == '__main__':
    test_task_1_1()
    test_task_1_2()
    test_task_2_1()
    test_task_2_2()
    test_task_2_3()
    test_task_2_4()
    test_task_3_1()
    test_task_3_2()
    test_task_3_3()
    test_task_3_4()
    test_task_5_1()
    test_task_5_2()
    test_task_5_3()
    test_task_5_4()