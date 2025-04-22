# Run the following cell to import utilities

import numpy as np
import time

from utils import State, Action, load_data, LocalBoardAction, ImmutableState
data = load_data()
assert len(data) == 80000
for state, value in data[:1]:
    print(state)
    print(f"Value = {value}\n\n")
# data prep
data_arr = np.array(data, dtype=object)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LassoCV, LinearRegression
import numpy as np

data_arr = np.array(data)
get_fill_num = np.vectorize(lambda x: x.fill_num)
fill_nums = get_fill_num(data_arr[:, 0])
mask = fill_nums == 1
data1 = data_arr[mask]

mask = fill_nums == 2
data2 = data_arr[mask]


def heuristic_local_eval(board, hor_weight, diag_weight, denom):
    winning_path_1 = 0
    winning_path_2 = 0
    for i in range(3):
        count_1 = 0
        count_2 = 0
        for j in range(3):
            if board[i][j] == 1: count_1+=1
            if board[i][j] == 2: count_2+=1
        if count_1 != 0 and count_2 != 0:
            continue
        if count_2 == 0 and count_1 != 0:
            winning_path_1 += hor_weight * count_1
        if count_1 == 0 and count_2 != 0:
            winning_path_2 += hor_weight * count_2
    for i in range(3):
        count_1 = 0
        count_2 = 0
        for j in range(3):
            if board[j][i] == 1: count_1+=1
            if board[j][i] == 2: count_2+=1
        if count_1 != 0 and count_2 != 0:
            continue
        if count_2 == 0 and count_1 != 0:
            winning_path_1 += hor_weight * count_1
        if count_1 == 0 and count_2 != 0:
            winning_path_2 += hor_weight * count_2
    for i in range(3):
        count_1 = 0
        count_2 = 0
        if board[i][i] == 1: count_1+=1
        if board[i][i] == 2: count_2+=1
    if count_1 != 0 and count_2 != 0: pass
    if count_2 == 0 and count_1 != 0:
        winning_path_1 += diag_weight * count_1
    if count_1 == 0 and count_2 != 0:
        winning_path_2 += diag_weight * count_2
    for i in range(3):
        count_1 = 0
        count_2 = 0
        if board[i][2-i] == 1: count_1+=1
        if board[i][2-i] == 2: count_2+=1
    if count_1 != 0 and count_2 != 0: pass
    if count_2 == 0 and count_1 != 0:
        winning_path_1 += diag_weight * count_1
    if count_1 == 0 and count_2 != 0:
        winning_path_2 += diag_weight * count_2

    return (winning_path_1 - winning_path_2) / denom

def global_board_eval(board):
    return np.sum(board)

def separate_state_score(data):
    states_1 = data[:, 0]
    scores_1 = data[:, 1]
    return parameterize_state(states_1), scores_1
def parameterize_state(filtered_states) -> np.array:
    engineered_features = []
    hor_weight = 1
    diag_weight = 1
    denom = 8
    for state in filtered_states:
        flattened_board = state.board.flatten()
        heuristic_features = []
        for i in range(3):
            for j in range(3):
                heuristic_feature = heuristic_local_eval(state.board[i][j], hor_weight, diag_weight, denom)
                heuristic_features.append(heuristic_feature)
        global_feature = global_board_eval(state.board)
        features = np.concatenate([heuristic_features])
        engineered_features.append(features)
    return np.array(engineered_features)

training_set_20, test_set_20 = train_test_split(data1, test_size=0.2, random_state=0)

train_state_20, train_score_20 = separate_state_score(training_set_20)

test_state_20, test_score_20 = separate_state_score(test_set_20)
train_state_20[train_state_20 == 2] = -1
test_state_20[test_state_20 == 2] = -1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_state_20 = scaler.fit_transform(train_state_20)
test_state_20 = scaler.transform(test_state_20)


# reg_20 = LinearRegression().fit(train_state_20, train_score_20)
# score_reg = reg_20.score(test_state_20, test_score_20)
# print(score_reg)



from sklearn import model_selection, svm

def gaussian_kernel_svr(X: np.ndarray, y: np.ndarray):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.3, random_state=42)
    
    gaussian_kernel_regressor = svm.SVR(kernel='rbf')
    gaussian_kernel_regressor.fit(X_train, y_train)
    
    predictions = gaussian_kernel_regressor.predict(X_test)
    score = gaussian_kernel_regressor.score(X_test, y_test) * 100
    
    return predictions, score

# print(gaussian_kernel_svr(train_state_20, train_score_20))

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'epsilon': [0.01, 0.1, 0.2, 0.5]
}

grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5)
grid_search.fit(train_state_20, train_score_20)
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation R²:", grid_search.best_score_)

