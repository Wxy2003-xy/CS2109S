import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LassoCV, LinearRegression
from utils import State, Action, ImmutableState, load_data, is_terminal, get_all_valid_actions, change_state, terminal_utility, get_random_valid_action, is_valid_action
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
from collections import OrderedDict
from utils import load_data, State, get_all_valid_actions

def is_local_board_terminal(board: np.ndarray, player: int) -> int:
    """
    Checks if the given player wins in a 3x3 Tic Tac Toe board.

    Args:
        board: np.ndarray of shape (3, 3)
        player: 1 or 2

    Returns:
        1 if the given player wins,
        0 if it's a draw,
        -1 if the game is not over or player didn't win.
    """
    for i in range(3):
        if np.all(board[i, :] == player) or np.all(board[:, i] == player):
            return 1
    if np.all(np.diag(board) == player) or np.all(np.diag(np.fliplr(board)) == player):
        return 1
    if np.all(board != 0):
        return 0
    return -1

local_board_mask = np.array([[0.2, 0.17, 0.2],
                             [0.17, 0.22, 0.17],
                             [0.2, 0.17, 0.2]])

def check_win_condition(board):
    """Checks winner in a flat 1D list of 9 cells"""
    win_lines = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # cols
        [0, 4, 8], [2, 4, 6]              # diagonals
    ]
    for a in [1, -1]:
        for line in win_lines:
            if sum(board[i] for i in line) == a * 3:
                return a
    return 0
def real_evaluate_square(pos):
    evaluation = 0
    points = [0.2, 0.17, 0.2, 0.17, 0.22, 0.17, 0.2, 0.17, 0.2]

    for i, p in enumerate(pos):
        evaluation -= p * points[i]

    # reward lines made by player 1
    if sum(pos[i] for i in [0,1,2]) == 2: evaluation -= 6
    if sum(pos[i] for i in [3,4,5]) == 2: evaluation -= 6
    if sum(pos[i] for i in [6,7,8]) == 2: evaluation -= 6
    if sum(pos[i] for i in [0,3,6]) == 2: evaluation -= 6
    if sum(pos[i] for i in [1,4,7]) == 2: evaluation -= 6
    if sum(pos[i] for i in [2,5,8]) == 2: evaluation -= 6
    if sum(pos[i] for i in [0,4,8]) == 2: evaluation -= 7
    if sum(pos[i] for i in [2,4,6]) == 2: evaluation -= 7

    # punish threats from player -1 (opponent)
    a = -1
    for line in [[0,1,2], [3,4,5], [6,7,8], [0,3,6], [1,4,7], [2,5,8], [0,4,8], [2,4,6]]:
        for i in range(3):
            subset = [pos[line[j]] for j in range(3) if j != i]
            if subset.count(a) == 2 and pos[line[i]] == -a:
                evaluation -= 9

    # reward threats from player 1
    a = 1
    for line in [[0,1,2], [3,4,5], [6,7,8], [0,3,6], [1,4,7], [2,5,8], [0,4,8], [2,4,6]]:
        for i in range(3):
            subset = [pos[line[j]] for j in range(3) if j != i]
            if subset.count(a) == 2 and pos[line[i]] == -a:
                evaluation += 9

    evaluation -= check_win_condition(pos) * 12
    return evaluation

def flatten_subboard(board3x3):
    return board3x3.flatten().tolist()

def evaluate_game(position_4d, prev):
    evale = 0
    main_bd = []
    evaluator_mul = [1.4, 1, 1.4, 1, 1.75, 1, 1.4, 1, 1.4]

    for i in range(3):
        for j in range(3):
            index = 3 * i + j  # flatten the 3x3 meta-board index to 0-8
            board_ij = flatten_subboard(position_4d[i, j])
            sq_eval = real_evaluate_square(board_ij)
            evale += sq_eval * 1.5 * evaluator_mul[index]

            if (i, j) == tuple(prev):
                evale += sq_eval * evaluator_mul[index]

            tmp_ev = check_win_condition(board_ij)
            evale -= tmp_ev * evaluator_mul[index]

            main_bd.append(tmp_ev)

    evale -= check_win_condition(main_bd) * 5000
    evale += real_evaluate_square(main_bd) * 150
    return evale



def minimax(state: ImmutableState, depth: int, alpha: float, beta: float, maximizing_player: bool, player: int) -> float:
    if is_terminal(state):
        return terminal_utility(state)
    if depth == 0:
        score = evaluation_game(state)
        print(state, score)
        return score

    actions = get_all_valid_actions(state)
    if maximizing_player:
        max_eval = float("-inf")
        for action in actions:
            new_state = change_state(state, action)
            eval = minimax(new_state, depth - 1, alpha, beta, False, player)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float("inf")
        for action in actions:
            new_state = change_state(state, action)
            eval = minimax(new_state, depth - 1, alpha, beta, True, player)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


class StudentAgent:
    def __init__(self):
        """Instantiates your agent.
        """

    def choose_action(self, state: State) -> Action:
        player = state._state.fill_num
        if state._state.prev_local_action == None:
            return (1, 1, 1, 1)
        valid_actions = get_all_valid_actions(state._state)

        best_score = float("-inf")
        best_action = valid_actions[0]

        for action in valid_actions:
            new_state = change_state(state._state, action)
            score = minimax(new_state, depth=2, alpha=float("-inf"), beta=float("inf"), maximizing_player=False, player=player)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

data = load_data()
assert len(data) == 80000
for state, value in data[:3]:
    print(state)
    print(f"Value = {value}\n\n")

