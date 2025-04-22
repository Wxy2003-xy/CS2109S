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
def subboard_eval_sigmoid(raw_score):
    return 2 * (1 / (1 + np.exp(-raw_score))) - 1
def subboard_eval(board, hor_weight, diag_weight): 
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
    return (winning_path_1 - winning_path_2)

def symmetry_score(global_board):
    flipped_lr = np.fliplr(global_board)
    flipped_ud = np.flipud(global_board)
    rotated = np.rot90(global_board, 2)
    mirror_diag = np.transpose(global_board)

    sims = [
        1 - np.mean(np.abs(global_board - flipped_lr)),
        1 - np.mean(np.abs(global_board - flipped_ud)),
        1 - np.mean(np.abs(global_board - rotated)),
        1 - np.mean(np.abs(global_board - mirror_diag))
    ]
    
    return np.mean(sims)

def global_board_approx(board, threshold=0.3):
    result = np.zeros_like(board)
    for i in range(3):
        for j in range(3):
            if board[i][j] - threshold < -1:
                result[i][j] = -1
            elif board[i][j] + threshold > 1:
                result[i][j] = 1
            else:
                result[i][j] = 0
    return result

def evaluation_function(state:ImmutableState,
                        hor_weight, diag_weight, center_weight, corner_weight,
                        current_board_weight,
                        global_approx_threshold,
                        global_hor_weight,global_diag_weight, global_board_weight,
                        global_center_weight, global_corner_weight,
                        symmetry_weight):
    print(state.prev_local_action)
    board = state.board 
    prev = state.prev_local_action 
    player = state.fill_num 
    local_board_scores = np.zeros((3,3))
    center_control_1 = 0
    center_control_2 = 0
    corner_control_1 = 0
    corner_control_2 = 0
    for i in range(3):
        for j in range(3):
            subboard = board[i][j]
            local_board_scores[i][j] = subboard_eval(subboard, hor_weight, diag_weight)
            if i == prev[0] and j == prev[1]:
                local_board_scores[i][j] *= current_board_weight
            for a in range(3):
                for b in range(3):
                    if a == 1 and b == 1:
                        if subboard[a][b] == 1:
                            center_control_1+=1
                        if subboard[a][b] == 2:
                            center_control_2+=1
                    if a == b or abs(a - b) == 2:
                        if subboard[a][b] == 1:
                            corner_control_1+=1
                        if subboard[a][b] == 2:
                            corner_control_2+=1
    total_center_control = center_weight*(center_control_1 - center_control_2)
    total_corner_control = corner_weight*(corner_control_1 - corner_control_2)
    raw_symmetry_score = symmetry_score(local_board_scores)
    global_winning_board = global_board_approx(local_board_scores, global_approx_threshold)
    global_center_control = global_center_weight*global_winning_board[1][1]
    global_corner_control = global_corner_weight * (
        global_winning_board[0][0] + global_winning_board[0][2] +
        global_winning_board[2][0] + global_winning_board[2][2]
    )
    global_score = global_board_weight*subboard_eval(global_winning_board, global_hor_weight, global_diag_weight)

    pre_sym_score = total_center_control + total_corner_control + global_center_control + global_corner_control + global_score

    if player == 1:
        if pre_sym_score >= 0:
            return pre_sym_score
        else:
            return np.tanh(pre_sym_score + symmetry_weight*raw_symmetry_score)
    else:
        if pre_sym_score > 0:
            return np.tanh(pre_sym_score - raw_symmetry_score)
        else:
            return pre_sym_score


def minimax(state: ImmutableState, depth: int, alpha: float, beta: float, maximizing_player: bool, player: int) -> float:
    if is_terminal(state):
        return terminal_utility(state)
    
    if depth == 0:
        return evaluation_function(
            state,
            hor_weight=1.0,
            diag_weight=1.2,
            center_weight=0.3,
            corner_weight=0.2,
            current_board_weight=1.1,
            global_approx_threshold=0.3,
            global_hor_weight=1.0,
            global_diag_weight=1.2,
            global_board_weight=1.5,
            global_center_weight=0.4,
            global_corner_weight=0.3,
            symmetry_weight=0.2
        )

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
# ============================================================================================================
def local_board_position(prev):
    if prev == (1, 1):
        return 0    # center board
    if prev[0] == prev[1] or abs(prev[0] - prev[1] == 2):
        return 1    # corner board
    return 2        # edge board

def is_local_board_empty(board):
    return np.all(board == 0)

def is_local_board_terminal(board: np.ndarray) -> int:
    """
    Checks if a 3x3 Tic Tac Toe board is in a terminal state.
    Returns:
        1 if Player 1 wins,
        2 if Player 2 wins,
        0 if it's a draw,
        -1 if the game is not over.
    """
    for i in range(3):
        if np.all(board[i, :] == 1) or np.all(board[:, i] == 1):
            return 1
        if np.all(board[i, :] == 2) or np.all(board[:, i] == 2):
            return 2
    if np.all(np.diag(board) == 1) or np.all(np.diag(np.fliplr(board)) == 1):
        return 1
    if np.all(np.diag(board) == 2) or np.all(np.diag(np.fliplr(board)) == 2):
        return 2
    if np.all(board != 0):
        return 0
    return -1


def local_board_eval(board, player, params):
# params[2consec, 2non_consec, ]
    # terminal case
    if is_local_board_terminal(board) == player:
        if player == 2:
            return -1
        return 1
    if is_local_board_terminal(board) == 0:
        return 0
    # winning path
    winning_hor = 0
    winning_ver = 0
    for i in range(3):
        # 110: block will send to corner (good)
        if board[i][0] == player and board[i][1] == player and board[i][2] == 0:
            winning_hor += params[0]
        elif board[i][1] == player and board[i][2] == player and board[i][0] == 0:
            winning_hor += params[0]
        # 101: block will send to edge (not as good)
        elif board[i][0] == player and board[i][2] == player and board[i][1] == 0:
            winning_hor += params[1]
    for i in range(3):
        # 110 vertical
        if board[0][i] == player and board[1][i] == player and board[2][i] == 0:
            winning_ver += params[0]
        elif board[1][i] == player and board[2][i] == player and board[0][i] == 0:
            winning_ver += params[0]
        # 101 vertical
        elif board[0][i] == player and board[2][i] == player and board[1][i] == 0:
            winning_ver += params[1]

def eval_early_game(state: ImmutableState, params:np.ndarray):
    board = state.board
    prev = state.prev_local_action
    player = state.fill_num
    # meta
    # sending enemy to empty board (good)
    is_new_local_board_empty = 0
    if is_new_local_board_empty(board[prev[0]][prev[1]]):
        is_new_local_board_empty = 1
    # sending opp to center board (very bad)
    is_new_local_board_center = 0
    if prev == (1, 1):
        is_new_local_board_center = 1
    # sending opp to corner board (bad)
    is_new_local_board_corner = 0
    if local_board_position(prev) == 1:
        is_new_local_board_corner = 1
    # sending opp to edge board (good)
    is_new_local_board_edge = 0
    if local_board_position(prev) == 2:
        is_new_local_board_edge = 1
    
    # local board stats







class StudentAgent:
    def __init__(self):
        self.depth = 0
        self.first_player = False

    def choose_action(self, state: State) -> Action:
        prev = state._state.prev_local_action
        state = state._state
        valid_actions = get_all_valid_actions(state)
        if self.depth == 0:
            if len(valid_actions) > 9:
                self.first_player = True
                self.depth += 1
                return (1, 1, 1, 1) # book move first 1
            if len(valid_actions) <= 9:
                if local_board_position(prev) == 0:
                    if is_valid_action(state, (1, 1, 1, 1)):
                        self.depth += 1
                        return (1, 1, 1, 1)
                    else:
                        self.depth += 1
                        return (1, 1, 0, 0)
                if local_board_position(prev) == 1:
                    return 