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
minimax_cache = {}

def generate_new_moves(player:int, state:tuple):
    new_states = []
    for i in range(len(state)):
        if state[i] == 0:
            new_state = state[:i] + (player,) + state[i+1:]
            if legal_state(new_state):
                new_states.append(new_state)
    return new_states
def legal_state(state:tuple) -> bool:
    x = 0
    o = 0
    for i in range(9):
        if state[i] == 1: x+=1
        if state[i] == 2: o+=1
        if abs(x - o) > 1: return False
    return True

def local_board_status(board: tuple) -> int:
    win = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6)
    ]
    for i, j, k in win:
        if board[i] == board[j] == board[k] != 0:
            return board[i]
    if all(cell != 0 for cell in board):
        return 3
    return 0

def local_is_terminal(state: tuple) -> bool:
    return local_board_status(state) in [1, 2, 3] 

def local_board_eval(local_board:np.array) -> float:
    status = local_board_status(local_board)
    if status == 1:
        return 1.0
    if status == 2:
        return 0.0
    if status == 3:
        return 0.5
    assert False, "Board is not terminal"

from functools import lru_cache

@lru_cache(maxsize=None)
def local_minimax_enum(state: tuple, player_1:bool, depth:int = 0) -> float:
    if state in minimax_cache:
        return minimax_cache[state]
    if local_is_terminal(state):
        eval_score = local_board_eval(state)
        minimax_cache[state] = eval_score
        return eval_score
    if player_1: 
        best = float('-inf')
        moves = generate_new_moves(1, state)
        if len(moves) == 0: 
            minimax_cache[state] = 0.5
            return 0.5
        for m in moves:
            v = local_minimax_enum(m, False, depth+1)
            best = max(best, v)
        minimax_cache[state] = best
        return best
    if not player_1:
        best = float('inf') 
        moves = generate_new_moves(2, state)
        if len(moves) == 0: 
            minimax_cache[state] = 0.5
            return 0.5
        for m in moves:
            v = local_minimax_enum(m, True, depth+1)
            best = min(best, v)
        minimax_cache[state] = best
        return best


    
'''
def get_all_valid_actions(state: ImmutableState) -> list[Action]:
    if state.prev_local_action is None:
        return _get_all_valid_free_actions(state)
    prev_row, prev_col = state.prev_local_action
    if state.local_board_status[prev_row][prev_col] != 0:
        return _get_all_valid_free_actions(state)
    valid_actions: list[Action] = []
    for i in range(3):
        for j in range(3):
            if state.board[prev_row][prev_col][i][j] == 0:
                valid_actions.append((prev_row, prev_col, i, j))
    return valid_actions
'''
def map_actions_to_board(state:State) -> np.ndarray:
    board_map = np.zeros((3, 3, 3, 3))
    actions = get_all_valid_actions(state._state)
    for a in actions:
        board_map[a[0]][a[1]][a[2]][a[3]] = 1
    return board_map

def player_mask(fill_num:1|2) -> np.ndarray:
    if fill_num == 1:
        return np.zeros((3,3,3,3))
    if fill_num == 2:
        return np.ones((3,3,3,3))
    return np.zeros((3,3,3,3))

def DQN_extract_features(state: State) -> np.ndarray:  # (6, 9, 9)
    imm_state = state._state
    board = imm_state.board.copy()  # Don't modify the original state

    player = imm_state.fill_num

    board_2d = board.reshape(9, 9)  # flatten the 4D board into 2D (9x9)
    norm_board = board_2d.copy() 
    norm_board[norm_board == 2] = -1
    player1_pieces = (board_2d == 1).astype(np.float32)  # mask: 1 where player 1 has pieces
    player2_pieces = (board_2d == 2).astype(np.float32)  # mask: 1 where player 2 has pieces

    player_map = np.full((9, 9), player, dtype=np.float32)  # broadcast player number
    avail_actions = map_actions_to_board(state).astype(np.float32).reshape(9,9)  # 1s for valid moves

    local_map = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            local_board = imm_state.board[i][j]
            local_board_tuple = tuple(local_board.flatten())
            local_map[i][j] = local_minimax_enum(local_board_tuple, player==1, 0)
    broadcasted = np.kron(local_map, np.ones((3, 3), dtype=local_map.dtype))
    print(broadcasted)
    stacked_features = np.stack([norm_board, player1_pieces, player2_pieces, player_map, broadcasted, avail_actions], axis=0)  # shape: (6, 9, 9)
    return stacked_features



def extract_features(state: State) -> np.ndarray:   # (3, 6, 6)
    board = state.board
    p1_layer = np.zeros((6, 6), dtype=np.float32)   
    p2_layer = np.zeros((6, 6), dtype=np.float32)
    meta_layer = np.zeros((6, 6), dtype=np.float32)

    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    val = board[i][j][k][l]
                    x, y = i * 2 + k // 2, j * 2 + l // 2
                    if val == 1:
                        p1_layer[x][y] = 1.0
                    elif val == 2:
                        p2_layer[x][y] = 1.0

    for i in range(3):
        for j in range(3):
            meta_val = state.local_board_status[i][j]
            meta_layer[i*2:(i+1)*2, j*2:(j+1)*2] = meta_val / 3.0

    return np.stack([p1_layer, p2_layer, meta_layer], axis=0) 


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc1 = nn.Linear(16 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))           # (3, 6, 6) -> (16, 6, 6)
        x = self.pool(x)                    # (16, 6, 6) -> (16, 2, 2)
        x = x.view(-1, 16 * 2 * 2)          # flatten to (64,)
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))      # ⬅️ outputs in [-1, 1]
class CNNModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(6, 16, kernel_size=3, padding=1)   # (6, 9, 9) → (16, 9, 9)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # (16, 9, 9) → (32, 9, 9)
        self.fc1 = nn.Linear(32 * 9 * 9, 64)                       # flatten to (N, 2592) → (N, 64)
        self.fc2 = nn.Linear(64, 1)                                # (N, 64) → (N, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return (self.fc2(x))  # Normalize output to [-1, 1]


data = load_data()
print(data[0])
data_arr = np.array(data, dtype=object)
states = data_arr[:, 0]
scores = data_arr[:, 1]

fill_nums = np.array([s.fill_num for s in states])
data_p1 = data_arr[fill_nums == 1]
data_p2 = data_arr[fill_nums == 2]
X = np.stack([DQN_extract_features(state) for state, _ in data])
y = np.array([[utility] for _, utility in data], dtype=np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
mask = ~((y_test == -1.0) | (y_test == 0.0) | (y_test == 1.0)).squeeze()
mask_train = ~((y_train == -1.0) | (y_train == 0.0) | (y_train == 1.0)).squeeze()

# Apply mask
X_train = X_train[mask_train]
y_train = y_train[mask_train]
X_test = X_test[mask]
y_test = y_test[mask]
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
# flatten for DQN extract
# X_train_tensor = X_train_tensor.view(X_train_tensor.size(0), -1)
# X_test_tensor = X_test_tensor.view(X_test_tensor.size(0), -1)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel2().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

print("🚀 Training on:", device)


for epoch in tqdm(range(50), desc="Training"):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0 or epoch == 0:
        model.eval()
        with torch.no_grad():
            train_preds = model(X_train_tensor.to(device))
            test_preds = model(X_test_tensor.to(device))
            train_loss = criterion(train_preds, y_train_tensor.to(device)).item()
            test_loss = criterion(test_preds, y_test_tensor.to(device)).item()

        print(f"Epoch {epoch + 1:>3}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")

model.eval()
with torch.no_grad():
    train_preds = model(X_train_tensor.to(device))
    test_preds = model(X_test_tensor.to(device))
    train_loss = criterion(train_preds, y_train_tensor.to(device)).item()
    test_loss = criterion(test_preds, y_test_tensor.to(device)).item()

print("✅ Final Training Loss:", train_loss)
print("✅ Final Test Loss:", test_loss)

state_dict = model.state_dict()
hardcoded_weights = OrderedDict({k: v.cpu().numpy().tolist() for k, v in state_dict.items()})

with open("model.json", "w") as f:
    json.dump(hardcoded_weights, f, indent=4)

import matplotlib.pyplot as plt

with torch.no_grad():
    y_pred_test = model(X_test_tensor.to(device)).cpu().numpy()

plt.scatter(y_test, y_pred_test, alpha=0.3)
plt.plot([-1, 1], [-1, 1], 'r--')  # ideal line
plt.xlabel("True utility")
plt.ylabel("Predicted utility")
plt.title("Test Predictions vs True Values")
plt.grid(True)
plt.show()