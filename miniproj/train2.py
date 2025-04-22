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

def DQN_extract_features(state: State) -> np.ndarray:  # (4, 9, 9)
    imm_state = state._state
    board = imm_state.board.copy()  # Don't modify the original state
    player = imm_state.fill_num

    board_2d = board.reshape(9, 9)  # flatten the 4D board into 2D (9x9)

    player1_pieces = (board_2d == 1).astype(np.float32)  # mask: 1 where player 1 has pieces
    player2_pieces = (board_2d == 2).astype(np.float32)  # mask: 1 where player 2 has pieces

    player_map = np.full((9, 9), player, dtype=np.float32)  # broadcast player number
    avail_actions = map_actions_to_board(state).astype(np.float32).reshape(9,9)  # 1s for valid moves
    stacked_features = np.stack([player1_pieces, player2_pieces, player_map, avail_actions], axis=0)  # shape: (4, 9, 9)
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
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc1 = nn.Linear(128, 128) #
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x)) 
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 2 * 2)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
class CNNModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 9 * 9, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
# 81*81*4 -> 81*81 -> 81 -> 9 -> 1
# no conv 
class CNNModelSmall_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=1)    # (4, 9, 9) → (16, 9, 9)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)   # (16, 9, 9) → (32, 9, 9)
        self.pool = nn.AdaptiveAvgPool2d((3, 3))                   # down to (32, 3, 3)
        self.fc1 = nn.Linear(32 * 3 * 3, 64)                       # 288 → 64
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # (B, 4, 9, 9) → (B, 16, 9, 9)
        x = F.relu(self.conv2(x))   # (B, 16, 9, 9) → (B, 32, 9, 9)
        x = self.pool(x)            # (B, 32, 9, 9) → (B, 32, 3, 3)
        x = x.view(x.size(0), -1)   # (B, 32*3*3)
        x = F.relu(self.fc1(x))     # → (B, 64)
        return self.fc2(x)          # → (B, 1)

class DQNNModel(nn.Module):
    def __init__(self): 
        super().__init__()
        self.f1 = nn.Linear(9*9*4, 81)
        self.f2 = nn.Linear(9*9, 9)
        self.f3 = nn.Linear(9, 3)
        self.f4 = nn.Linear(3, 1)

    def forward(self, x):
        x = F.tanh(self.f1(x)) 
        x = F.tanh(self.f2(x))
        x = F.tanh(self.f3(x))
        x = self.f4(x)
        return x   

data = load_data()
print(data[0])
data_arr = np.array(data, dtype=object)
states = data_arr[:, 0]
scores = data_arr[:, 1]

fill_nums = np.array([s.fill_num for s in states])
data_p1 = data_arr[fill_nums == 1]
data_p2 = data_arr[fill_nums == 2]

# Labels for data_p2
XX = np.stack([DQN_extract_features(state) for state, _ in data_p2])
y_p2 = np.array([[utility] for _, utility in data_p2], dtype=np.float32)

# Train/test split for player 2
X2_train, X2_test, y2_train, y_test = train_test_split(XX, y_p2, test_size=0.2, random_state=42)

# Convert to tensors
X_train_tensor = torch.tensor(X2_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y2_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X2_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModelSmall_1().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("🚀 Training on:", device)


for epoch in tqdm(range(100), desc="Training"):
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

with open("model2.json", "w") as f:
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

# ✅ Final Training Loss: 0.06117632985115051
# ✅ Final Test Loss: 0.3930588960647583

# ✅ Final Training Loss: 0.06513166427612305
# ✅ Final Test Loss: 0.4262593984603882