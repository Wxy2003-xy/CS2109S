import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
from collections import OrderedDict
from utils import load_data, State

def extract_features(state: State) -> np.ndarray:
    board = state.board
    local_status = state.local_board_status

    p1_layer = np.zeros((6, 6), dtype=np.float32)
    p2_layer = np.zeros((6, 6), dtype=np.float32)
    meta_layer = np.zeros((6, 6), dtype=np.float32)
    turn_layer = np.full((6, 6), float(state.fill_num == 1), dtype=np.float32)
    valid_moves_layer = np.zeros((6, 6), dtype=np.float32)
    prev_move_layer = np.zeros((6, 6), dtype=np.float32)
    center_control_layer = np.zeros((6, 6), dtype=np.float32)
    local_near_win_layer = np.zeros((6, 6), dtype=np.float32)
    free_move_layer = np.full((6, 6), float(
        state.prev_local_action is None or
        local_status[state.prev_local_action[0]][state.prev_local_action[1]] != 0
    ), dtype=np.float32)
    meta_center_control_layer = np.zeros((6, 6), dtype=np.float32)
    local_dominance_layer = np.zeros((6, 6), dtype=np.float32)
    meta_near_win_layer = np.zeros((6, 6), dtype=np.float32)

    for i in range(3):
        for j in range(3):
            sub = board[i][j]
            meta_status = local_status[i][j]

            # Local board ownership
            if meta_status == 1:
                local_dominance_layer[i*2:(i+1)*2, j*2:(j+1)*2] = 1.0
            elif meta_status == 2:
                local_dominance_layer[i*2:(i+1)*2, j*2:(j+1)*2] = -1.0

            # Meta board heatmap
            meta_layer[i*2:(i+1)*2, j*2:(j+1)*2] = meta_status / 3.0

            # Center control of local board
            center_val = sub[1][1]
            if center_val != 0:
                center_control_layer[i*2:(i+1)*2, j*2:(j+1)*2] = center_val / 2.0

            # Local 2-in-a-rows
            for player in [1, 2]:
                for line in [
                    [sub[0][0], sub[0][1], sub[0][2]],
                    [sub[1][0], sub[1][1], sub[1][2]],
                    [sub[2][0], sub[2][1], sub[2][2]],
                    [sub[0][0], sub[1][0], sub[2][0]],
                    [sub[0][1], sub[1][1], sub[2][1]],
                    [sub[0][2], sub[1][2], sub[2][2]],
                    [sub[0][0], sub[1][1], sub[2][2]],
                    [sub[0][2], sub[1][1], sub[2][0]],
                ]:
                    if line.count(player) == 2 and line.count(0) == 1:
                        value = 0.5 if player == 1 else -0.5
                        local_near_win_layer[i*2:(i+1)*2, j*2:(j+1)*2] += value

            for k in range(3):
                for l in range(3):
                    val = sub[k][l]
                    x, y = i * 2 + k // 2, j * 2 + l // 2
                    if val == 1:
                        p1_layer[x][y] = 1.0
                    elif val == 2:
                        p2_layer[x][y] = 1.0

    # Meta-board center control
    center_meta_status = local_status[1][1]
    if center_meta_status == 1:
        meta_center_control_layer[2:4, 2:4] = 1.0
    elif center_meta_status == 2:
        meta_center_control_layer[2:4, 2:4] = -1.0

    # Meta-board near win (2 local boards won in a row)
    for player in [1, 2]:
        for line in [
            [local_status[0][0], local_status[0][1], local_status[0][2]],
            [local_status[1][0], local_status[1][1], local_status[1][2]],
            [local_status[2][0], local_status[2][1], local_status[2][2]],
            [local_status[0][0], local_status[1][0], local_status[2][0]],
            [local_status[0][1], local_status[1][1], local_status[2][1]],
            [local_status[0][2], local_status[1][2], local_status[2][2]],
            [local_status[0][0], local_status[1][1], local_status[2][2]],
            [local_status[0][2], local_status[1][1], local_status[2][0]],
        ]:
            if line.count(player) == 2 and line.count(0) == 1:
                value = 0.5 if player == 1 else -0.5
                meta_near_win_layer += value

    # Valid move mask
    for action in state.get_all_valid_actions():
        i, j, k, l = action
        x, y = i * 2 + k // 2, j * 2 + l // 2
        valid_moves_layer[x][y] = 1.0

    # Previous move marker
    if state.prev_local_action is not None:
        i, j = state.prev_local_action
        prev_move_layer[i*2:(i+1)*2, j*2:(j+1)*2] = 1.0

    # Stack all channels
    return np.stack([
        p1_layer,
        p2_layer,
        meta_layer,
        turn_layer,
        valid_moves_layer,
        prev_move_layer,
        center_control_layer,
        local_near_win_layer,
        free_move_layer,
        meta_center_control_layer,
        local_dominance_layer,
        meta_near_win_layer
    ], axis=0)





class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.fc1 = nn.Linear(32 * 2 * 2, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.conv1(x)) 
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 2 * 2)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

# raw_data = load_data()


# data = []
# for state, utility in raw_data:
#     data.append((state, utility))
#     inverted_state = state.invert()
#     inverted_utility = 1.0 - utility
#     data.append((inverted_state, inverted_utility))

data = load_data()

X = np.stack([extract_features(state) for state, _ in data])
y = np.array([[utility] for _, utility in data], dtype=np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

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
