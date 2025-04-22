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

def extract_features(state: State, hyperparams) -> np.ndarray:
    meta_status_const = hyperparams[0]
    central_val_const = hyperparams[1]
    meta_cor_const = hyperparams[2]
    meta_edge_const = hyperparams[3]
    board = state.board
    local_status = state.local_board_status
    # state parameters
    p1_layer = np.zeros((9,9), dtype=np.float32)
    p2_layer = np.zeros((9,9), dtype=np.float32)
    meta_layer = np.zeros((9, 9), dtype=np.float32)
    turn_layer = np.full((9, 9), float(state.fill_num == 1), dtype=np.float32)
    # local boards status
    local_near_win_layer = np.zeros((9, 9), dtype=np.float32)
    center_control_layer = np.zeros((9, 9), dtype=np.float32)
    local_dominance_layer = np.zeros((9, 9), dtype=np.float32)
    # move related 
    valid_moves_layer = np.zeros((9, 9), dtype=np.float32)
    prev_move_layer = np.zeros((9, 9), dtype=np.float32)
    free_move_layer = np.full((9, 9), float(
        state.prev_local_action is None or
        local_status[state.prev_local_action[0]][state.prev_local_action[1]] != 0
    ), dtype=np.float32)
    # meta board control logic
    meta_center_control_layer = np.zeros((9, 9), dtype=np.float32)
    meta_corner_control_layer = np.zeros((9,9), dtype=np.float32)
    meta_edge_control_layer = np.zeros((9,9), dtype=np.float32)
    meta_near_win_layer = np.zeros((9, 9), dtype=np.float32)
    # strategy logic
    target_pressure_layer = np.zeros((9, 9), dtype=np.float32)
    opp_waste_move_layer = np.zeros((9,9), dtype=np.float32)

    # Meta-board corner control
    corner_indices = [(0, 0), (0, 2), (2, 0), (2, 2)]
    for i, j in corner_indices:
        corner_meta_status = local_status[i][j]
        if corner_meta_status == 1:
            meta_corner_control_layer[i*3:(i+1)*3, j*3:(j+1)*3] = 1.0 / meta_cor_const
        elif corner_meta_status == 2:
            meta_corner_control_layer[i*3:(i+1)*3, j*3:(j+1)*3] = -1.0 / meta_cor_const
    # Meta-board edge control
    edge_indices = [(0, 1), (1, 0), (1, 2), (2, 1)]
    for i, j in edge_indices:
        edge_meta_status = local_status[i][j]
        if edge_meta_status == 1:
            meta_edge_control_layer[i*3:(i+1)*3, j*3:(j+1)*3] = 1.0 / meta_edge_const
        elif edge_meta_status == 2:
            meta_edge_control_layer[i*3:(i+1)*3, j*3:(j+1)*3] = -1.0 / meta_edge_const

    for i in range(3):
        for j in range(3):
            sub = board[i][j]
            meta_status = local_status[i][j]

            # Local board ownership
            if meta_status == 1:
                local_dominance_layer[i*3:(i+1)*3, j*3:(j+1)*3] = 1.0
            elif meta_status == 2:
                local_dominance_layer[i*3:(i+1)*3, j*3:(j+1)*3] = -1.0

            # Meta board heatmap
            meta_layer[i*3:(i+1)*3, j*3:(j+1)*3] = meta_status / meta_status_const

            # Center control of local board
            center_val = sub[1][1]
            if center_val != 0:
                center_control_layer[i*3:(i+1)*3, j*3:(j+1)*3] = center_val / central_val_const
            count_p1 = np.sum(sub == 1)
            count_p2 = np.sum(sub == 2)
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
                        value = 0.2 if player == 1 else -0.2
                        local_near_win_layer[i*3:(i+1)*3, j*3:(j+1)*3] += value
                        # Overcommit logic: if opponent made >3 moves and still not won
                        if player == 2 and count_p2 > 3 and local_status[i][j] == 0:
                            opp_waste_move_layer[i*3:(i+1)*3, j*3:(j+1)*3] += -0.1
                        elif player == 1 and count_p1 > 3 and local_status[i][j] == 0:
                            opp_waste_move_layer[i*3:(i+1)*3, j*3:(j+1)*3] += 0.1

            for k in range(3):
                for l in range(3):
                    val = sub[k][l]
                    x = i*3 + k
                    y = j*3 + l
                    if val == 1:
                        p1_layer[x][y] = 1.0
                    elif val == 2:
                        p2_layer[x][y] = 1.0
    # Meta-board center control
    center_meta_status = local_status[1][1]
    if center_meta_status == 1:
        meta_center_control_layer[3:6, 3:6] = 1.0
    elif center_meta_status == 2:
        meta_center_control_layer[3:6, 3:6] = -1.0

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
        x, y = i * 3 + k, j * 3 + l
        valid_moves_layer[x][y] = 1.0

    # Previous move marker
    if state.prev_local_action is not None:
        i, j = state.prev_local_action
        prev_move_layer[i*3:(i+1)*3, j*3:(j+1)*3] = 1.0

    if state.prev_local_action is not None:
        target_i, target_j = state.prev_local_action
        sub = board[target_i][target_j]
        total_moves = np.sum((sub == 1) | (sub == 2))
        pressure_score = 1 / (1 + np.exp(- (total_moves - 4.5)))

        target_pressure_layer[target_i*3:(target_i+1)*3, target_j*3:(target_j+1)*3] = pressure_score

    return np.stack([
        # board state
        p1_layer,
        p2_layer,
        meta_layer,
        turn_layer,

        # local board status
        local_near_win_layer,
        center_control_layer,
        local_dominance_layer,

        # move-related
        valid_moves_layer,
        prev_move_layer,
        free_move_layer,

        # meta board control
        meta_center_control_layer,
        meta_corner_control_layer,
        meta_edge_control_layer,
        meta_near_win_layer,

        # strategic
        target_pressure_layer,
        opp_waste_move_layer,
    ], axis=0)



class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool2d((3, 3))
        self.fc1 = nn.Linear(32 * 3 * 3, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = x.view(-1, 32 * 3 * 3)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


raw_data = load_data()

hyperparams = [1.0, 1.0, 3.0, 3.5]
data = []
for state, utility in raw_data:
    data.append((state, utility))
    inverted_state = state.invert()
    inverted_utility = 1.0 - utility
    data.append((inverted_state, inverted_utility))

data = load_data()
X = np.stack([extract_features(state, hyperparams) for state, _ in data])
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
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

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

with open("model4.json", "w") as f:
    json.dump(hardcoded_weights, f, indent=4)


# import itertools
# import csv
# import torch
# import numpy as np
# from sklearn.model_selection import train_test_split
# from torch.utils.data import TensorDataset, DataLoader
# from tqdm import tqdm

# # Define your CNNModel and extract_features(state, hyperparams) before this block

# # Define search space
# meta_range = [1.0, 1.5, 2.0, 2.5, 3.0]
# center_range = [1.0, 1.5, 2.0, 2.5, 3.0]
# meta_cor = [1.0, 1.5, 2.0, 2.5, 3.0]
# meta_ed = [1.0, 1.5, 2.0, 2.5, 3.0]
# param_grid = list(itertools.product(meta_range, center_range, meta_cor, meta_ed))  # limit to 10 combos
# import random

# raw_data = random.sample(load_data(), 4000)

# # Load reduced data for efficiency
# raw_data = load_data()[:4000]
# random.seed(42)
# raw_data = random.sample(load_data(), 4000)

# # Prepare CSV logger
# csv_file = "tuning_results.csv"
# with open(csv_file, "w", newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(["meta_const", "center_const", "train_loss", "test_loss"])

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# for meta_const, center_const, meta_cor, meta_ed in tqdm(param_grid, desc="Hyperparam Tuning"):

#     hyperparams = [meta_const, center_const, meta_cor, meta_ed]
#     X = np.stack([extract_features(state, hyperparams) for state, _ in raw_data])
#     y = np.array([[utility] for _, utility in raw_data], dtype=np.float32)

#     # Split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
#     y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
#     X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
#     y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

#     train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)

#     # Init model
#     model = CNNModel().to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#     criterion = torch.nn.MSELoss()

#     # Train for a few epochs
#     for epoch in range(50):
#         model.train()
#         for xb, yb in train_loader:
#             xb, yb = xb.to(device), yb.to(device)
#             optimizer.zero_grad()
#             loss = criterion(model(xb), yb)
#             loss.backward()
#             optimizer.step()

#     # Eval
#     model.eval()
#     with torch.no_grad():
#         train_preds = model(X_train_tensor.to(device))
#         test_preds = model(X_test_tensor.to(device))
#         train_loss = criterion(train_preds, y_train_tensor.to(device)).item()
#         test_loss = criterion(test_preds, y_test_tensor.to(device)).item()

#     # Log results
#     with open(csv_file, "a", newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow([meta_const, center_const, meta_cor, meta_ed, round(train_loss, 5), round(test_loss, 5)])

'''
Model: 16,9,9
Epoch   1: Train Loss = 0.2013, Test Loss = 0.2020
Training:  18%|███████████████████████████████████▊                                                                                                                                                                   | 9/50 [05:00<22:37, 33.11s/it]Epoch  10: Train Loss = 0.1235, Test Loss = 0.1321
Training:  38%|███████████████████████████████████████████████████████████████████████████▏                                                                                                                          | 19/50 [10:24<16:34, 32.07s/it]Epoch  20: Train Loss = 0.1025, Test Loss = 0.1243
Training:  58%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                                                                                   | 29/50 [15:51<11:17, 32.25s/it]Epoch  30: Train Loss = 0.0886, Test Loss = 0.1248
Training:  78%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                                           | 39/50 [18:42<02:54, 15.87s/it]Epoch  40: Train Loss = 0.0743, Test Loss = 0.1267
Training:  98%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████    | 49/50 [21:19<00:15, 15.61s/it]Epoch  50: Train Loss = 0.0657, Test Loss = 0.1313
Training: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [21:37<00:00, 25.95s/it]
✅ Final Training Loss: 0.065719835460186
✅ Final Test Loss: 0.13130368292331696
'''