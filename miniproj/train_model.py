import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
from collections import OrderedDict
from utils import State, Action, ImmutableState, load_data, is_terminal, get_all_valid_actions, change_state, terminal_utility, get_random_valid_action, is_valid_action

def extract_features(state: State, hyperparams) -> np.ndarray:
    meta_status_const = hyperparams[0]
    central_val_const = hyperparams[1]
    meta_cor_const = hyperparams[2]
    meta_edge_const = hyperparams[3]
    A, B, C, D, E = hyperparams[4:9]
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
    total_placed = 0
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
            # Local 2-in-a-rows
            for player in [1, 2]:
                for line_index, line in enumerate([
                    [sub[0][0], sub[0][1], sub[0][2]],  # Row 0
                    [sub[1][0], sub[1][1], sub[1][2]],  # Row 1
                    [sub[2][0], sub[2][1], sub[2][2]],  # Row 2
                    [sub[0][0], sub[1][0], sub[2][0]],  # Col 0
                    [sub[0][1], sub[1][1], sub[2][1]],  # Col 1
                    [sub[0][2], sub[1][2], sub[2][2]],  # Col 2
                    [sub[0][0], sub[1][1], sub[2][2]],  # Diag \
                    [sub[0][2], sub[1][1], sub[2][0]],  # Diag /
                ]):
                    if line.count(player) == 2 and line.count(0) == 1:
                        if line_index < 6:
                            indices = [
                                (0, line_index % 3), (1, line_index % 3), (2, line_index % 3)
                            ] if line_index >= 3 else [
                                (line_index, 0), (line_index, 1), (line_index, 2)
                            ]
                            pos_vals = [sub[i][j] for i, j in indices]
                            empty_pos = [k for k, v in enumerate(pos_vals) if v == 0]
                            occupied = [k for k in range(3) if k != empty_pos[0]]
                            if {occupied[0], occupied[1]} == {0, 2}:
                                reward = B
                            elif {occupied[0], occupied[1]} in [{0, 1}, {1, 2}]:
                                reward = A
                            else:
                                reward = C
                        else:
                            if line_index == 6:
                                indices = [(0, 0), (1, 1), (2, 2)]
                            else:
                                indices = [(0, 2), (1, 1), (2, 0)]
                            pos_vals = [sub[i][j] for i, j in indices]
                            occupied = [k for k, v in enumerate(pos_vals) if v == player]
                            if 1 in occupied:
                                reward = D
                            else:
                                reward = E

                        value = reward if player == 1 else -reward
                        local_near_win_layer[i*3:(i+1)*3, j*3:(j+1)*3] += value

                        # Overcommit logic
                        if player == 2 and count_p2 > 3 and local_status[i][j] == 0:
                            opp_waste_move_layer[i*3:(i+1)*3, j*3:(j+1)*3] += -0.1
                        elif player == 1 and count_p1 > 3 and local_status[i][j] == 0:
                            opp_waste_move_layer[i*3:(i+1)*3, j*3:(j+1)*3] += 0.1


            for k in range(3):
                for l in range(3):
                    if board[i][j][k][l] != 0:
                        total_placed += 1
                    val = sub[k][l]
                    x = i*3 + k
                    y = j*3 + l
                    if val == 1:
                        p1_layer[x][y] = 1.0
                    elif val == 2:
                        p2_layer[x][y] = 1.0
    total_placed = total_placed / 81
    game_progress = np.full((9, 9), total_placed, dtype=np.float32)
    # Meta-board center control

    
    center_meta_status = local_status[1][1]
    if center_meta_status == 1:
        meta_center_control_layer[3:6, 3:6] = 1.0
    elif center_meta_status == 2:
        meta_center_control_layer[3:6, 3:6] = -1.0

    # Meta-board near win (2 local boards won in a row)
    for player in [1, 2]:
        for line_index, line in enumerate([
            [local_status[0][0], local_status[0][1], local_status[0][2]],  # Row 0
            [local_status[1][0], local_status[1][1], local_status[1][2]],  # Row 1
            [local_status[2][0], local_status[2][1], local_status[2][2]],  # Row 2
            [local_status[0][0], local_status[1][0], local_status[2][0]],  # Col 0
            [local_status[0][1], local_status[1][1], local_status[2][1]],  # Col 1
            [local_status[0][2], local_status[1][2], local_status[2][2]],  # Col 2
            [local_status[0][0], local_status[1][1], local_status[2][2]],  # Diag \
            [local_status[0][2], local_status[1][1], local_status[2][0]],  # Diag /
        ]):
            if line.count(player) == 2 and line.count(0) == 1:
                # Determine reward based on line pattern
                if line_index < 6:  # horizontal or vertical
                    # Convert to indices in meta-board
                    indices = [
                        (0, line_index % 3), (1, line_index % 3), (2, line_index % 3)
                    ] if line_index >= 3 else [
                        (line_index, 0), (line_index, 1), (line_index, 2)
                    ]
                    pos_vals = [local_status[i][j] for i, j in indices]
                    empty_pos = [k for k, v in enumerate(pos_vals) if v == 0]
                    occupied = [k for k in range(3) if k != empty_pos[0]]
                    if {occupied[0], occupied[1]} == {0, 2}:  # corners
                        reward = B
                    elif {occupied[0], occupied[1]} == {0, 1} or {occupied[0], occupied[1]} == {1, 2}:  # edge + corner
                        reward = A
                    else:
                        reward = C
                else:  # diagonal
                    if line_index == 6:
                        indices = [(0, 0), (1, 1), (2, 2)]
                    else:
                        indices = [(0, 2), (1, 1), (2, 0)]
                    pos_vals = [local_status[i][j] for i, j in indices]
                    occupied = [k for k, v in enumerate(pos_vals) if v == player]
                    if 1 in occupied:  # center included
                        reward = D
                    else:  # two corners
                        reward = E
                meta_near_win_layer += reward if player == 1 else -reward

    # Valid move mask
    for action in get_all_valid_actions(state):
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
        game_progress,
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
            nn.Conv2d(17, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU()
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

hyperparams = [1.0, 1.0, 3.0, 3.5, 0.4, 0.3, 0.5, 0.5, 0.4]
data = []
for state, utility in raw_data:
    data.append((state, utility))
    inverted_state = state.invert()
    inverted_utility = 1.0 - utility
    data.append((inverted_state, inverted_utility))

data = load_data()
print(data[12:13])
X = np.stack([extract_features(state, hyperparams) for state, _ in data])
print(X[67])
y = np.array([[utility] for _, utility in data], dtype=np.float32)
y_flat = y.flatten()
num_bins = 20
bin_edges = np.linspace(-1, 1, num_bins + 1)
bin_indices = np.digitize(y_flat, bin_edges)  # bin each y value
from collections import defaultdict
bin_to_indices = defaultdict(list)
for idx, bin_idx in enumerate(bin_indices):
    bin_to_indices[bin_idx].append(idx)
min_samples = min(len(indices) for indices in bin_to_indices.values())
selected_indices = []
for indices in bin_to_indices.values():
    selected = np.random.choice(indices, min_samples, replace=False)
    selected_indices.extend(selected)

X= X[selected_indices]
y = y[selected_indices]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(len(X_train))
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
print("🚀 Training on:", device)

best_loss = float('inf')
patience = 10
patience_counter = 0
best_model_state = None

for epoch in tqdm(range(100), desc="Training"):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_preds = model(X_train_tensor.to(device))
        test_preds = model(X_test_tensor.to(device))
        train_loss = criterion(train_preds, y_train_tensor.to(device)).item()
        test_loss = criterion(test_preds, y_test_tensor.to(device)).item()

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch + 1:>3}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}")

    # Early stopping check
    if test_loss < best_loss:
        best_loss = test_loss
        best_model_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"⏹️ Early stopping at epoch {epoch + 1}, best test loss: {best_loss:.4f}")
            break

# Restore best model
model.load_state_dict(best_model_state)

# Final evaluation
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

with open("model_tp.json", "w") as f:
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

'''


raw_data = load_data()

hyperparams = [1.0, 1.0, 3.0, 3.5]
data = []
for state, utility in raw_data:
    data.append((state, utility))
    inverted_state = state.invert()
    inverted_utility = 1.0 - utility
    data.append((inverted_state, inverted_utility))
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
criterion = nn.MSELoss()

# Load and prepare data
data = load_data()
X = np.stack([extract_features(state, hyperparams) for state, _ in data])
y = np.array([[utility] for _, utility in data], dtype=np.float32)
y_flat = y.flatten()
near_minus1 = np.isclose(y_flat, -1.0, atol=0.05)
near_plus1 = np.isclose(y_flat, 1.0, atol=0.05)

# Final mask: keep only values not near ±1 and not exactly 0
mask = ~(near_minus1 | near_plus1 | (y_flat == 0))
# Apply mask
X = X[mask]
y = y[mask]
# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Model predictions
model.eval()
with torch.no_grad():
    train_preds = model(X_train_tensor.to(device))
    test_preds = model(X_test_tensor.to(device))
    train_loss = criterion(train_preds, y_train_tensor.to(device)).item()
    test_loss = criterion(test_preds, y_test_tensor.to(device)).item()

# Convert to NumPy
y_train_true = y_train_tensor.cpu().numpy().flatten()
y_test_true = y_test_tensor.cpu().numpy().flatten()
train_preds_np = train_preds.cpu().numpy().flatten()
test_preds_np = test_preds.cpu().numpy().flatten()

# Plot
plt.figure(figsize=(12, 5))

# Train
plt.subplot(1, 2, 1)
plt.scatter(y_train_true, train_preds_np, alpha=0.5)
plt.xlabel("True Values (Train)")
plt.ylabel("Predicted Values")
plt.title(f"Train Loss: {train_loss:.4f}")
plt.plot([min(y_train_true), max(y_train_true)], [min(y_train_true), max(y_train_true)], 'r--')

# Test
plt.subplot(1, 2, 2)
plt.scatter(y_test_true, test_preds_np, alpha=0.5)
plt.xlabel("True Values (Test)")
plt.ylabel("Predicted Values")
plt.title(f"Test Loss: {test_loss:.4f}")
plt.plot([min(y_test_true), max(y_test_true)], [min(y_test_true), max(y_test_true)], 'r--')

plt.tight_layout()
plt.show()