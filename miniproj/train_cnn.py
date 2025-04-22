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


def extract_features(state: State, hp) -> np.ndarray:
    ms_c, cc_c, mcor_c, medg_c = hp[:4]
    b, s = state.board, state.local_board_status

    # base layers
    p1 = np.zeros((9, 9), dtype=np.float32)
    p2 = np.zeros((9, 9), dtype=np.float32)
    meta = np.zeros((9, 9), dtype=np.float32)
    turn = np.full((9, 9), float(state.fill_num == 1), dtype=np.float32)

    # local board layers
    near = np.zeros((9, 9), dtype=np.float32)
    center = np.zeros((9, 9), dtype=np.float32)
    dom = np.zeros((9, 9), dtype=np.float32)

    # move-related layers
    valid = np.zeros((9, 9), dtype=np.float32)
    prev = np.zeros((9, 9), dtype=np.float32)
    free = np.full((9, 9), float(state.prev_local_action is None or s[state.prev_local_action[0]][state.prev_local_action[1]] != 0), dtype=np.float32)

    # meta board control
    mcen = np.zeros((9, 9), dtype=np.float32)
    mcor = np.zeros((9, 9), dtype=np.float32)
    medg = np.zeros((9, 9), dtype=np.float32)
    mnear = np.zeros((9, 9), dtype=np.float32)

    # strategy layers
    press = np.zeros((9, 9), dtype=np.float32)
    waste = np.zeros((9, 9), dtype=np.float32)

    win_lines = lambda g: [[g[a][b] for a, b in idxs] for idxs in [
        [(0,0),(0,1),(0,2)], [(1,0),(1,1),(1,2)], [(2,0),(2,1),(2,2)],
        [(0,0),(1,0),(2,0)], [(0,1),(1,1),(2,1)], [(0,2),(1,2),(2,2)],
        [(0,0),(1,1),(2,2)], [(0,2),(1,1),(2,0)]
    ]]

    for i, j in [(0,0),(0,2),(2,0),(2,2)]:
        val = s[i][j]
        mcor[i*3:(i+1)*3, j*3:(j+1)*3] = 1.0 / mcor_c if val == 1 else -1.0 / mcor_c if val == 2 else 0.0

    for i, j in [(0,1),(1,0),(1,2),(2,1)]:
        val = s[i][j]
        medg[i*3:(i+1)*3, j*3:(j+1)*3] = 1.0 / medg_c if val == 1 else -1.0 / medg_c if val == 2 else 0.0

    for i in range(3):
        for j in range(3):
            sub = b[i][j]
            ms = s[i][j]

            dom[i*3:(i+1)*3, j*3:(j+1)*3] = 1.0 if ms == 1 else -1.0 if ms == 2 else 0.0
            meta[i*3:(i+1)*3, j*3:(j+1)*3] = ms / ms_c

            if sub[1][1] != 0:
                center[i*3:(i+1)*3, j*3:(j+1)*3] = sub[1][1] / cc_c

            c1, c2 = np.sum(sub == 1), np.sum(sub == 2)

            for p in [1, 2]:
                for line in win_lines(sub):
                    if line.count(p) == 2 and line.count(0) == 1:
                        v = 0.2 if p == 1 else -0.2
                        near[i*3:(i+1)*3, j*3:(j+1)*3] += v
                        if s[i][j] == 0:
                            waste_val = 0.1 if p == 1 else -0.1
                            if (p == 1 and c1 > 3) or (p == 2 and c2 > 3):
                                waste[i*3:(i+1)*3, j*3:(j+1)*3] += waste_val

            for k in range(3):
                for l in range(3):
                    val = sub[k][l]
                    x, y = i*3 + k, j*3 + l
                    if val == 1: p1[x][y] = 1.0
                    elif val == 2: p2[x][y] = 1.0

    if s[1][1] == 1:
        mcen[3:6, 3:6] = 1.0
    elif s[1][1] == 2:
        mcen[3:6, 3:6] = -1.0

    for p in [1, 2]:
        for line in win_lines(s):
            if line.count(p) == 2 and line.count(0) == 1:
                mnear += 0.45 if p == 1 else -0.45

    for a in get_all_valid_actions(state):
        i, j, k, l = a
        valid[i*3 + k, j*3 + l] = 1.0

    if state.prev_local_action:
        i, j = state.prev_local_action
        prev[i*3:(i+1)*3, j*3:(j+1)*3] = 1.0
        sub = b[i][j]
        t = np.sum((sub == 1) | (sub == 2))
        press[i*3:(i+1)*3, j*3:(j+1)*3] = 1 / (1 + np.exp(-(t - 4.5)))

    return np.stack([
        p1, p2, meta, turn,
        near, center, dom,
        valid, prev, free,
        mcen, mcor, medg, mnear,
        press, waste
    ], axis=0)


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU()
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
        x = self.dropout(F.gelu(self.fc1(x)))
        return torch.tanh(self.fc2(x))
hyperparams = [1.0, 1.0, 3.0, 3.5]


raw_data = load_data()

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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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
import matplotlib.pyplot as plt

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

with open("model7.json", "w") as f:
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
Training:   0%|                                                                                                                                                                                                              | 0/100 [00:00<?, ?it/s]Epoch   1: Train Loss = 0.2156, Test Loss = 0.2152
Training:   9%|█████████████████▊                                                                                                                                                                                    | 9/100 [02:40<26:25, 17.43s/it]Epoch  10: Train Loss = 0.1563, Test Loss = 0.1656
Training:  19%|█████████████████████████████████████▍                                                                                                                                                               | 19/100 [05:37<23:49, 17.65s/it]Epoch  20: Train Loss = 0.1127, Test Loss = 0.1315
Training:  29%|█████████████████████████████████████████████████████████▏                                                                                                                                           | 29/100 [08:41<21:26, 18.12s/it]Epoch  30: Train Loss = 0.1003, Test Loss = 0.1283
Training:  39%|████████████████████████████████████████████████████████████████████████████▊                                                                                                                        | 39/100 [11:44<18:36, 18.30s/it]Epoch  40: Train Loss = 0.0841, Test Loss = 0.1258
Training:  49%|████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                    | 49/100 [14:40<14:45, 17.36s/it]Epoch  50: Train Loss = 0.0736, Test Loss = 0.1259
Training:  59%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                                | 59/100 [17:46<12:27, 18.23s/it]Epoch  60: Train Loss = 0.0687, Test Loss = 0.1293
Training:  69%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                                                             | 69/100 [20:54<09:35, 18.57s/it]Epoch  70: Train Loss = 0.0604, Test Loss = 0.1275
Training:  79%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                         | 79/100 [27:48<21:57, 62.73s/it]Epoch  80: Train Loss = 0.0701, Test Loss = 0.1467
Training:  89%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎                     | 89/100 [31:20<03:59, 21.80s/it]Epoch  90: Train Loss = 0.0526, Test Loss = 0.1315
Training:  99%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████  | 99/100 [34:48<00:20, 20.48s/it]Epoch 100: Train Loss = 0.0474, Test Loss = 0.1303
Training: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [35:12<00:00, 21.13s/it]
✅ Final Training Loss: 0.047390203922986984
✅ Final Test Loss: 0.13026899099349976
model6: 17 c
✅ Final Training Loss: 0.051630232483148575
✅ Final Test Loss: 0.123011015355587
model_ 16 (remove meta edge ctrl)
✅ Final Training Loss: 0.0476309210062027
✅ Final Test Loss: 0.13130676746368408
model_7 remove oppo waste layer
✅ Final Training Loss: 0.05019061267375946
✅ Final Test Loss: 0.12852895259857178
'''