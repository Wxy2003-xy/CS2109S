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


# def extract_features(state: State, hyperparams) -> np.ndarray:
    
#     meta_status_scale = hyperparams[0]
#     central_val_scale = hyperparams[1]
#     corner_importance = hyperparams[2]
#     edge_importance = hyperparams[3]
    
    
#     board = state.board
#     local_status = state.local_board_status
#     player = state.fill_num
#     opponent = 3 - player
    
    
#     features = {}
    
    
#     features['player_pieces'] = np.zeros((9, 9), dtype=np.float32)
#     features['opponent_pieces'] = np.zeros((9, 9), dtype=np.float32)
#     features['empty_cells'] = np.zeros((9, 9), dtype=np.float32)
    
    
#     features['turn_indicator'] = np.full((9, 9), float(player == 1), dtype=np.float32)
#     features['free_move_indicator'] = np.full((9, 9), float(
#         state.prev_local_action is None or
#         local_status[state.prev_local_action[0]][state.prev_local_action[1]] != 0
#     ), dtype=np.float32)
#     features['valid_moves'] = np.zeros((9, 9), dtype=np.float32)
#     features['previous_move'] = np.zeros((9, 9), dtype=np.float32)
#     features['game_progress'] = np.zeros((9, 9), dtype=np.float32)
    
    
#     features['local_win_threat'] = np.zeros((9, 9), dtype=np.float32)  
#     features['local_block_needed'] = np.zeros((9, 9), dtype=np.float32)  
#     features['center_control'] = np.zeros((9, 9), dtype=np.float32)  
#     features['local_board_status'] = np.zeros((9, 9), dtype=np.float32)  
#     features['target_pressure'] = np.zeros((9, 9), dtype=np.float32)  
#     features['forced_bad_move'] = np.zeros((9, 9), dtype=np.float32)  
    
    
#     features['meta_center_control'] = np.zeros((9, 9), dtype=np.float32)
#     features['meta_corner_control'] = np.zeros((9, 9), dtype=np.float32)
#     features['meta_edge_control'] = np.zeros((9, 9), dtype=np.float32)
#     features['meta_win_opportunity'] = np.zeros((9, 9), dtype=np.float32)
#     features['meta_strategic_path'] = np.zeros((9, 9), dtype=np.float32)
#     features['locked_board'] = np.zeros((9, 9), dtype=np.float32)
    
    
#     features['fork_opportunity'] = np.zeros((9, 9), dtype=np.float32)  
#     features['board_completion'] = np.zeros((9, 9), dtype=np.float32)  
#     features['diagonal_strength'] = np.zeros((9, 9), dtype=np.float32)  
#     features['center_to_corner_pattern'] = np.zeros((9, 9), dtype=np.float32)  

    
#     total_placed = 0
    
    
#     for i in range(3):
#         for j in range(3):
#             sub_board = board[i][j]
#             meta_status = local_status[i][j]
            
            
#             region = (slice(i*3, (i+1)*3), slice(j*3, (j+1)*3))
            
            
#             features['local_board_status'][region] = meta_status / meta_status_scale
            
            
#             if meta_status == player:
#                 features['meta_strategic_path'][region] = 0.8
#             elif meta_status == opponent:
#                 features['meta_strategic_path'][region] = -0.5
            
            
#             if meta_status == 0 and np.all(sub_board != 0):
#                 features['locked_board'][region] = 1.0
#                 features['board_completion'][region] = 1.0
#             else:
                
#                 filled_cells = np.count_nonzero(sub_board)
#                 features['board_completion'][region] = filled_cells / 9.0
            
            
#             center_val = sub_board[1][1]
#             if center_val != 0:
#                 center_strength = 1.0 if center_val == player else -0.8
#                 features['center_control'][region] = center_strength / central_val_scale
                
                
#                 corner_sum = sum(1 for corner in [(0,0), (0,2), (2,0), (2,2)] if sub_board[corner] == center_val)
#                 if corner_sum >= 2 and center_val == player:
#                     features['center_to_corner_pattern'][region] = 0.7
            
            
#             if (i, j) in [(0,0), (0,2), (2,0), (2,2)]:  
#                 if meta_status == player:
#                     features['meta_corner_control'][region] = 1.0 / corner_importance
#                 elif meta_status == opponent:
#                     features['meta_corner_control'][region] = -1.0 / corner_importance
#             elif (i, j) in [(0,1), (1,0), (1,2), (2,1)]:  
#                 if meta_status == player:
#                     features['meta_edge_control'][region] = 1.0 / edge_importance
#                 elif meta_status == opponent:
#                     features['meta_edge_control'][region] = -1.0 / edge_importance
#             elif (i, j) == (1, 1):  
#                 if meta_status == player:
#                     features['meta_center_control'][region] = 1.5
#                 elif meta_status == opponent:
#                     features['meta_center_control'][region] = -1.2
            
            
#             player_near_wins = 0
#             opponent_near_wins = 0
#             player_fork_opportunities = 0
            
            
#             win_lines = [
#                 sub_board[0, :], sub_board[1, :], sub_board[2, :],  
#                 sub_board[:, 0], sub_board[:, 1], sub_board[:, 2],  
#                 np.array([sub_board[0][0], sub_board[1][1], sub_board[2][2]]),  
#                 np.array([sub_board[0][2], sub_board[1][1], sub_board[2][0]])   
#             ]
            
            
#             diag1 = [sub_board[0][0], sub_board[1][1], sub_board[2][2]]
#             diag2 = [sub_board[0][2], sub_board[1][1], sub_board[2][0]]
#             diag_control = 0
            
#             for d in [diag1, diag2]:
#                 p_count = d.count(player)
#                 o_count = d.count(opponent)
#                 if p_count > 0 and o_count == 0:
#                     diag_control += p_count * 0.2
#                 elif o_count > 0 and p_count == 0:
#                     diag_control -= o_count * 0.15
            
#             if diag_control != 0:
#                 features['diagonal_strength'][region] = diag_control
                
            
#             for line in win_lines:
#                 p_count = np.sum(line == player)
#                 o_count = np.sum(line == opponent)
#                 empty_count = np.sum(line == 0)
                
                
#                 if p_count == 2 and empty_count == 1:
#                     player_near_wins += 1
#                     features['local_win_threat'][region] += 0.3
                
                
#                 if o_count == 2 and empty_count == 1:
#                     opponent_near_wins += 1
#                     features['local_block_needed'][region] += 0.4
                
                
#                 if p_count == 1 and empty_count == 2:
#                     player_fork_opportunities += 1
            
            
#             if player_fork_opportunities >= 2:
#                 features['fork_opportunity'][region] = 0.5 * min(player_fork_opportunities / 3, 1.0)
            
            
#             if opponent_near_wins > 1 and meta_status == 0:
#                 features['forced_bad_move'][region] = -0.2 * opponent_near_wins
                
            
#             for k in range(3):
#                 for l in range(3):
#                     cell_val = sub_board[k][l]
#                     pos = (i*3+k, j*3+l)
                    
#                     if cell_val == player:
#                         features['player_pieces'][pos] = 1.0
#                     elif cell_val == opponent:
#                         features['opponent_pieces'][pos] = 1.0
#                     else:  
#                         features['empty_cells'][pos] = 1.0
                    
#                     if cell_val != 0:
#                         total_placed += 1

    
#     features['game_progress'].fill(total_placed / 81.0)
    
    
#     meta_lines = [
#         [local_status[0][0], local_status[0][1], local_status[0][2]],  
#         [local_status[1][0], local_status[1][1], local_status[1][2]],
#         [local_status[2][0], local_status[2][1], local_status[2][2]],
#         [local_status[0][0], local_status[1][0], local_status[2][0]],  
#         [local_status[0][1], local_status[1][1], local_status[2][1]],
#         [local_status[0][2], local_status[1][2], local_status[2][2]],
#         [local_status[0][0], local_status[1][1], local_status[2][2]],  
#         [local_status[0][2], local_status[1][1], local_status[2][0]],
#     ]
    
    
#     for line in meta_lines:
#         p_count = line.count(player)
#         o_count = line.count(opponent)
#         empty_count = line.count(0)
        
        
#         if p_count == 2 and empty_count == 1:
#             features['meta_win_opportunity'] += 0.6
#         elif p_count == 1 and empty_count == 2:
#             features['meta_strategic_path'] += 0.3
            
        
#         if o_count == 2 and empty_count == 1:
            
#             features['meta_win_opportunity'] -= 0.5
    
    
#     if state.prev_local_action is not None:
#         i, j = state.prev_local_action
#         region = (slice(i*3, (i+1)*3), slice(j*3, (j+1)*3))
        
        
#         features['previous_move'][region] = 1.0
        
        
#         sub_board = board[i][j]
#         filled_cells = np.count_nonzero(sub_board)
#         center_occupied = sub_board[1][1] != 0
        
        
#         center_factor = 0.2 if center_occupied else 0
#         pressure_score = (filled_cells + center_factor) / 9.0
#         features['target_pressure'][region] = pressure_score
    
    
#     for action in get_all_valid_actions(state):
#         i, j, k, l = action
#         features['valid_moves'][i*3+k, j*3+l] = 1.0
    
    
#     return np.stack([
#         features['game_progress'],             
#         features['turn_indicator'],            
#         features['free_move_indicator'],       
#         features['player_pieces'],             
#         features['opponent_pieces'],           
#         features['empty_cells'],               
#         features['valid_moves'],               
#         features['local_board_status'],        
#         features['local_win_threat'],          
#         features['local_block_needed'],        
#         features['center_control'],            
#         features['meta_win_opportunity'],      
#         features['meta_strategic_path'],       
#         features['previous_move'],             
#         features['target_pressure'],           
#         features['meta_center_control'],       
#         features['meta_corner_control'],       
#         features['meta_edge_control'],         
#         features['diagonal_strength'],         
#         features['fork_opportunity'],          
#         features['board_completion'],          
#         features['forced_bad_move'],           
#         features['center_to_corner_pattern'],  
#         features['locked_board'],              
#     ], axis=0)


def extract_features(state: State, hyperparams) -> np.ndarray:
    meta_status_scale, central_val_scale, corner_importance, edge_importance = hyperparams
    
    board = state.board
    local_status = state.local_board_status
    player = state.fill_num
    opponent = 3 - player
    
    
    features = {
        'player_pieces': np.zeros((9, 9), dtype=np.float32),
        'opponent_pieces': np.zeros((9, 9), dtype=np.float32),
        'empty_cells': np.zeros((9, 9), dtype=np.float32),
        'turn_indicator': np.full((9, 9), float(player == 1), dtype=np.float32),
        'free_move_indicator': np.full((9, 9), float(
            state.prev_local_action is None or
            local_status[state.prev_local_action[0]][state.prev_local_action[1]] != 0
        ), dtype=np.float32),
        'valid_moves': np.zeros((9, 9), dtype=np.float32),
        'previous_move': np.zeros((9, 9), dtype=np.float32),
        'local_win_threat': np.zeros((9, 9), dtype=np.float32),
        'local_block_needed': np.zeros((9, 9), dtype=np.float32),
        'center_control': np.zeros((9, 9), dtype=np.float32),
        'local_board_status': np.zeros((9, 9), dtype=np.float32),
        'target_pressure': np.zeros((9, 9), dtype=np.float32),
        'forced_bad_move': np.zeros((9, 9), dtype=np.float32),
        'meta_center_control': np.zeros((9, 9), dtype=np.float32),
        'meta_corner_control': np.zeros((9, 9), dtype=np.float32),
        'meta_edge_control': np.zeros((9, 9), dtype=np.float32),
        'meta_win_opportunity': np.zeros((9, 9), dtype=np.float32),
        'meta_strategic_path': np.zeros((9, 9), dtype=np.float32),
        'locked_board': np.zeros((9, 9), dtype=np.float32),
        'fork_opportunity': np.zeros((9, 9), dtype=np.float32),
        'board_completion': np.zeros((9, 9), dtype=np.float32),
        'diagonal_strength': np.zeros((9, 9), dtype=np.float32),
        'center_to_corner_pattern': np.zeros((9, 9), dtype=np.float32),
    }
    
    total_placed = 0
    
    
    positions = np.zeros((3, 3, 3, 3, 2), dtype=np.int32)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    positions[i, j, k, l] = (i*3+k, j*3+l)
    
    
    region_slices = {}
    for i in range(3):
        for j in range(3):
            region_slices[(i, j)] = (slice(i*3, (i+1)*3), slice(j*3, (j+1)*3))
    
    
    corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
    edges = [(0, 1), (1, 0), (1, 2), (2, 1)]
    
    
    for i in range(3):
        for j in range(3):
            sub_board = board[i][j]
            meta_status = local_status[i][j]
            region = region_slices[(i, j)]
            
            
            features['local_board_status'][region] = meta_status / meta_status_scale
            
            
            for k in range(3):
                for l in range(3):
                    cell_val = sub_board[k][l]
                    pos = positions[i, j, k, l]
                    
                    if cell_val == player:
                        features['player_pieces'][tuple(pos)] = 1.0
                    elif cell_val == opponent:
                        features['opponent_pieces'][tuple(pos)] = 1.0
                    else:
                        features['empty_cells'][tuple(pos)] = 1.0
                    
                    if cell_val != 0:
                        total_placed += 1
            
            
            if meta_status == player:
                features['meta_strategic_path'][region] = 0.8
            elif meta_status == opponent:
                features['meta_strategic_path'][region] = -0.5
            
            
            filled_cells = np.count_nonzero(sub_board)
            if meta_status == 0 and filled_cells == 9:
                features['locked_board'][region] = 1.0
                features['board_completion'][region] = 1.0
            else:
                features['board_completion'][region] = filled_cells / 9.0
            
            
            center_val = sub_board[1][1]
            if center_val != 0:
                center_strength = 1.0 if center_val == player else -0.8
                features['center_control'][region] = center_strength / central_val_scale
                
                
                corner_sum = sum(1 for corner in [(0,0), (0,2), (2,0), (2,2)] if sub_board[corner] == center_val)
                if corner_sum >= 2 and center_val == player:
                    features['center_to_corner_pattern'][region] = 0.7
            
            
            if (i, j) in corners:
                if meta_status == player:
                    features['meta_corner_control'][region] = 1.0 / corner_importance
                elif meta_status == opponent:
                    features['meta_corner_control'][region] = -1.0 / corner_importance
            elif (i, j) in edges:
                if meta_status == player:
                    features['meta_edge_control'][region] = 1.0 / edge_importance
                elif meta_status == opponent:
                    features['meta_edge_control'][region] = -1.0 / edge_importance
            elif (i, j) == (1, 1):
                if meta_status == player:
                    features['meta_center_control'][region] = 1.5
                elif meta_status == opponent:
                    features['meta_center_control'][region] = -1.2
            
            
            win_lines = [
                sub_board[0, :], sub_board[1, :], sub_board[2, :],
                sub_board[:, 0], sub_board[:, 1], sub_board[:, 2],
                np.array([sub_board[0][0], sub_board[1][1], sub_board[2][2]]),
                np.array([sub_board[0][2], sub_board[1][1], sub_board[2][0]])
            ]
            
            player_near_wins = 0
            opponent_near_wins = 0
            player_fork_opportunities = 0
            
            
            diag1 = np.array([sub_board[0][0], sub_board[1][1], sub_board[2][2]])
            diag2 = np.array([sub_board[0][2], sub_board[1][1], sub_board[2][0]])
            
            for diag in [diag1, diag2]:
                p_count = np.sum(diag == player)
                o_count = np.sum(diag == opponent)
                if p_count > 0 and o_count == 0:
                    features['diagonal_strength'][region] += p_count * 0.2
                elif o_count > 0 and p_count == 0:
                    features['diagonal_strength'][region] -= o_count * 0.15
            
            
            for line in win_lines:
                p_count = np.sum(line == player)
                o_count = np.sum(line == opponent)
                empty_count = np.sum(line == 0)
                
                if p_count == 2 and empty_count == 1:
                    player_near_wins += 1
                    features['local_win_threat'][region] += 0.3
                
                if o_count == 2 and empty_count == 1:
                    opponent_near_wins += 1
                    features['local_block_needed'][region] += 0.4
                
                if p_count == 1 and empty_count == 2:
                    player_fork_opportunities += 1
            
            
            if player_fork_opportunities >= 2:
                features['fork_opportunity'][region] = 0.5 * min(player_fork_opportunities / 3, 1.0)
            
            
            if opponent_near_wins > 1 and meta_status == 0:
                features['forced_bad_move'][region] = -0.2 * opponent_near_wins
    
    
    features['game_progress'] = np.full((9, 9), total_placed / 81.0, dtype=np.float32)
    
    
    meta_lines = [
        [local_status[0][0], local_status[0][1], local_status[0][2]],
        [local_status[1][0], local_status[1][1], local_status[1][2]],
        [local_status[2][0], local_status[2][1], local_status[2][2]],
        [local_status[0][0], local_status[1][0], local_status[2][0]],
        [local_status[0][1], local_status[1][1], local_status[2][1]],
        [local_status[0][2], local_status[1][2], local_status[2][2]],
        [local_status[0][0], local_status[1][1], local_status[2][2]],
        [local_status[0][2], local_status[1][1], local_status[2][0]],
    ]
    
    for line in meta_lines:
        p_count = line.count(player)
        o_count = line.count(opponent)
        empty_count = line.count(0)
        
        if p_count == 2 and empty_count == 1:
            features['meta_win_opportunity'] += 0.6
        elif p_count == 1 and empty_count == 2:
            features['meta_strategic_path'] += 0.3
        
        if o_count == 2 and empty_count == 1:
            features['meta_win_opportunity'] -= 0.5
    
    
    if state.prev_local_action is not None:
        i, j = state.prev_local_action
        region = region_slices[(i, j)]
        
        features['previous_move'][region] = 1.0
        
        sub_board = board[i][j]
        filled_cells = np.count_nonzero(sub_board)
        center_occupied = sub_board[1][1] != 0
        
        center_factor = 0.2 if center_occupied else 0
        pressure_score = (filled_cells + center_factor) / 9.0
        features['target_pressure'][region] = pressure_score
    
    
    for action in get_all_valid_actions(state):
        i, j, k, l = action
        features['valid_moves'][i*3+k, j*3+l] = 1.0
    
    
    return np.stack([
        features['game_progress'],
        features['turn_indicator'],
        features['free_move_indicator'],
        features['player_pieces'],
        features['opponent_pieces'],
        features['empty_cells'],
        features['valid_moves'],
        features['local_board_status'],
        features['local_win_threat'],
        features['local_block_needed'],
        features['center_control'],
        features['meta_win_opportunity'],
        features['meta_strategic_path'],
        features['previous_move'],
        features['target_pressure'],
        features['meta_center_control'],
        features['meta_corner_control'],
        features['meta_edge_control'],
        features['diagonal_strength'],  
        features['fork_opportunity'],
        features['board_completion'],
        features['forced_bad_move'],
        features['center_to_corner_pattern'],
        features['locked_board'],
    ], axis=0)

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
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
hyperparams = [1.0, 2.0, 3.0, 3.5]
data = load_data()

X = np.stack([extract_features(state, hyperparams) for state, _ in data])
y = np.array([[utility] for _, utility in data], dtype=np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)
criterion = nn.MSELoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
print("🚀 Training on:", device)

num_epochs = 50
for epoch in tqdm(range(num_epochs), desc="Training"):
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
