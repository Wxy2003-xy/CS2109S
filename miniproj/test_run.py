
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from utils import State, Action, ImmutableState, load_data, is_terminal, get_all_valid_actions, change_state, terminal_utility, get_random_valid_action, is_valid_action

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
    for action in get_all_valid_actions(state):
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



def extract_features_2(state: State) -> np.ndarray:
    board = state.board
    local_status = state.local_board_status
    p1_layer = np.zeros((9,9), dtype=np.float32)
    p2_layer = np.zeros((9,9), dtype=np.float32)
    meta_layer = np.zeros((9, 9), dtype=np.float32)
    turn_layer = np.full((9, 9), float(state.fill_num == 1), dtype=np.float32)
    valid_moves_layer = np.zeros((9, 9), dtype=np.float32)
    prev_move_layer = np.zeros((9, 9), dtype=np.float32)
    center_control_layer = np.zeros((9, 9), dtype=np.float32)
    local_near_win_layer = np.zeros((9, 9), dtype=np.float32)
    free_move_layer = np.full((9, 9), float(
        state.prev_local_action is None or
        local_status[state.prev_local_action[0]][state.prev_local_action[1]] != 0
    ), dtype=np.float32)
    meta_center_control_layer = np.zeros((9, 9), dtype=np.float32)
    local_dominance_layer = np.zeros((9, 9), dtype=np.float32)
    meta_near_win_layer = np.zeros((9, 9), dtype=np.float32)

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
            meta_layer[i*3:(i+1)*3, j*3:(j+1)*3] = meta_status / 3.0

            # Center control of local board
            center_val = sub[1][1]
            if center_val != 0:
                center_control_layer[i*3:(i+1)*3, j*3:(j+1)*3] = center_val / 2.0

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
                        local_near_win_layer[i*3:(i+1)*3, j*3:(j+1)*3] += value

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
    for action in get_all_valid_actions(state):
        i, j, k, l = action
        x, y = i * 3 + k, j * 3 + l
        valid_moves_layer[x][y] = 1.0

    # Previous move marker
    if state.prev_local_action is not None:
        i, j = state.prev_local_action
        prev_move_layer[i*3:(i+1)*3, j*3:(j+1)*3] = 1.0

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

class CNNModel2(nn.Module):
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
            score = minimax(new_state, depth=1, alpha=float("-inf"), beta=float("inf"), maximizing_player=False, player=player)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action
class RandomStudentAgent(StudentAgent):
    def choose_action(self, state: State) -> Action:
        player = state._state.fill_num
        if state._state.prev_local_action == None:
            return (1, 1, 1, 1)
        valid_actions = get_all_valid_actions(state._state)

        best_score = float("-inf")
        best_action = valid_actions[0]

        for action in valid_actions:
            new_state = change_state(state._state, action)
            score = minimax_ref(new_state, depth=2, alpha=float("-inf"), beta=float("inf"), maximizing_player=False, player=player)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

def run(your_agent: StudentAgent, opponent_agent: RandomStudentAgent, start_num: int):
    your_agent_stats = {"timeout_count": 0, "invalid_count": 0}
    opponent_agent_stats = {"timeout_count": 0, "invalid_count": 0}
    turn_count = 0  
    
    state = State(fill_num=start_num)
    
    while not state.is_terminal():
        turn_count += 1

        agent_name = "your_agent" if state.fill_num == 1 else "opponent_agent"
        agent = your_agent if state.fill_num == 1 else opponent_agent
        stats = your_agent_stats if state.fill_num == 1 else opponent_agent_stats

        start_time = time.time()
        action = agent.choose_action(state.clone())
        end_time = time.time()
        
        random_action = state.get_random_valid_action()
        if end_time - start_time > 3:
            print(f"{agent_name} timed out!")
            stats["timeout_count"] += 1
            action = random_action
        if not state.is_valid_action(action):
            print(f"{agent_name} made an invalid action!")
            stats["invalid_count"] += 1
            action = random_action
                
        state = state.change_state(action)

    # print(f"== {your_agent.__class__.__name__} (1) vs {opponent_agent.__class__.__name__} (2) - First Player: {start_num} ==")
            
    if state.terminal_utility() == 1:
        # print("You win!")
        return 1
    elif state.terminal_utility() == 0:
        # print("You lose!")
        return 0
    else:
        # print("Draw")
        return -1

    for agent_name, stats in [("your_agent", your_agent_stats), ("opponent_agent", opponent_agent_stats)]:
        print(f"{agent_name} statistics:")
        print(f"Timeout count: {stats['timeout_count']}")
        print(f"Invalid count: {stats['invalid_count']}")
        
    print(f"Turn count: {turn_count}\n")

device = torch.device("cpu")
model = CNNModel2()
model_ref = CNNModel() 
# Convert weights to tensors
state_dict = OrderedDict({k: torch.tensor(v, dtype=torch.float32) for k, v in model_weight.items()})
state_dict_2 = OrderedDict({k: torch.tensor(v, dtype=torch.float32) for k, v in model_weight_ref.items()})

model.load_state_dict(state_dict)
model.eval()

model_ref.load_state_dict(state_dict_2)
model_ref.eval()
print("Model device:", next(model.parameters()).device, model)
print("Model2 device:", next(model_ref.parameters()).device, model_ref)

def minimax(state: ImmutableState, depth: int, alpha: float, beta: float, maximizing_player: bool, player: int) -> float:
    if is_terminal(state):
        return terminal_utility(state)
    if depth == 0:
        features = extract_features_2(state)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        state_score = 0
        with torch.no_grad():
            state_score = model(features_tensor).item()
            del features_tensor 
        return state_score

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

def minimax_ref(state: ImmutableState, depth: int, alpha: float, beta: float, maximizing_player: bool, player: int) -> float:
    if is_terminal(state):
        return terminal_utility(state)
    if depth == 0:
        features = extract_features(state)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        state_score = 0
        with torch.no_grad():
            state_score = model_ref(features_tensor).item()
            del features_tensor 
        return state_score

    actions = get_all_valid_actions(state)
    if maximizing_player:
        max_eval = float("-inf")
        for action in actions:
            new_state = change_state(state, action)
            eval = minimax_ref(new_state, depth - 1, alpha, beta, False, player)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float("inf")
        for action in actions:
            new_state = change_state(state, action)
            eval = minimax_ref(new_state, depth - 1, alpha, beta, True, player)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


your_agent = lambda: StudentAgent()
opponent_agent = lambda: RandomStudentAgent()
win_count = 0
draw_count = 0
lose_count = 0
for i in range(50):
    c = run(your_agent(), opponent_agent(), 1)
    if c == 1:
        win_count += 1
    if c == 0:
        draw_count += 1
    if c == -1:
        lose_count += 1
    print("Gmae iteration: ", i, ", game status: ", win_count, ":", draw_count, ":", lose_count)
print(win_count, draw_count, lose_count)
# run(your_agent(), opponent_agent(), 2)