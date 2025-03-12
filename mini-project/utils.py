import numpy as np

Action = tuple[int, int, int, int]

def board_status(board: np.ndarray) -> int:
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != 0:
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != 0:
            return board[0][i]
    if board[0][0] == board[1][1] == board[2][2] != 0:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != 0:
        return board[0][2]
    for i in range(3):
        for j in range(3):
            if board[i][j] == 0:
                return 0
    return 3

class State:
    def __init__(self,
                 board: np.ndarray = np.array([[[[0 for i in range(3)]for j in range(3)] for k in range(3)] for l in range(3)]),
                 fill_num: 1 | 2 = 1,
                 local_board_status: np.ndarray = np.array([[0 for i in range(3)] for j in range(3)]),
                 ):
        self.board = board
        self.fill_num = fill_num
        self.local_board_status = local_board_status
    
    def update_local_board_status(self) -> None:
        for i in range(3):
            for j in range(3):
                self.local_board_status[i][j] = board_status(self.board[i][j])
    
    def is_valid_action(self, action: Action, prev_action: Action | None) -> bool:
        if not isinstance(action, tuple):
            return False
        if len(action) != 4:
            return False
        i, j, k, l = action
        if type(i) != int or type(j) != int or type(k) != int or type(l) != int:
            return False
        if self.local_board_status[i][j] != 0:
            return False
        if self.board[i][j][k][l] != 0:
            return False
        if prev_action is None:
            return True
        _0, _1, prev_row, prev_col = prev_action
        if prev_row == i and prev_col == j:
            return True
        return self.local_board_status[prev_row][prev_col] != 0
    
    def get_all_valid_actions(self, prev_action: Action | None) -> list[Action]:
        if prev_action is None:
            return self._get_all_valid_free_actions()
        _0, _1, prev_row, prev_col = prev_action
        if self.local_board_status[prev_row][prev_col] != 0:
            return self._get_all_valid_free_actions()
        valid_actions: list[Action] = []
        for i in range(3):
            for j in range(3):
                if self.board[prev_row][prev_col][i][j] == 0:
                    valid_actions.append((prev_row, prev_col, i, j))
        return valid_actions
    
    def _get_all_valid_free_actions(self) -> list[Action]:
        valid_actions: list[Action] = []
        for i in range(3):
            for j in range(3):
                if self.local_board_status[i][j] != 0:
                    continue
                for k in range(3):
                    for l in range(3):
                        if self.board[i][j][k][l] == 0:
                            valid_actions.append((i, j, k, l))
        return valid_actions
    
    def get_random_valid_action(self, prev_action: Action | None) -> Action:
        valid_actions = self.get_all_valid_actions(prev_action)
        return valid_actions[np.random.randint(len(valid_actions))]
    
    def change_state(self, action: Action, in_place: bool = False) -> "State":
        i, j, k, l = action
        if in_place:
            self.board[i][j][k][l] = self.fill_num
            self.update_local_board_status()
            return self
        new_board = self.board.copy()
        new_board[i][j][k][l] = self.fill_num
        new_local_board_status = self.local_board_status.copy()
        new_state = State(board=new_board,
                          fill_num=3-self.fill_num,
                          local_board_status=new_local_board_status)
        new_state.update_local_board_status()
        return new_state
    
    def is_terminal(self) -> bool:
        return board_status(self.local_board_status) != 0
    
    def terminal_utility(self) -> float:
        status = board_status(self.local_board_status)
        if status == 1:
            return 1.0
        if status == 2:
            return 0.0
        if status == 3:
            return 0.5
        assert False, "Board is not terminal"
    
    def clone(self) -> "State":
        return State(board=self.board.copy(),
                     fill_num=self.fill_num,
                     local_board_status=self.local_board_status.copy(),
                     )


U = 0
X = 1
O = 2
D = 3

class Utils:
    def compute_wins() -> list[bool]:
        result: list[bool] = []
        winning_lines: list[int] = [
            0b100010001,
            0b001010100,
            0b000000111,
            0b000111000,
            0b111000000,
            0b001001001,
            0b010010010,
            0b100100100,
        ]
        for i in range(0b111111111):
            is_win = False
            for line in winning_lines:
                if i & line == line:
                    is_win = True
                    break
            result.append(is_win)
        return result

    wins: list[bool] = compute_wins()

class Board:
    def __init__(self, sub_boards: list[tuple[int, int]] = None, sub_board_index: int = 9, is_X: bool = True, metas: tuple[int, int, int] = None):
        self.sub_boards = sub_boards if sub_boards else [
            (0, 0) for _ in range(9)
        ]
        self.sub_board_index = sub_board_index
        self.is_X = is_X
        if metas:
            self.Xmeta, self.Ometa, self.Dmeta = metas
        else:
            self.Xmeta, self.Ometa, self.Dmeta = Board._find_meta(self.sub_boards)
        self.winner = self._determine_winner()
    
    def _find_meta(sub_boards: list[tuple[int, int]]):
        Xmeta = 0
        Ometa = 0
        Dmeta = 0
        for i, sub_board in enumerate(sub_boards):
            if Utils.wins[sub_board[0]]:
                Xmeta |= 1 << i
            elif Utils.wins[sub_board[1]]:
                Ometa |= 1 << i
            elif (sub_board[0] | sub_board[1]) == 0b111111111:
                Dmeta |= 1 << i
        return Xmeta, Ometa, Dmeta
    
    def from_state(state: State, prev_action: Action | None) -> "Board":
        if prev_action is None:
            sub_board_index = 9
        else:
            i, j, k, l = prev_action
            sub_board_index = 3 * k + l
        is_X = state.fill_num == 1
        sub_boards: list[tuple[int, int]] = [Board.sub_board_from_numpy_array(state.board[i][j]) for i in range(3) for j in range(3)]
        Xmeta, Ometa, Dmeta = Board._find_meta(sub_boards)
        if sub_board_index != 9 and (((Xmeta | Ometa | Dmeta) >> sub_board_index) & 1) == 1:
            sub_board_index = 9
        return Board(sub_boards, sub_board_index, is_X, (Xmeta, Ometa, Dmeta))
    
    def to_state_and_action(self) -> tuple[State, tuple[int, int] | None]:
        state = self._to_state()
        action = self._to_action()
        return state, action
    
    def _to_state(self) -> State:
        arr = []
        for i in range(3):
            row = []
            for j in range(3):
                row.append(Board.sub_board_to_numpy_array(self.sub_boards[3 * i + j]))
            arr.append(row)
        state = State(board=np.array(arr), fill_num=1 if self.is_X else 2)
        state.update_local_board_status()
        return state
    
    def _to_action(self) -> tuple[int, int] | None:
        if self.sub_board_index == 9:
            return None
        # first two indices do not matter
        return (self.sub_board_index // 3, self.sub_board_index % 3)
    
    def from_compact_string(string: str) -> "Board":
        parts = string.split(" ")
        assert len(parts) == 10
        sub_boards = [tuple(map(int, part.split(","))) for part in parts[:-1]]
        extras = parts[-1].split(",")
        sub_board_index = int(extras[0])
        is_X = extras[1] == "0"
        Xmeta, Ometa, Dmeta = Board._find_meta(sub_boards)
        assert (((Xmeta | Ometa | Dmeta) >> sub_board_index) & 1) == 0 or sub_board_index == 9
        return Board(sub_boards, sub_board_index, is_X)

    def sub_board_from_numpy_array(array: np.ndarray) -> tuple[int, int]:
        X_board = 0
        O_board = 0
        for i in range(3):
            for j in range(3):
                if array[i][j] == 1:
                    X_board += 1 << (3 * i + j)
                elif array[i][j] == 2:
                    O_board += 1 << (3 * i + j)
        return X_board, O_board
    
    def sub_board_to_numpy_array(sub_board: tuple[int, int]) -> np.ndarray:
        X_board, O_board = sub_board
        array = np.zeros((3, 3), dtype=int)
        for i in range(3):
            for j in range(3):
                if ((X_board >> (3 * i + j)) & 1) == 1:
                    array[i][j] = 1
                elif ((O_board >> (3 * i + j)) & 1) == 1:
                    array[i][j] = 2
        return array
    
    def actions(self) -> list[int]:
        result: list[int] = []
        if self.sub_board_index == 9:
            for i, sub_board in enumerate(self.sub_boards):
                if (((self.Xmeta | self.Ometa | self.Dmeta) >> i) & 1) == 1:
                    continue
                occ_board = sub_board[0] | sub_board[1]
                for j in range(9):
                    if ((occ_board >> j) & 1) == 0:
                        result.append(i * 9 + j)
        else:
            occ_board = self.sub_boards[self.sub_board_index][0] | self.sub_boards[self.sub_board_index][1]
            for j in range(9):
                if ((occ_board >> j) & 1) == 0:
                    result.append(self.sub_board_index * 9 + j)
        return result

    def move(self, action: int) -> "Board":
        board_index = action // 9 if self.sub_board_index == 9 else self.sub_board_index
        new_sub_boards = self.sub_boards.copy()
        if self.is_X:
            new_sub_boards[board_index] = (new_sub_boards[board_index][0] | (1 << (action % 9)), new_sub_boards[board_index][1])
        else:
            new_sub_boards[board_index] = (new_sub_boards[board_index][0], new_sub_boards[board_index][1] | (1 << (action % 9)))
        Xmeta = self.Xmeta
        Ometa = self.Ometa
        Dmeta = self.Dmeta
        if Utils.wins[new_sub_boards[board_index][0]]:
            Xmeta |= 1 << board_index
        elif Utils.wins[new_sub_boards[board_index][1]]:
            Ometa |= 1 << board_index
        elif (new_sub_boards[board_index][0] | new_sub_boards[board_index][1]) == 0b111111111:
            Dmeta |= 1 << board_index
        next_sub_board_index = action % 9
        if (((Xmeta | Ometa | Dmeta) >> next_sub_board_index) & 1) == 1:
            next_sub_board_index = 9
        return Board(new_sub_boards, next_sub_board_index, not self.is_X)
    
    def _determine_winner(self):
        if Utils.wins[self.Xmeta]:
            return X
        elif Utils.wins[self.Ometa]:
            return O
        elif (self.Xmeta | self.Ometa | self.Dmeta) == 0b111111111:
            return D
        else:
            return U
    
    def compact_string(self) -> str:
        result = ""
        for sub_board in self.sub_boards:
            result += f"{sub_board[0]},{sub_board[1]} "
        result += f"{self.sub_board_index},{0 if self.is_X else 1}"
        return result
    
    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Board):
            return False
        return self.sub_board_index == other.sub_board_index \
            and self.is_X == other.is_X \
            and self.sub_boards == other.sub_boards

def action_from_tuple(action: Action) -> int:
    i, j , k, l = action
    return (3 * i + j) * 9 + (3 * k + l)

def to_action_tuple(action: int) -> Action:
    index = action // 9
    cell = action % 9
    return (index // 3, index % 3, cell // 3, cell % 3)

def load_data(filename: str = "data.uttt") -> list[tuple[Board, float]]:
    with open(filename) as f:
        data_results = f.read().strip().split("\n")
    data: list[tuple[Board, float]] = []
    for data_result in data_results:
        board_str, value_str = data_result.strip().split("|")
        board: Board = Board.from_compact_string(board_str)
        state, action = board.to_state_and_action()
        value = float(value_str) if board.is_X else -float(value_str)
        data.append((state, action, value))
    return data
