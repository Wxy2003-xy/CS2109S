import copy
from enum import Enum

# Enum class for player colour
class Player(Enum):
    BLACK = 'black'
    WHITE = 'white'

    # returns the opponent of the current player
    def get_opponent(self):
        if self == Player.BLACK:
            return Player.WHITE
        else:
            return Player.BLACK

# board row and column -> these are constant
ROW, COL = 6, 6
INF = 90129012
WIN = 21092109
MOVE_NONE = (-1, -1), (-1, -1)
TIME_LIMIT = 10

Move = tuple[tuple[int, int], tuple[int, int]]
Board = list[list[str]]

# prints out the current state of the board in a comprehensible way
def print_state(board: Board) -> None:
    horizontal_rule = "+" + ("-" * 5 + "+") * COL
    for row in board:
        print(horizontal_rule)
        print(f"|  {'  |  '.join(' ' if tile == '_' else tile for tile in row)}  |")
    print(horizontal_rule)

# checks if a move made for the current player is valid or not.
# Move source: src (row, col), move destination: dst (row, col)
def is_valid_move(
        board: Board,
        src: tuple[int, int],
        dst: tuple[int, int],
        current_player: Player
    ) -> bool:
    """
    Checks whether the given move is a valid move.

    Parameters
    ----------
    board: 2D list-of-lists. Contains characters "B", "W", and "_"
    representing black pawn, white pawn, and empty cell respectively.

    src: tuple[int, int]. Source position of the pawn.

    dst: tuple[int, int]. Destination position of the pawn.

    current_player: Player. The colour of the current player to move.

    Returns
    -------
    A boolean indicating whether the given move from `src` to `dst` is valid.
    """
    sr, sc = src
    dr, dc = dst
    player_piece = "B" if current_player == Player.BLACK else "W"
    direction = 1 if current_player == Player.BLACK else -1

    if board[sr][sc] != player_piece:
        return False
    if not (0 <= dr < ROW and 0 <= dc < COL):
        return False
    if dr != sr + direction:
        return False
    if abs(dc - sc) > 1:
        return False
    if dc == sc and board[dr][dc] != "_":
        return False
    if dc != sc and board[dr][dc] == player_piece:
        return False

    return True

# makes a move effective on the board by modifying board state,
# or returning a new board with updated board state
def make_move(
        board: Board,
        src: tuple[int, int],
        dst: tuple[int, int],
        current_player: Player,
        in_place: bool = True
    ) -> Board:
    """
    Updates the board configuration by modifying existing values if in_place is set to True,
    or creating a new board with updated values if in_place is set to False.

    Parameters
    ----------
    board: 2D list-of-lists. Contains characters "B", "W", and "_"
    representing black pawn, white pawn, and empty cell respectively.

    src: tuple[int, int]. Source position of the pawn.

    dst: tuple[int, int]. Destination position of the pawn.

    current_player: Player. The colour of the current player to move.

    in_place: bool. Whether the modification is to be made in-place or to a deep copy of the given
    `board`.

    Returns
    -------
    The modified board.
    """
    if not in_place:
        board = copy.deepcopy(board)
    if is_valid_move(board, src, dst, current_player):
        sr, sc = src
        dr, dc = dst
        board[sr][sc] = "_"
        if current_player == Player.BLACK:
            board[dr][dc] = "B"
        else:
            board[dr][dc] = "W"
    return board

# checks if game is over
def is_game_over(board: Board) -> bool:
    """
    Returns True if game is over.

    Parameters
    ----------
    board: 2D list-of-lists. Contains characters "B", "W", and "_"
    representing black pawn, white pawn, and empty cell respectively.

    Returns
    -------
    A bool representing whether the game is over.
    """
    if any(tile == "B" for tile in board[5]) or any(tile == "W" for tile in board[0]):
        return True
    wcount, bcount = 0, 0
    for row in board:
        for tile in row:
            if tile == "B":
                bcount += 1
            elif tile == "W":
                wcount += 1
    return bcount == 0 or wcount == 0
