### Task 1.1: State representation

### Task 1.2: Initial and goal states

### Task 1.3: State transitions

"""
Run this cell before you start!
"""

import random
import time

from typing import List, Tuple, Callable

def transition(route: List[int]) -> List[List[int]]:
    """
    Generates new routes to be used in the next iteration in the hill-climbing algorithm.

    Args:
        route (List[int]): The current route as a list of cities in the order of travel.

    Returns:
        new_routes (List[List[int]]): New routes to be considered.
    """
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_1_3():
    def test_transition(route):
        sorted_route = sorted(route)
        result = transition(route)
        assert result is not None, "Transition function returns an empty list."
        assert any(result), "Transition function returns an empty list."
        for new_route in result:
            assert len(new_route) == len(sorted_route), "New route does not have the same number of cities as the original route."
            assert sorted(new_route) == sorted_route, "New route does not contain all cities present in the original route."
    
    permutation_route = list(range(4))
    new_permutation_routes = transition(permutation_route)
    assert len(new_permutation_routes) < 24, "Your transition function may have generated too many new routes by enumerating all possible states."
    
    test_transition([1, 3, 2, 0])
    test_transition([7, 8, 6, 3, 5, 4, 9, 2, 0, 1])

### Task 1.4: Evaluation function

def evaluation_func(
    cities: int,
    distances: List[Tuple[int]],
    route: List[int]
) -> float:
    """
    Computes the evaluation score of a route

    Args:
        cities (int): The number of cities to be visited.

        distances (List[Tuple[int]]): The list of distances between every two cities. Each distance
            is represented as a tuple in the form of (c1, c2, d), where c1 and c2 are the two cities
            and d is the distance between them. The length of the list should be equal to cities *
            (cities - 1)/2.

        route (List[int]): The current route as a list of cities in the order of travel.

    Returns:
        h_n (float): the evaluation score.
    """
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_1_4():
    cities = 4
    distances = [(1, 0, 10), (0, 3, 22), (2, 1, 8), (2, 3, 30), (1, 3, 25), (0, 2, 15)]
    
    
    route_1 = evaluation_func(cities, distances, [0, 1, 2, 3])
    route_2 = evaluation_func(cities, distances, [2, 1, 3, 0])
    route_3 = evaluation_func(cities, distances, [1, 3, 2, 0])
    route_4 = evaluation_func(cities, distances, [2, 3, 0, 1])
    
    assert route_1 == route_2
    assert route_1 > route_3
    assert route_1 == route_4, "Have you considered the cost to travel from the last city to the headquarter (first)?"

### Task 1.5: Explain your evaluation function

### Task 1.6: Implement hill-climbing

def hill_climbing(
    cities: int,
    distances: List[Tuple[int]],
    transition: Callable,
    evaluation_func: Callable
) -> List[int]:
    """
    Hill climbing finds the solution to reach the goal from the initial.

    Args:
        cities (int): The number of cities to be visited.

        distances (List[Tuple[int]]): The list of distances between every two cities. Each distance
            is represented as a tuple in the form of (c1, c2, d), where c1 and c2 are the two cities
            and d is the distance between them. The length of the list should be equal to cities *
            (cities - 1)/2.

        transition (Callable): A function that generates new routes to be used in the next
            iteration in the hill-climbing algorithm. Will be provided on Coursemology.

        evaluation_func (Callable): A function that computes the evaluation score of a route. Will
            be provided on Coursemology.

    Returns:
        route (List[int]): The shortest route, represented by a list of cities in the order to be
            traversed.
    """
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_1_6():
    def test_hill_climbing(cities: int, distances: List[Tuple[int]], transition: callable, evaluation_func: callable):
        route = hill_climbing(cities, distances, transition, evaluation_func)
        assert sorted(route) == list(range(cities)), "New route does not contain all cities present in the original route."
    
    cities_1 = 4
    distances_1 = [(1, 0, 10), (0, 3, 22), (2, 1, 8), (2, 3, 30), (1, 3, 25), (0, 2, 15)]
    
    test_hill_climbing(cities_1, distances_1, transition, evaluation_func)
    
    cities_2 = 10
    distances_2 = [(2, 7, 60), (1, 6, 20), (5, 4, 70), (9, 8, 90), (3, 7, 54), (2, 5, 61),
        (4, 1, 106), (0, 6, 51), (3, 1, 45), (0, 5, 86), (9, 2, 73), (8, 4, 14), (0, 1, 51),
        (9, 7, 22), (3, 2, 22), (8, 1, 120), (5, 7, 92), (5, 6, 60), (6, 2, 10), (8, 3, 78),
        (9, 6, 82), (0, 2, 41), (2, 8, 99), (7, 8, 71), (0, 9, 32), (4, 0, 73), (0, 3, 42),
        (9, 1, 80), (4, 2, 85), (5, 9, 113), (3, 6, 28), (5, 8, 81), (3, 9, 72), (9, 4, 81),
        (5, 3, 45), (7, 4, 60), (6, 8, 106), (0, 8, 85), (4, 6, 92), (7, 6, 70), (7, 0, 22),
        (7, 1, 73), (4, 3, 64), (5, 1, 80), (2, 1, 22)]
    
    test_hill_climbing(cities_2, distances_2, transition, evaluation_func)

### Task 1.7: Improve hill-climbing with random restarts

def hill_climbing_with_random_restarts(
    cities: int,
    distances: List[Tuple[int]],
    transition: Callable,
    evaluation_func: Callable,
    hill_climbing: Callable,
    repeats: int = 10
) -> List[int]:
    """
    Hill climbing with random restarts finds the solution to reach the goal from the initial.

    Args:
        cities (int): The number of cities to be visited.

        distances (List[Tuple[int]]): The list of distances between every two cities. Each distance
            is represented as a tuple in the form of (c1, c2, d), where c1 and c2 are the two cities
            and d is the distance between them. The length of the list should be equal to cities *
            (cities - 1)/2.

        transition (Callable): The transition function to be used in hill climbing. Will be
            provided on Coursemology.

        evaluation_func (Callable): The evaluation function to be used in hill climbing. Will be
            provided on Coursemology.

        hill_climbing (Callable): The hill climbing function to be used for each restart. Will be
            provided on Coursemology.

        repeats (int): The number of times hill climbing to be repeated. The default value is 10.

    Returns:
        route (List[int]): The shortest route, represented by a list of cities in the order to be
            traversed.
    """
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_1_7():
    def test_random_restarts(cities: int, distances: List[Tuple[int]], transition, evaluation_func, hill_climbing, hill_climbing_with_random_restarts, repeats: int = 10):
        count: int = 0
        def hill_climbing_wrapper(*args, **kwargs):
            nonlocal count
            count += 1
            return hill_climbing(*args, **kwargs)
        route = hill_climbing_with_random_restarts(cities, distances, transition, evaluation_func, hill_climbing_wrapper, repeats)
        assert sorted(route) == list(range(cities)), "New route does not contain all cities present in the original route."
        assert count == repeats, "`hill_climbing` should be called exactly 'repeats' times."
    
    cities_1 = 4
    distances_1 = [(1, 0, 10), (0, 3, 22), (2, 1, 8), (2, 3, 30), (1, 3, 25), (0, 2, 15)]
    
    test_random_restarts(cities_1, distances_1, transition, evaluation_func, hill_climbing, hill_climbing_with_random_restarts)
    
    cities_2 = 10
    distances_2 = [(2, 7, 60), (1, 6, 20), (5, 4, 70), (9, 8, 90), (3, 7, 54), (2, 5, 61),
        (4, 1, 106), (0, 6, 51), (3, 1, 45), (0, 5, 86), (9, 2, 73), (8, 4, 14), (0, 1, 51),
        (9, 7, 22), (3, 2, 22), (8, 1, 120), (5, 7, 92), (5, 6, 60), (6, 2, 10), (8, 3, 78),
        (9, 6, 82), (0, 2, 41), (2, 8, 99), (7, 8, 71), (0, 9, 32), (4, 0, 73), (0, 3, 42),
        (9, 1, 80), (4, 2, 85), (5, 9, 113), (3, 6, 28), (5, 8, 81), (3, 9, 72), (9, 4, 81),
        (5, 3, 45), (7, 4, 60), (6, 8, 106), (0, 8, 85), (4, 6, 92), (7, 6, 70), (7, 0, 22),
        (7, 1, 73), (4, 3, 64), (5, 1, 80), (2, 1, 22)]
    
    test_random_restarts(cities_2, distances_2, transition, evaluation_func, hill_climbing, hill_climbing_with_random_restarts)

### Task 1.8: Comparison between local search and other search algorithms

"""
Run this cell before you start!
"""

import copy
from typing import Callable, Union

import breakthrough

Score = Union[int, float]
Move = tuple[tuple[int, int], tuple[int, int]]
Board = list[list[str]]

def evaluate(board: Board) -> Score:
    """
    Returns the score of the current position.

    Parameters
    ----------
    board: 2D list of lists. Contains characters "B", "W", and "_",
    representing black pawn, white pawn, and empty cell, respectively.

    Returns
    -------
    An evaluation (as a Score).
    """
    bcount = 0
    wcount = 0
    for r, row in enumerate(board):
        for tile in row:
            if tile == "B":
                if r == 5:
                    return breakthrough.WIN
                bcount += 1
            elif tile == "W":
                if r == 0:
                    return -breakthrough.WIN
                wcount += 1
    if wcount == 0:
        return breakthrough.WIN
    if bcount == 0:
        return -breakthrough.WIN
    return bcount - wcount

### Task 2.1: Implement a function to generate all valid moves

def generate_valid_moves(
    board: Board,
    current_player: breakthrough.Player
) -> list[Move]:
    """
    Generates a list containing all possible moves in a particular position for the current player
    to move. Return an empty list if there are no possible moves.

    Parameters
    ----------
    board: 2D list of lists. Contains characters "B", "W", and "_",
        representing black pawn, white pawn, and empty cell, respectively.

    current_player: breakthrough.Player, the colour of the current player to move.

    Returns
    -------
    A list of Moves.
    """
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_2_1():
    board_21 = [
        list("__B___"),
        list("______"),
        list("______"),
        list("______"),
        list("______"),
        list("_W____"),
    ]
    
    board_22 = [
        list("____B_"),
        list("_____B"),
        list("_W____"),
        list("__W___"),
        list("______"),
        list("______"),
    ]
    
    board_23 = [
        list("______"),
        list("__B___"),
        list("_WWW__"),
        list("______"),
        list("______"),
        list("______"),
    ]
    
    assert sorted(generate_valid_moves(board_21, breakthrough.Player.BLACK)) == [((0, 2), (1, 1)), ((0, 2), (1, 2)), ((0, 2), (1, 3))], "Moves generated for black on board_21 is incorrect"
    assert sorted(generate_valid_moves(board_21, breakthrough.Player.WHITE)) == [((5, 1), (4, 0)), ((5, 1), (4, 1)), ((5, 1), (4, 2))], "Moves generated for white on board_21 is incorrect"
    assert sorted(generate_valid_moves(board_22, breakthrough.Player.BLACK)) == [((0, 4), (1, 3)), ((0, 4), (1, 4)), ((1, 5), (2, 4)), ((1, 5), (2, 5))], "Moves generated for black on board_22 is incorrect"
    assert sorted(generate_valid_moves(board_22, breakthrough.Player.WHITE)) == [((2, 1), (1, 0)), ((2, 1), (1, 1)), ((2, 1), (1, 2)), ((3, 2), (2, 2)), ((3, 2), (2, 3))], "Moves generated for white on board_22 is incorrect"
    assert sorted(generate_valid_moves(board_23, breakthrough.Player.BLACK)) == [((1, 2), (2, 1)), ((1, 2), (2, 3))], "Moves generated for black on board_23 is incorrect"
    assert sorted(generate_valid_moves(board_23, breakthrough.Player.WHITE)) == [((2, 1), (1, 0)), ((2, 1), (1, 1)), ((2, 1), (1, 2)), ((2, 2), (1, 1)), ((2, 2), (1, 3)), ((2, 3), (1, 2)), ((2, 3), (1, 3)), ((2, 3), (1, 4))], "Moves generated for white on board_23 is incorrect"

### Task 3.1: Implement minimax

def minimax(
    board: Board,
    depth: int,
    max_depth: int,
    current_player: breakthrough.Player,
    generate_valid_moves: Callable
) -> tuple[Score, Move]:
    """
    Finds the best move for the current player and corresponding evaluation from black's
    perspective for the input board state. Return breakthrough.MOVE_NONE if no move is possible
    (e.g. when the game is over).

    Parameters
    ----------
    board: 2D list of lists. Contains characters "B", "W", and "_",
        representing black pawn, white pawn, and empty cell, respectively. Your function may modify
        the board internally, but the original board passed as an argument must remain unchanged.

    depth: int, the depth to search for the best move. When this is equal to `max_depth`, you
        should get the evaluation of the position using the provided heuristic function.

    max_depth: int, the maximum depth for cutoff.

    current_player: breakthrough.Player, the colour of the current player to move.

    generate_valid_moves: Callable, move generation function. Will be provided on Coursemology.

    Returns
    -------
    A tuple (evaluation, ((src_row, src_col), (dst_row, dst_col))):
    evaluation: the best score that the current player to move can achieve.
    src_row, src_col: position of the pawn to move.
    dst_row, dst_col: position to move the pawn to.
    """
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

preservation_board = [
    list("___B__"),
    list("______"),
    list("_B__B_"),
    list("W____W"),
    list("___W__"),
    list("_W____"),
]

game_over_board_1 = [
    list("______"),
    list("_W____"),
    list("______"),
    list("______"),
    list("______"),
    list("______"),
]

game_over_board_2 = [
    list("______"),
    list("______"),
    list("______"),
    list("______"),
    list("_B____"),
    list("______"),
]

max_depth_board_1 = [
    list("______"),
    list("W_____"),
    list("______"),
    list("_____B"),
    list("______"),
    list("______"),
]

max_depth_board_2 = [
    list("______"),
    list("______"),
    list("W_____"),
    list("______"),
    list("_____B"),
    list("______"),
]

player_switching_board = [
    list("___B__"),
    list("______"),
    list("_B__B_"),
    list("W____W"),
    list("______"),
    list("___W__"),
]

board_31 = [
    list("______"),
    list("___B__"),
    list("____BB"),
    list("___WB_"),
    list("_B__WW"),
    list("_WW___"),
]

board_32 = [
    list("______"),
    list("_____B"),
    list("_W____"),
    list("______"),
    list("______"),
    list("______"),
]

board_33 = [
    list("______"),
    list("__B___"),
    list("W_____"),
    list("___B_B"),
    list("__W___"),
    list("______"),
]

board_34 = [
    list("______"),
    list("____BB"),
    list("__B___"),
    list("_W_B_B"),
    list("__W_W_"),
    list("___WW_"),
]

def invoke_search_fn(search_fn, board, max_depth, current_player):
    if "alpha_beta" in search_fn.__name__:
        return search_fn(board, 0, max_depth, -breakthrough.INF, breakthrough.INF, current_player, generate_valid_moves)
    else:
        return search_fn(board, 0, max_depth, current_player, generate_valid_moves)

def test_board_preservation(search_fn):
    control_board = copy.deepcopy(preservation_board)
    input_board = copy.deepcopy(preservation_board)
    invoke_search_fn(search_fn, input_board, 2, breakthrough.Player.BLACK)
    assert control_board == input_board, "Your function may be modifying the board by making moves in place."

def test_game_over(search_fn, board, expected_score):
    score, move = invoke_search_fn(search_fn, board, 1, breakthrough.Player.BLACK)
    assert score == expected_score, f"Your function might not have terminated when the game is over.\nExpected score: {expected_score}, Actual score: {score}"
    assert move == breakthrough.MOVE_NONE, "Your function might not be returning breakthrough.MOVE_NONE when no moves are possible or it might be generating moves for the opponent instead."

def test_max_depth(search_fn, board, current_player, expected_moves):
    score, move = invoke_search_fn(search_fn, board, 1, current_player)
    assert score == 0, f"Your function may not be terminating at max depth or you may be initialising {current_player.value}'s score incorrectly.\nExpected score: 0, Actual score: {score}"
    assert move in expected_moves, f"Your function does not move {current_player.value} pieces during {current_player.value}'s turn, or initialises {current_player.value}'s score incorrectly or terminates early."

def test_player_switching(search_fn, current_player):
    score, _ = invoke_search_fn(search_fn, player_switching_board, 2, current_player)
    assert score != 2, "Your function may not be switching player's colours correctly after making a move."
    assert score == 0, "Your function may not be making the most optimal move at each depth."

def test_search(search_fn, board, max_depth, current_player, expected_score, expected_moves):
    score, move = invoke_search_fn(search_fn, board, max_depth, current_player)
    assert score == expected_score, f"Final evaluation score should be {expected_score} instead of {score}."
    assert move in expected_moves, f"Your function does not correctly move {current_player.value}'s pieces despite having the correct evaluation."

def test_task_3_1():
    test_board_preservation(minimax)
    test_game_over(minimax, game_over_board_1, -breakthrough.WIN)
    test_game_over(minimax, game_over_board_2, breakthrough.WIN)
    test_max_depth(minimax, max_depth_board_1, breakthrough.Player.BLACK, [((3, 5), (4, 4)), ((3, 5), (4, 5))])
    test_max_depth(minimax, max_depth_board_2, breakthrough.Player.WHITE, [((2, 0), (1, 0)), ((2, 0), (1, 1))])
    test_player_switching(minimax, breakthrough.Player.BLACK)
    test_player_switching(minimax, breakthrough.Player.WHITE)
    
    test_search(minimax, board_31, 1, breakthrough.Player.BLACK, breakthrough.WIN, [((4, 1), (5, 0)), ((4, 1), (5, 2))])
    test_search(minimax, board_32, 4, breakthrough.Player.BLACK, -breakthrough.WIN, [((1, 5), (2, 5)), ((1, 5), (2, 4))])
    test_search(minimax, board_33, 3, breakthrough.Player.WHITE, -breakthrough.WIN, [((2, 0), (1, 0)), ((2, 0), (1, 1))])
    test_search(minimax, board_34, 3, breakthrough.Player.WHITE, -1, [((3, 1), (2, 2)), ((4, 2), (3, 3)), ((4, 4), (3, 3)), ((4, 4), (3, 5))])

### Task 4.1: Integrate alpha-beta pruning into minimax

def minimax_alpha_beta(
    board: Board,
    depth: int,
    max_depth: int,
    alpha: Score,
    beta: Score,
    current_player: breakthrough.Player,
    generate_valid_moves: Callable
) -> tuple[Score, Move]:
    """
    Finds the best move for the current player and corresponding evaluation from black's
    perspective for the input board state. Return breakthrough.MOVE_NONE if no move is possible
    (e.g. when the game is over).

    Parameters
    ----------
    board: 2D list of lists. Contains characters "B", "W", and "_",
        representing black pawn, white pawn, and empty cell, respectively. Your function may modify
        the board internally, but the original board passed as an argument must remain unchanged.

    depth: int, the depth to search for the best move. When this is equal to `max_depth`, you
        should get the evaluation of the position using the provided heuristic function.

    max_depth: int, the maximum depth for cutoff.

    alpha: Score. The alpha value in a given state.

    beta: Score. The beta value in a given state.

    current_player: breakthrough.Player, the colour of the current player
        to move.

    generate_valid_moves: Callable, move generation function. Will be
        provided on Coursemology.

    Returns
    -------
    A tuple (evaluation, ((src_row, src_col), (dst_row, dst_col))):
    evaluation: the best score that the current player to move can achieve.
    src_row, src_col: position of the pawn to move.
    dst_row, dst_col: position to move the pawn to.
    """
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

board_41 = [
    list("______"),
    list("__BB__"),
    list("____BB"),
    list("WBW_B_"),
    list("____WW"),
    list("_WW___"),
]

board_42 = [
    list("____B_"),
    list("__BB__"),
    list("______"),
    list("_WWW__"),
    list("____W_"),
    list("______"),
]

def test_task_4_1():
    test_board_preservation(minimax_alpha_beta)
    test_game_over(minimax_alpha_beta, game_over_board_1, -breakthrough.WIN)
    test_game_over(minimax_alpha_beta, game_over_board_2, breakthrough.WIN)
    test_max_depth(minimax_alpha_beta, max_depth_board_1, breakthrough.Player.BLACK, [((3, 5), (4, 4)), ((3, 5), (4, 5))])
    test_max_depth(minimax_alpha_beta, max_depth_board_2, breakthrough.Player.WHITE, [((2, 0), (1, 0)), ((2, 0), (1, 1))])
    test_player_switching(minimax_alpha_beta, breakthrough.Player.BLACK)
    test_player_switching(minimax_alpha_beta, breakthrough.Player.WHITE)
    
    test_search(minimax_alpha_beta, board_41, 3, breakthrough.Player.BLACK, breakthrough.WIN, [((3, 4), (4, 5))])
    test_search(minimax_alpha_beta, board_42, 6, breakthrough.Player.BLACK, -breakthrough.WIN, [((0, 4), (1, 4)), ((0, 4), (1, 5)), ((1, 2), (2, 2)), ((1, 2), (2, 1)), ((1, 2), (2, 3)), ((1, 3), (2, 3)), ((1, 3), (2, 2)), ((1, 3), (2, 4))])
    test_search(minimax_alpha_beta, board_33, 3, breakthrough.Player.WHITE, -breakthrough.WIN, [((2, 0), (1, 0)), ((2, 0), (1, 1))])
    test_search(minimax_alpha_beta, board_34, 3, breakthrough.Player.WHITE, -1, [((3, 1), (2, 2)), ((4, 2), (3, 3)), ((4, 4), (3, 3)), ((4, 4), (3, 5))])

### Task 5.1: Implement an improved heuristic function

def improved_evaluate(board: Board) -> Score:
    """
    Returns the score of the current position with an improved heuristic.

    Parameters
    ----------
    board: 2D list of lists. Contains characters "B", "W", and "_",
        representing black pawn, white pawn, and empty cell, respectively.

    Returns
    -------
    An improved evaluation (as a Score).
    """
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_5_1():
    board_51 = [
        list("___B__"),
        list("___W__"),
        list("______"),
        list("__B___"),
        list("______"),
        list("______"),
    ]
    
    board_52 = [
        list("___BW_"),
        list("___W__"),
        list("______"),
        list("______"),
        list("______"),
        list("______"),
    ]
    
    board_53 = [
        list("______"),
        list("______"),
        list("______"),
        list("__B___"),
        list("______"),
        list("______"),
    ]
    
    assert improved_evaluate(board_51) == 0, "Your improved evaluation function should return 0 for this board."
    assert improved_evaluate(board_52) == -breakthrough.WIN, "Your improved evaluation function does not correctly evaluate won positions."
    assert improved_evaluate(board_53) == breakthrough.WIN, "Your improved evaluation function does not correctly evaluate won positions."


if __name__ == '__main__':
    test_task_1_3()
    test_task_1_4()
    test_task_1_6()
    test_task_1_7()
    test_task_2_1()
    test_task_3_1()
    test_task_4_1()
    test_task_5_1()