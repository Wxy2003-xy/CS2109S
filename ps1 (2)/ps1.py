### Task 1.1  - State Representation
# All possible distribution of m missionaries and c cannibals on both sides of the river.
# Which can be encoded as a string $(0)^{m_l}(1)^{c_l}\#(0)^{m-m_l}(1)^{c-c_l}$ where 0 represents missionaries and 1 represent cannibals

### Task 1.2  - Initial & Goal States
# Initial: m missionaries and c cannibals are all on the right side of the river, $\#(0)^{m}(1)^{c}$
# goal: m missionaries and c cannibals are all on the left side of the river, $(0)^{m}(1)^{c}\#$

### Task 1.3  - Representation Invariant
# All valid state should have 0 or at least as many missionaries as cannibals on both sides of the river
# $\forall (0)^{m_l}(1)^{c_l}\#(0)^{m-m_l}(1)^{c-c_l} \in S (m_l = 0 \lor m_l \geq c_l) \land (m-m_l = 0 \lor m-m_l \geq c-c_l)$
# This can be used as a state validity check to significantly reduce search space, 
# check if the constraint is satisfied before adding the state to the frontier during search

### Task 1.4  - Which Search Algorithm Should We Pick?
# BFS. without visited memory DFS and any variants of tree search that uses DFS may stuck 
# in loop before finding the solution. only BFS (and UCS, which in this case is effecively the same 
# as BFS since actions are assumed to have the same cost and thus cumulative cost == depth) 
# ensures the completeness of the search and guarentees termination

### Task 1.5  - Completeness and Optimality
# O(mc)

# Assuming m >= c, as otherwise the starting state would be invalid. pair each c with a m, all people become 
# $(01)^{c}(0)^{m-c}$

# insert 1 \# in the (01)^{c} segment and (0)^{m-c} each, 
# representing the river, there are $(m-c)\multiply c$ valid partition, thus mc-c^2 valid states, asymptotically O(mc) since m >= c
### Task 1.6  - Implement Search (No Visited Memory)

def mnc_search(m, c):  
    '''
    Solution should be the action taken from the root node (initial state) to 
    the leaf node (goal state) in the search tree.

    Parameters
    ----------    
    m: no. of missionaries
    c: no. of cannibals
    
    Returns
    ----------    
    Returns the solution to the problem as a tuple of steps. Each step is a tuple of two numbers x and y, indicating the number of missionaries and cannibals on the boat respectively as the boat moves from one side of the river to another. If there is no solution, return False.
    '''
    """ YOUR CODE HERE """
    
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_1_6():
    # Note: These solutions are not necessarily unique! (i.e. there may be other optimal solutions.)
    assert mnc_search(2,1) == ((2, 0), (1, 0), (1, 1))
    assert mnc_search(2,2) == ((1, 1), (1, 0), (2, 0), (1, 0), (1, 1))
    assert mnc_search(3,3) == ((1, 1), (1, 0), (0, 2), (0, 1), (2, 0), (1, 1), (2, 0), (0, 1), (0, 2), (1, 0), (1, 1))
    assert mnc_search(0, 0) == False

### Task 1.7 - Implement Search With Visited Memory

def mnc_search_with_visited(m,c):
    '''
    Modify your search algorithm in Task 1.6 by adding visited memory to speed it up!

    Parameters
    ----------    
    m: no. of missionaries
    c: no. of cannibals
    
    Returns
    ----------    
    Returns the solution to the problem as a tuple of steps. Each step is a tuple of two numbers x and y, indicating the number of missionaries and cannibals on the boat respectively as the boat moves from one side of the river to another. If there is no solution, return False.
    '''
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_1_7():
    # Note: These solutions are not necessarily unique! (i.e. there may be other optimal solutions.)
    assert mnc_search_with_visited(2,1) == ((2, 0), (1, 0), (1, 1))
    assert mnc_search_with_visited(2,2) == ((1, 1), (1, 0), (2, 0), (1, 0), (1, 1))
    assert mnc_search_with_visited(3,3) == ((1, 1), (1, 0), (0, 2), (0, 1), (2, 0), (1, 1), (2, 0), (0, 1), (0, 2), (1, 0), (1, 1))
    assert mnc_search_with_visited(0,0) == False

### Task 1.8 - Search With vs Without Visited Memory

import copy
import heapq
import math
import os
import random
import sys
import time

import utils
import cube

from typing import List, Tuple, Callable
from functools import partial

"""
We provide implementations for the Node and PriorityQueue classes in utils.py, but you can implement your own if you wish
"""
from utils import Node
from utils import PriorityQueue

### Task 2.1 - Design a heuristic for A* Search

def heuristic_func(problem: cube.Cube, state: cube.State) -> float:
    r"""
    Computes the heuristic value of a state
    
    Args:
        problem (cube.Cube): the problem to compute
        state (cube.State): the state to be evaluated
        
    Returns:
        h_n (float): the heuristic value 
    """
    h_n = 0.0
    goals = problem.goal

    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

    return h_n

# goal state
cube_goal = {
    'initial': [['N', 'U', 'S'],
                ['N', 'U', 'S'],
                ['N', 'U', 'S']],
    'goal': [['N', 'U', 'S'],
             ['N', 'U', 'S'],
             ['N', 'U', 'S']],
    'solution': [],
}

# one step away from goal state
cube_one_step = {
    'initial': [['S', 'N', 'U'],
                ['N', 'U', 'S'],
                ['N', 'U', 'S']],
    'goal': [['N', 'U', 'S'],
             ['N', 'U', 'S'],
             ['N', 'U', 'S']],
    'solution': [[0, 'left']],
}

# transposes the cube
cube_transpose = {
    'initial': [['S', 'O', 'C'],
                ['S', 'O', 'C'],
                ['S', 'O', 'C']],
    'goal': [['S', 'S', 'S'],
             ['O', 'O', 'O'],
             ['C', 'C', 'C']],
    'solution': [[2, 'right'], [1, 'left'], [1, 'down'], [2, 'up']],
}

# flips the cube
cube_flip = {
    'initial': [['N', 'U', 'S'],
                ['N', 'U', 'S'],
                ['N', 'U', 'S']],
    'goal': [['S', 'U', 'N'],
             ['N', 'S', 'U'],
             ['U', 'N', 'S']],
    'solution': [[0, 'left'], [1, 'right'], [0, 'up'], [1, 'down']],
}

# intermediate state for cube_flip
cube_flip_intermediate = {
    'initial': [['U', 'S', 'N'],
                ['N', 'U', 'S'],
                ['N', 'U', 'S']],
    'goal': [['S', 'U', 'N'],
             ['N', 'S', 'U'],
             ['U', 'N', 'S']],
    'solution': [[1, 'right'], [0, 'up'], [1, 'down']],
}


# 3x4 cube
cube_3x4 = {
    'initial': [[1, 1, 9, 0],
                [2, 2, 0, 2],
                [9, 0, 1, 9]],
    'goal': [[1, 0, 9, 2],
             [2, 1, 0, 9],
             [2, 1, 0, 9]],
    'solution': [[1, 'down'], [3, 'up'], [2, 'left']],
}

def test_task_2_1():
    def test_heuristic(heuristic_func, case):
        problem = cube.Cube(cube.State(case['initial']), cube.State(case['goal']))
        assert heuristic_func(problem, problem.goal) == 0, "Heuristic is not 0 at the goal state"
        assert heuristic_func(problem, problem.initial) <= len(case['solution']), "Heuristic is not admissible"
    
    test_heuristic(heuristic_func, cube_goal)
    test_heuristic(heuristic_func, cube_one_step)
    test_heuristic(heuristic_func, cube_transpose)
    test_heuristic(heuristic_func, cube_flip)
    test_heuristic(heuristic_func, cube_flip_intermediate)
    test_heuristic(heuristic_func, cube_3x4)

### Task 2.2 - Implement A* search 

def astar_search(problem: cube.Cube, heuristic_func: Callable):
    r"""
    A* Search finds the solution to reach the goal from the initial.
    If no solution is found, return False.
    
    Args:
        problem (cube.Cube): Cube instance
        heuristic_func (Callable): heuristic function for the A* search

    Returns:
        solution (List[Action]): the action sequence
    """
    fail = True
    solution = []

    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """
    
    if fail:
        return False
    return solution

def test_search(algorithm, case):
    problem = cube.Cube(cube.State(case['initial']), cube.State(case['goal']))
    start_time = time.perf_counter()
    solution = algorithm(problem)
    print(f"{algorithm.__name__}(goal={case['goal']}) took {time.perf_counter() - start_time:.4f} seconds")
    if solution is False:
        assert case['solution'] is False
        return
    verify_output = problem.verify_solution(solution, _print=False)
    assert verify_output['valid'], f"Fail to reach goal state with solution {solution}"
    assert verify_output['cost'] <= len(case['solution']), f"Cost is not optimal."

def test_task_2_2():
    def astar_heuristic_search(problem): 
        return astar_search(problem, heuristic_func=heuristic_func)
        
    test_search(astar_heuristic_search, cube_goal)
    test_search(astar_heuristic_search, cube_one_step)
    test_search(astar_heuristic_search, cube_transpose)
    test_search(astar_heuristic_search, cube_flip)
    test_search(astar_heuristic_search, cube_flip_intermediate)
    test_search(astar_heuristic_search, cube_3x4)

### Task 2.3 - Consistency & Admissibility

### Task 2.4 - Implement Uninformed Search

def uninformed_search(problem: cube.Cube):
    r"""
    Uninformed Search finds the solution to reach the goal from the initial.
    If no solution is found, return False.
    
    Args:
        problem (cube.Cube): Cube instance

    Returns:
        solution (List[Action]): the action sequence
    """
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_2_4():
    test_search(uninformed_search, cube_goal)
    test_search(uninformed_search, cube_one_step)
    test_search(uninformed_search, cube_transpose)
    test_search(uninformed_search, cube_flip)
    test_search(uninformed_search, cube_flip_intermediate)
    test_search(uninformed_search, cube_3x4)

### Task 2.5 - Uninformed vs Informed Search


if __name__ == '__main__':
    test_task_1_6()
    test_task_1_7()
    test_task_2_1()
    test_task_2_2()
    test_task_2_4()