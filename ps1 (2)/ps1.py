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

# #include <bits/stdc++.h>
# using namespace std;

# void transition(tuple<pair<int, int>, pair<int, int>, vector<pair<int, int>>, bool, int> node,
#                 queue<tuple<pair<int, int>, pair<int, int>, vector<pair<int, int>>, bool, int>>* q, 
#                 set<tuple<int, int, bool>>* visited);
# bool is_goal(tuple<pair<int, int>, pair<int, int>, vector<pair<int, int>>, bool, int> node);
# vector<pair<int, int>> mnc_search(int m, int c);
# // GPT generated test cases
# int main() {
# }

# vector<pair<int, int>> mnc_search(int m, int c) {
#     queue<tuple<pair<int, int>, pair<int, int>, vector<pair<int, int>>, bool, int>> q;
#     set<tuple<int, int, bool>> visited;
#     vector<pair<int, int>> result;
#     q.push(make_tuple(make_pair(0, 0), make_pair(m, c), result, true, 0));
#     visited.insert(make_tuple(0, 0, true));

#     while (!q.empty()) {
#         tuple<pair<int, int>, pair<int, int>, vector<pair<int, int>>, bool, int> state = q.front();
#         q.pop();
#         if (is_goal(state)) {
#             return get<2>(state);
#         }
#         transition(state, &q, &visited);
#     }
#     return result;
# }
# //                      left                right           path           side   lvl   
# bool is_goal(tuple<pair<int, int>, pair<int, int>, vector<pair<int, int>>, bool, int> node) {
#     return get<1>(node) == make_pair(0,0);
# }
# //                      left               right            path               side lvl
# void transition(tuple<pair<int, int>, pair<int, int>, vector<pair<int, int>>, bool, int> node,
#                 queue<tuple<pair<int, int>, pair<int, int>, vector<pair<int, int>>, bool, int>>* q, 

#                 set<tuple<int, int, bool>>* visited) {
#     int m_l = get<0>(node).first;
#     int m_r = get<1>(node).first;
#     int c_l = get<0>(node).second;
#     int c_r = get<1>(node).second;
#     vector<pair<int, int>> path = get<2>(node);
#     bool side = get<3>(node);
#     int curr_lvl = get<4>(node);

#     vector<pair<int, int>> moves = {{2, 0}, {0, 2}, {1, 0}, {0, 1}, {1, 1}};
#     for (auto move : moves) {
#         int dm = move.first; 
#         int dc = move.second;
#         int new_m_l = side ? m_l + dm : m_l - dm;
#         int new_c_l = side ? c_l + dc : c_l - dc;
#         int new_m_r = side ? m_r - dm : m_r + dm;
#         int new_c_r = side ? c_r - dc : c_r + dc;
#         if (new_m_l >= 0 && new_c_l >= 0 && new_m_r >= 0 && new_c_r >= 0 &&
#             (new_m_l >= new_c_l || new_m_l == 0) &&
#             (new_m_r >= new_c_r || new_m_r == 0)) {
#             tuple<int, int, bool> mem = make_tuple(new_m_l, new_c_l, !side);
#             if (visited->find(mem) == visited->end()) {
#                 visited->insert(mem);
#                 vector<pair<int, int>> updated_path = path;
#                 updated_path.push_back(move);
#                 q->push(make_tuple(make_pair(new_m_l, new_c_l), make_pair(new_m_r, new_c_r),
#                         updated_path, !side, curr_lvl + 1));
#             }
#         }
#     }
# }
# from collections import deque
# from typing import Tuple, List, Deque, Union

# State = Tuple[Tuple[int, int], Tuple[int, int], List[Tuple[int, int]], bool, int]
# StateKey = Tuple[int, int, bool]
# def valid(new_m_l:int, new_c_l:int, new_m_r:int, new_c_r:int) -> bool:
#     return (new_m_l >= 0 and new_c_l >= 0 and new_m_r >= 0 and new_c_r >= 0 
#             and (new_m_l >= new_c_l or new_m_l == 0) and (new_m_r >= new_c_r or new_m_r == 0))
# def transition(node: State, q: Deque[State]) -> None:
#     m_l: int = node[0][0]
#     c_l: int = node[0][1]
#     m_r: int = node[1][0] 
#     c_r: int = node[1][1]
#     path: List[Tuple[int, int]] = node[2]
#     side: bool = node[3]
#     curr_lvl: int = node[4]
#     moves: List[Tuple[int, int]] = [(2,0), (0,2), (1,0), (0,1), (1,1)]
#     for move in moves: 
#         dm:int = move[0]
#         dc:int = move[1]
#         new_m_l:int = m_l + dm if side else m_l - dm
#         new_c_l:int = c_l + dc if side else c_l - dc
#         new_m_r:int = m_r - dm if side else m_r + dm
#         new_c_r:int = c_r - dc if side else c_r + dc
#         if (valid(new_m_l, new_c_l, new_m_r, new_c_r)): 
#             updated_path = path.copy()
#             updated_path.append(move)
#             new_state: State = ((new_m_l, new_c_l), (new_m_r, new_c_r), updated_path, not side, curr_lvl + 1)
#             q.append(new_state)

# def transition_mem(node: State, q: Deque[State], visited: set[StateKey]) -> None: 
#     m_l: int = node[0][0]
#     c_l: int = node[0][1]
#     m_r: int = node[1][0] 
#     c_r: int = node[1][1]
#     path: List[Tuple[int, int]] = node[2]
#     side: bool = node[3]
#     curr_lvl: int = node[4]
#     moves: List[Tuple[int, int]] = [(2,0), (0,2), (1,0), (0,1), (1,1)]
#     for move in moves: 
#         dm:int = move[0]
#         dc:int = move[1]
#         new_m_l:int = m_l + dm if side else m_l - dm
#         new_c_l:int = c_l + dc if side else c_l - dc
#         new_m_r:int = m_r - dm if side else m_r + dm
#         new_c_r:int = c_r - dc if side else c_r + dc
#         if (valid(new_m_l, new_c_l, new_m_r, new_c_r)): 
#             mem:Tuple[int, int, bool] = (new_m_l, new_c_l, not side)
#             if mem not in visited:
#                 visited.add(mem)
#                 updated_path = path.copy()
#                 updated_path.append(move)
#                 new_state: State = ((new_m_l, new_c_l), (new_m_r, new_c_r), updated_path, not side, curr_lvl + 1)
#                 q.append(new_state)
# def is_goal(node: State) -> bool:
#     return node[1] == (0,0)
# def mnc_search(m, c) -> Union[List[Tuple[int, int]], bool]:  
#     q: Deque[State] = deque()
#     result: List[Tuple[int, int]] = []
#     q.append(((0, 0), (m, c), result, True, 0))
#     while q:
#         state: State = q.popleft()
#         if is_goal(state): 
#             if state[2] == []:
#                 return False
#             return state[2]
#         transition(state, q)
#     return False
#     '''
#     Solution should be the action taken from the root node (initial state) to 
#     the leaf node (goal state) in the search tree.

#     Parameters
#     ----------    
#     m: no. of missionaries
#     c: no. of cannibals
    
#     Returns
#     ----------    
#     Returns the solution to the problem as a tuple of steps. Each step is a tuple of two numbers x and y, indicating the number of missionaries and cannibals on the boat respectively as the boat moves from one side of the river to another. If there is no solution, return False.
#     '''
#     """ YOUR CODE HERE """
    
#     raise NotImplementedError
#     """ YOUR CODE END HERE """

# def test_task_1_6():
#     # Note: These solutions are not necessarily unique! (i.e. there may be other optimal solutions.)
#     assert mnc_search(2,1) == ((2, 0), (1, 0), (1, 1))
#     assert mnc_search(2,2) == ((1, 1), (1, 0), (2, 0), (1, 0), (1, 1))
#     assert mnc_search(3,3) == ((1, 1), (1, 0), (0, 2), (0, 1), (2, 0), (1, 1), (2, 0), (0, 1), (0, 2), (1, 0), (1, 1))
#     assert mnc_search(0, 0) == False

# ### Task 1.7 - Implement Search With Visited Memory

# def mnc_search_with_visited(m,c):
#     q: Deque[State] = deque()
#     visited: set[StateKey] = set()
#     result: List[Tuple[int, int]] = []
#     q.append(((0, 0), (m, c), result, True, 0))
#     while q:
#         state: State = q.popleft()
#         if is_goal(state): 
#             if state[2] == []:
#                 return False
#             return state[2]
#         transition_mem(state, q, visited)
#     return False
#     '''
#     Modify your search algorithm in Task 1.6 by adding visited memory to speed it up!

#     Parameters
#     ----------    
#     m: no. of missionaries
#     c: no. of cannibals
    
#     Returns
#     ----------    
#     Returns the solution to the problem as a tuple of steps. Each step is a tuple of two numbers x and y, indicating the number of missionaries and cannibals on the boat respectively as the boat moves from one side of the river to another. If there is no solution, return False.
#     '''
#     """ YOUR CODE HERE """
#     raise NotImplementedError
#     """ YOUR CODE END HERE """

# def test_task_1_7():
#     # Note: These solutions are not necessarily unique! (i.e. there may be other optimal solutions.)
#     assert mnc_search_with_visited(2,1) == ((2, 0), (1, 0), (1, 1))
#     assert mnc_search_with_visited(2,2) == ((1, 1), (1, 0), (2, 0), (1, 0), (1, 1))
#     assert mnc_search_with_visited(3,3) == ((1, 1), (1, 0), (0, 2), (0, 1), (2, 0), (1, 1), (2, 0), (0, 1), (0, 2), (1, 0), (1, 1))
#     assert mnc_search_with_visited(0,0) == False

# ### Task 1.8 - Search With vs Without Visited Memory

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


def unflatten_2d(iterable, shape):
    rows, cols = shape
    matrix = []
    for row in range(rows):
        matrix.append(iterable[row * cols : row * cols + cols])
    return matrix
def heuristic_func(problem: cube.Cube, state: cube.State) -> float:
    r"""
    Computes the heuristic value of a state
    
    Args:
        problem (cube.Cube): the problem to compute
        state (cube.State): the state to be evaluated
        
    Returns:
        h_n (float): the heuristic value 
    """
    state2d = cube.unflatten_2d(state.layout, state.shape)
    r = len(state2d)
    c = len(state2d[0])
    hashmap = dict()
    for i in range(0, r):
        for j in range(0, c):
            if (state2d[i][j] not in hashmap): 
                hashmap[state2d[i][j]] = -1

    h_n:float = 0.0
    goals = problem.goal
    goals2d = cube.unflatten_2d(goals.layout, goals.shape)
    for i in range(len(state2d)):
        for j in range(len(state2d[0])):
            goal_val = goals2d[i][j]
            if (state2d[i][j] == goal_val):
                continue
            else:
                n = 0
                for p in range(len(state2d)):
                    for q in range(len(state2d[0])):
                        if (state2d[p][q] == goal_val 
                            and (r * p + q > hashmap[goal_val]) 
                            and (state2d[p][q] != goals2d[p][q])):
                            n = r * p + q
                            hashmap[goal_val] = n
            h_n += min(abs(n % c - j), abs(j - n % c))
            h_n += min(abs(n / c - i), abs(i - n / c))
            print(h_n / (r * c))
    return h_n / (r * c)

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
        print(case['initial'] + case['goal'])
        print(heuristic_func(problem, problem.goal))
        assert heuristic_func(problem, problem.goal) == 0, "Heuristic is not 0 at the goal state"
        assert heuristic_func(problem, problem.initial) <= len(case['solution']), "Heuristic is not admissible"
    
    test_heuristic(heuristic_func, cube_goal)
    test_heuristic(heuristic_func, cube_one_step)
    test_heuristic(heuristic_func, cube_transpose)
    test_heuristic(heuristic_func, cube_flip)
    test_heuristic(heuristic_func, cube_flip_intermediate)
    test_heuristic(heuristic_func, cube_3x4)

class State:
    r"""State class describes the setting of the Cube

    Args:
         layout (List[List[int]]): a 2-D list that represents the layout of the cube's faces.

    Example:
        state = State([1,2,3],[4,5,6])
        This represents the state with a layout like:
            label:    0   1   2
                0   | 1 | 2 | 3 |
                1   | 4 | 5 | 6 |

    Methods:
        left(label): move the @label row left
            returns the copy of new state (State)

        right(label): move the @label row right
            returns the copy of new state (State)

        up(label): move the @label col up
            returns the copy of new state (State)

        down(label): move the @label col down
            returns the copy of new state (State)
    """

    def __init__(self, layout: List[List[int]]):
        self.__layout = flatten(layout)
        self.__shape = []
        while isinstance(layout, list):
            self.__shape.append(len(layout))
            layout = layout[0]

    def __eq__(self, state: "State"):
        if isinstance(state, State):
            same_shape = (
                state.shape[0] == self.__shape[0] and state.shape[1] == self.__shape[1]
            )
            same_layout = all([x == y for x, y in zip(self.__layout, state.layout)])
            return same_shape and same_layout
        else:
            return False

    def __hash__(self) -> int:
        return hash(tuple(self.__layout))

    def __repr__(self) -> str:
        return str({"shape": self.__shape, "layout": self.__layout})

    def __str__(self):
        # Header
        row_str = f"{' '*5} "
        for col in range(self.shape[1]):
            row_str += f"{col:^5d} "
        cube_str = row_str + "\n"
        cube_str += f"{' '*5}+{'-----+'*self.shape[1]}\n"
        # Content
        for row in range(self.shape[0]):
            row_str = f"{row:^5d}|"
            for col in range(self.shape[1]):
                row_str += f"{str(self.layout[row*self.shape[1]+col]):^5s}|"
            cube_str += row_str + "\n"
            cube_str += f"{' '*5}+{'-----+'*self.shape[1]}\n"

        return cube_str

    @property
    def shape(self):
        return copy.deepcopy(self.__shape)

    @property
    def layout(self):
        return copy.deepcopy(self.__layout)

    def left(self, label):
        layout = self.layout
        rows, cols = self.shape
        head = layout[label * cols]
        for i in range(cols - 1):
            layout[label * cols + i] = layout[label * cols + i + 1]
        layout[(label + 1) * cols - 1] = head
        return State(unflatten_2d(layout, self.shape))

    def right(self, label):
        layout = self.layout
        rows, cols = self.shape
        tail = layout[(label + 1) * cols - 1]
        for i in range(cols - 1, 0, -1):
            layout[label * cols + i] = layout[label * cols + i - 1]
        layout[label * cols] = tail
        return State(unflatten_2d(layout, self.shape))

    def up(self, label):
        layout = self.layout
        rows, cols = self.shape
        head = layout[label]
        for i in range(rows - 1):
            layout[label + cols * i] = layout[label + cols * (i + 1)]
        layout[label + cols * (rows - 1)] = head
        return State(unflatten_2d(layout, self.shape))

    def down(self, label):
        layout = self.layout
        rows, cols = self.shape
        tail = layout[label + cols * (rows - 1)]
        for i in range(rows - 1, 0, -1):
            layout[label + cols * i] = layout[label + cols * (i - 1)]
        layout[label] = tail
        return State(unflatten_2d(layout, self.shape))


### Task 2.2 - Implement A* search 
import heapq
from typing import Dict, List, Optional, Tuple, Union
from itertools import product
Action = List[Union[int, str]]
def unflatten_2d(iterable, shape):
    rows, cols = shape
    matrix = []
    for row in range(rows):
        matrix.append(iterable[row * cols : row * cols + cols])
    return matrix
def transition(cube, state, action_set, cur, pq, visited, heuristic_func):
    cost: float = cur[0]
    cur_path: List[Action] = cur[1]
    cur_config: State = cur[2]
    config = cur_config.layout
    for i in range(len(action_set)):
        new_config = getattr(cur_config, action_set[i][1])(action_set[i][0]) 
        new_cost = cost + 1.0 + heuristic_func(cube, new_config)
        if (new_config not in visited or visited[new_config] > new_cost):
            visited[new_config] = new_cost
            new_path = cur_path.copy() + [(action_set[i][0], action_set[i][1])]
            new_state = (new_cost, new_path, new_config)
            heapq.heappush(pq, new_state)
    
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
    vert_move = ["left", "right"]  # ✅ Store method names
    hori_move = ["up", "down"]
    hori_dim = []
    for i in range(len(unflatten_2d(problem.initial.layout, problem.initial.shape))):
        hori_dim.append(i) 
    vert_dim = []
    for i in range(len(unflatten_2d(problem.initial.layout, problem.initial.shape)[0])):
        vert_dim.append(i)
    action_set = list(product(hori_dim, vert_move)) + list(product(vert_dim, hori_move))
    #             cost,   partial_path, cur_config
    fail = True
    solution = []
    pq = []
    visited = dict()
    heapq.heappush(pq, (0.0, [], problem.initial))
    visited[problem.initial] = 0.0
    while pq:
        cur = heapq.heappop(pq)
        if (problem.goal_test(cur[2])):
            return cur[1]
        transition(problem, cur[2], action_set, cur, pq, visited, heuristic_func)
    return False
    
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
    # test_task_1_6()
    # test_task_1_7()
    test_task_2_1()
    test_task_2_2()
    test_task_2_4()