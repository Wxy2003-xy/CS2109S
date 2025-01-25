#include <bits/stdc++.h>
using namespace std;

void transition(tuple<pair<int, int>, pair<int, int>, vector<pair<int, int>>, bool, int> node,
                queue<tuple<pair<int, int>, pair<int, int>, vector<pair<int, int>>, bool, int>>* q, 
                set<tuple<int, int, bool>>* visited);
bool is_goal(tuple<pair<int, int>, pair<int, int>, vector<pair<int, int>>, bool, int> node);
vector<pair<int, int>> mnc_search(int m, int c);
// GPT generated test cases
int main() {
    // Test Case 1: Simple scenario with 3 missionaries and 3 cannibals
    int m1 = 3, c1 = 3;
    vector<pair<int, int>> result1 = mnc_search(m1, c1);
    cout << "Test Case 1 (3 Missionaries, 3 Cannibals):" << endl;
    if (result1.empty()) {
        cout << "No solution found!" << endl;
    } else {
        for (auto move : result1) {
            cout << "Move: (" << move.first << ", " << move.second << ")" << endl;
        }
    }
    cout << endl;

    // Test Case 2: Edge case with 0 missionaries and 0 cannibals
    int m2 = 0, c2 = 0;
    vector<pair<int, int>> result2 = mnc_search(m2, c2);
    cout << "Test Case 2 (0 Missionaries, 0 Cannibals):" << endl;
    if (result2.empty()) {
        cout << "No solution needed, already solved!" << endl;
    } else {
        for (auto move : result2) {
            cout << "Move: (" << move.first << ", " << move.second << ")" << endl;
        }
    }
    cout << endl;

    // Test Case 3: 2 missionaries and 2 cannibals
    int m3 = 2, c3 = 2;
    vector<pair<int, int>> result3 = mnc_search(m3, c3);
    cout << "Test Case 3 (2 Missionaries, 2 Cannibals):" << endl;
    if (result3.empty()) {
        cout << "No solution found!" << endl;
    } else {
        for (auto move : result3) {
            cout << "Move: (" << move.first << ", " << move.second << ")" << endl;
        }
    }
    cout << endl;

    // Test Case 4: 1 missionary and 1 cannibal
    int m4 = 1, c4 = 1;
    vector<pair<int, int>> result4 = mnc_search(m4, c4);
    cout << "Test Case 4 (1 Missionary, 1 Cannibal):" << endl;
    if (result4.empty()) {
        cout << "No solution found!" << endl;
    } else {
        for (auto move : result4) {
            cout << "Move: (" << move.first << ", " << move.second << ")" << endl;
        }
    }
    cout << endl;
    // Test Case 5: 4 missionaries and 3 cannibals
    int m5 = 4, c5 = 3;
    vector<pair<int, int>> result5 = mnc_search(m5, c5);
    cout << "Test Case 5 (4 Missionaries, 3 Cannibals):" << endl;
    if (result5.empty()) {
        cout << "No solution found!" << endl;
    } else {
        for (auto move : result5) {
            cout << "Move: (" << move.first << ", " << move.second << ")" << endl;
        }
    }
    cout << endl;
    return 0;
}


vector<pair<int, int>> mnc_search(int m, int c) {
    queue<tuple<pair<int, int>, pair<int, int>, vector<pair<int, int>>, bool, int>> q;
    set<tuple<int, int, bool>> visited;
    vector<pair<int, int>> result;
    q.push(make_tuple(make_pair(0, 0), make_pair(m, c), result, true, 0));
    visited.insert(make_tuple(0, 0, true));

    while (!q.empty()) {
        tuple<pair<int, int>, pair<int, int>, vector<pair<int, int>>, bool, int> state = q.front();
        q.pop();
        if (is_goal(state)) {
            return get<2>(state);
        }
        transition(state, &q, &visited);
    }
    return result;
}
//                      left                right           path           side   lvl   
bool is_goal(tuple<pair<int, int>, pair<int, int>, vector<pair<int, int>>, bool, int> node) {
    return get<1>(node) == make_pair(0,0);
}
//                      left               right            path               side lvl
void transition(tuple<pair<int, int>, pair<int, int>, vector<pair<int, int>>, bool, int> node,
                queue<tuple<pair<int, int>, pair<int, int>, vector<pair<int, int>>, bool, int>>* q, 

                set<tuple<int, int, bool>>* visited) {
    int m_l = get<0>(node).first;
    int m_r = get<1>(node).first;
    int c_l = get<0>(node).second;
    int c_r = get<1>(node).second;
    vector<pair<int, int>> path = get<2>(node);
    bool side = get<3>(node);
    int curr_lvl = get<4>(node);

    vector<pair<int, int>> moves = {{2, 0}, {0, 2}, {1, 0}, {0, 1}, {1, 1}};
    for (auto move : moves) {
        int dm = move.first; 
        int dc = move.second;
        int new_m_l = side ? m_l + dm : m_l - dm;
        int new_c_l = side ? c_l + dc : c_l - dc;
        int new_m_r = side ? m_r - dm : m_r + dm;
        int new_c_r = side ? c_r - dc : c_r + dc;
        if (new_m_l >= 0 && new_c_l >= 0 && new_m_r >= 0 && new_c_r >= 0 &&
            (new_m_l >= new_c_l || new_m_l == 0) &&
            (new_m_r >= new_c_r || new_m_r == 0)) {
            tuple<int, int, bool> mem = make_tuple(new_m_l, new_c_l, !side);
            if (visited->find(mem) == visited->end()) {
                visited->insert(mem);
                vector<pair<int, int>> updated_path = path;
                updated_path.push_back(move);
                q->push(make_tuple(make_pair(new_m_l, new_c_l), make_pair(new_m_r, new_c_r),
                        updated_path, !side, curr_lvl + 1));
            }
        }
    }
}


