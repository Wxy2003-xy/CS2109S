#include <bits/stdc++.h>
using namespace std;
using namespace chrono;
void transition(tuple<pair<int, int>, pair<int, int>, vector<pair<int, int>>, bool, int> node,
                queue<tuple<pair<int, int>, pair<int, int>, vector<pair<int, int>>, bool, int>>* q, 
                set<tuple<int, int, bool>>* visited);
void transition_no_vi(tuple<pair<int, int>, pair<int, int>, vector<pair<int, int>>, bool, int> node,
                queue<tuple<pair<int, int>, pair<int, int>, vector<pair<int, int>>, bool, int>>* q);
bool is_goal(tuple<pair<int, int>, pair<int, int>, vector<pair<int, int>>, bool, int> node);
vector<pair<int, int>> mnc_search(int m, int c);
vector<pair<int, int>> mnc_search_no_vi(int m, int c);
// GPT generated test cases
int main() {

    vector<pair<int, int>> a(5);
    // Test Case 1: Simple scenario with 3 missionaries and 3 cannibals
    
    int m1 = 3, c1 = 3;
    auto start1 = high_resolution_clock::now();
    vector<pair<int, int>> result1 = mnc_search(m1, c1);
    
    auto end1 = high_resolution_clock::now();
    auto duration1 = duration_cast<milliseconds>(end1 - start1).count();
    cout << "Test Case 1 (3 Missionaries, 3 Cannibals - With Visited): " << duration1 << " ms" << endl;

    auto start1_no_vi = high_resolution_clock::now();
    vector<pair<int, int>> result1_no_vi = mnc_search_no_vi(m1, c1);
    auto end1_no_vi = high_resolution_clock::now();
    auto duration1_no_vi = duration_cast<milliseconds>(end1_no_vi - start1_no_vi).count();
    cout << "Test Case 1 (3 Missionaries, 3 Cannibals - No Visited): " << duration1_no_vi << " ms" << endl;

    // Test Case 2: Edge case with 0 missionaries and 0 cannibals
    int m2 = 0, c2 = 0;
    auto start2 = high_resolution_clock::now();
    vector<pair<int, int>> result2 = mnc_search(m2, c2);
    auto end2 = high_resolution_clock::now();
    auto duration2 = duration_cast<milliseconds>(end2 - start2).count();
    cout << "Test Case 2 (0 Missionaries, 0 Cannibals - With Visited): " << duration2 << " ms" << endl;

    auto start2_no_vi = high_resolution_clock::now();
    vector<pair<int, int>> result2_no_vi = mnc_search_no_vi(m2, c2);
    auto end2_no_vi = high_resolution_clock::now();
    auto duration2_no_vi = duration_cast<milliseconds>(end2_no_vi - start2_no_vi).count();
    cout << "Test Case 2 (0 Missionaries, 0 Cannibals - No Visited): " << duration2_no_vi << " ms" << endl;

    // Test Case 3: 2 missionaries and 2 cannibals
    int m3 = 2, c3 = 2;
    auto start3 = high_resolution_clock::now();
    vector<pair<int, int>> result3 = mnc_search(m3, c3);
    auto end3 = high_resolution_clock::now();
    auto duration3 = duration_cast<milliseconds>(end3 - start3).count();
    cout << "Test Case 3 (2 Missionaries, 2 Cannibals - With Visited): " << duration3 << " ms" << endl;

    auto start3_no_vi = high_resolution_clock::now();
    vector<pair<int, int>> result3_no_vi = mnc_search_no_vi(m3, c3);
    auto end3_no_vi = high_resolution_clock::now();
    auto duration3_no_vi = duration_cast<milliseconds>(end3_no_vi - start3_no_vi).count();
    cout << "Test Case 3 (2 Missionaries, 2 Cannibals - No Visited): " << duration3_no_vi << " ms" << endl;

    // Test Case 4: 1 missionary and 1 cannibal
    int m4 = 1, c4 = 1;
    auto start4 = high_resolution_clock::now();
    vector<pair<int, int>> result4 = mnc_search(m4, c4);
    auto end4 = high_resolution_clock::now();
    auto duration4 = duration_cast<milliseconds>(end4 - start4).count();
    cout << "Test Case 4 (1 Missionary, 1 Cannibal - With Visited): " << duration4 << " ms" << endl;

    auto start4_no_vi = high_resolution_clock::now();
    vector<pair<int, int>> result4_no_vi = mnc_search_no_vi(m4, c4);
    auto end4_no_vi = high_resolution_clock::now();
    auto duration4_no_vi = duration_cast<milliseconds>(end4_no_vi - start4_no_vi).count();
    cout << "Test Case 4 (1 Missionary, 1 Cannibal - No Visited): " << duration4_no_vi << " ms" << endl;

    // Test Case 5: 4 missionaries and 3 cannibals
    int m5 = 4, c5 = 3;
    auto start5 = high_resolution_clock::now();
    vector<pair<int, int>> result5 = mnc_search(m5, c5);
    auto end5 = high_resolution_clock::now();
    auto duration5 = duration_cast<milliseconds>(end5 - start5).count();
    cout << "Test Case 5 (4 Missionaries, 3 Cannibals - With Visited): " << duration5 << " ms" << endl;

    auto start5_no_vi = high_resolution_clock::now();
    vector<pair<int, int>> result5_no_vi = mnc_search_no_vi(m5, c5);
    auto end5_no_vi = high_resolution_clock::now();
    auto duration5_no_vi = duration_cast<milliseconds>(end5_no_vi - start5_no_vi).count();
    cout << "Test Case 5 (4 Missionaries, 3 Cannibals - No Visited): " << duration5_no_vi << " ms" << endl;

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
vector<pair<int, int>> mnc_search_no_vi(int m, int c) {
    queue<tuple<pair<int, int>, pair<int, int>, vector<pair<int, int>>, bool, int>> q;
    vector<pair<int, int>> result;
    q.push(make_tuple(make_pair(0, 0), make_pair(m, c), result, true, 0));

    while (!q.empty()) {
        tuple<pair<int, int>, pair<int, int>, vector<pair<int, int>>, bool, int> state = q.front();
        q.pop();
        if (is_goal(state)) {
            return get<2>(state);
        }
        transition_no_vi(state, &q);
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
void transition_no_vi(tuple<pair<int, int>, pair<int, int>, vector<pair<int, int>>, bool, int> node,
                queue<tuple<pair<int, int>, pair<int, int>, vector<pair<int, int>>, bool, int>>* q) {
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
            vector<pair<int, int>> updated_path = path;
            updated_path.push_back(move);
            q->push(make_tuple(make_pair(new_m_l, new_c_l), make_pair(new_m_r, new_c_r),
                    updated_path, !side, curr_lvl + 1));
        }
    }
}


