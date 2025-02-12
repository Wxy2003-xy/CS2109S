#include <bits/stdc++.h>

using namespace std;
using namespace chrono;
struct StateVectorHash { // hash the state, to be used to keep visited mem
    size_t operator()(const vector<vector<int>>& v) const {
        size_t hash_value = 0;
        for (size_t row = 0; row < v.size(); row++) {
            for (size_t col = 0; col < v[row].size(); col++) {
                hash_value ^= std::hash<int>{}(v[row][col]) + 0x9e3779b9 + (hash_value << 6) + (hash_value >> 2);
                // gpt generated hash func
            }
        }
        return hash_value;
    }
};

class Cube {
    public:
        int r;
        int c;
        vector<pair<int, int>> moves;

        Cube(int r, int c) {
            this->r = r;
            this->c = c;
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < 2; j++) {
                    moves.push_back(make_pair(i, j));
                }
            }
            for (int i = 0; i < c; i++) {
                for (int j = 2; j < 4; j++) {
                    moves.push_back(make_pair(i, j));
                }
            }
        }
        vector<vector<int>> action(vector<vector<int>> new_config, int idx, int dir) {
            if (dir == 0) { // left
                rotate(new_config[idx].begin(), new_config[idx].begin() + 1, new_config[idx].end());
            } 
            else if (dir == 1) { // right
                rotate(new_config[idx].rbegin(), new_config[idx].rbegin() + 1, new_config[idx].rend());
            } 
            else if (dir == 2) { // up
                vector<int> temp(r);
                for (int i = 0; i < r; i++) temp[i] = new_config[i][idx];
                rotate(temp.begin(), temp.begin() + 1, temp.end());
                for (int i = 0; i < r; i++) new_config[i][idx] = temp[i];
            } 
            else if (dir == 3) { // down
                vector<int> temp(r);
                for (int i = 0; i < r; i++) temp[i] = new_config[i][idx];
                rotate(temp.rbegin(), temp.rbegin() + 1, temp.rend());
                for (int i = 0; i < r; i++) new_config[i][idx] = temp[i];
            }
            return new_config;
        }


        bool is_goal(const vector<vector<int>>& config, vector<vector<int>> goal) {
            if (config.size() != goal.size() || config[0].size() != goal[0].size()) {
                return false;
            }

            for (size_t i = 0; i < config.size(); i++) {
                for (size_t j = 0; j < config[i].size(); j++) {
                    if (config[i][j] != goal[i][j]) {
                        return false;
                    }
                }
            }
            return true;
        }
        void transition(tuple<vector<vector<int>>, vector<pair<int, int>>, int> state,
                queue<tuple<vector<vector<int>>, vector<pair<int, int>>, int>>* q, 
                set<vector<vector<int>>>* visited) {
            vector<vector<int>> curr_config = get<0>(state);
            vector<pair<int, int>> curr_path = get<1>(state);
            int curr_lvl = get<2>(state);
            for (auto move : moves) {
                vector<vector<int>> new_config = action(curr_config, move.first, move.second);
                if (visited->find(new_config) == visited->end()) {  // not in set
                    visited->insert(new_config);
                    vector<pair<int, int>> new_path(curr_path);
                    new_path.push_back(move);
                    q->push(make_tuple(new_config, new_path, curr_lvl+1));
                } else {
                    // if (new_config)
                }
            }
        }
        static bool pq_compare(const tuple<vector<vector<int>>, vector<pair<int, int>>, int>& a,
                        const tuple<vector<vector<int>>, vector<pair<int, int>>, int>& b) {
            return get<2>(a) > get<2>(b);
        }

        static bool astar_compare(const tuple<vector<vector<int>>, vector<pair<int, int>>, double>& a,
                                  const tuple<vector<vector<int>>, vector<pair<int, int>>, double>& b) {
            return get<2>(a) > get<2>(b);
        }
        void usc_transition(tuple<vector<vector<int>>, vector<pair<int, int>>, int> state,
                            priority_queue<tuple<vector<vector<int>>, vector<pair<int, int>>, int>,
                                            vector<tuple<vector<vector<int>>, vector<pair<int, int>>, int>>,
                                        decltype(&pq_compare)>& q,
                            unordered_map<vector<vector<int>>, int, StateVectorHash>& visited_cost) {

            vector<vector<int>> curr_config = get<0>(state);
            vector<pair<int, int>> curr_path = get<1>(state);
            int curr_lvl = get<2>(state);
            for (auto move : moves) {
                vector<vector<int>> new_config = action(curr_config, move.first, move.second);
                int new_cost = curr_lvl + 1;

                // Check if the state is new or reached with a lower cost
                if (!visited_cost.count(new_config) || visited_cost[new_config] > new_cost) {
                    visited_cost[new_config] = new_cost;
                    vector<pair<int, int>> new_path = curr_path;
                    new_path.push_back(move);
                    q.push(make_tuple(new_config, new_path, new_cost));
                } else {
                }
            }
        }
        //! ==========================================================================================
                // using heuristics
        double heuristic(const vector<vector<int>>& curr, const vector<vector<int>>& goal) {
            vector<int> misplaced_idx(r, -1); // to keep track of last misplaced element, so to not using it again
            double cost = 0;
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    int g = goal[i][j]; // what it should be
                    if (curr[i][j] == g) {
                        continue; // not misplaced, cont;
                    } else {
                        int n = 0; 
                        for (int p = 0; p < r; p++) {
                            for (int q = 0; q < c; q++) {
                                if (curr[p][q] == g // find a tile that should be placed here
                                        && (r * p + q > misplaced_idx[g]) // its index should be larger than the index of 
                                                                          // the last tile of the same type being used to replace misplaced tile
                                        && curr[p][q] != goal[p][q]) { // this tile will be moved thus it shoulnt be at the place it should be as well
                                    n = r * p + q; // if find one, update the index of the tile to replace the misplaced tile
                                    misplaced_idx[g] = n;
                                } 
                            }
                        }
                        cost += min(abs(n % c - j), abs(j - n % c)); // min: can rotate in 2 direction, take the shorted way
                        cost += min(abs(n / c - i), abs(i - n / c)); // abs: absolute distance in hor and vert dir
                                                                    // n % c - j: hor manhatten dist
                                                                    // n / r - i: vert manhatten dist
                    }
                }
            }
            cout<<cost / (r * c)<<endl;
            return cost / (r * c);
        }
        void astar_transition(tuple<vector<vector<int>>, vector<pair<int, int>>, double> state,
                            priority_queue<tuple<vector<vector<int>>, vector<pair<int, int>>, double>,
                                        vector<tuple<vector<vector<int>>, vector<pair<int, int>>, double>>,
                                        decltype(&astar_compare)>& q,
                            unordered_map<vector<vector<int>>, double, StateVectorHash>& visited_cost, const vector<vector<int>>& goal) {

            vector<vector<int>> curr_config = get<0>(state);
            vector<pair<int, int>> curr_path = get<1>(state);
            int weight = get<2>(state);

            for (auto move : moves) {
                vector<vector<int>> new_config = action(curr_config, move.first, move.second);
                double new_cost = static_cast<double>(weight) + heuristic(new_config, goal) + 1.0;
                // Check if the state is new or reached with a lower cost
                if (!visited_cost.count(new_config) || visited_cost[new_config] > new_cost) {
                    visited_cost[new_config] = new_cost;
                    vector<pair<int, int>> new_path = curr_path;
                    new_path.push_back(move);
                    q.push(make_tuple(new_config, new_path, new_cost));
                } else {
                }
            }
        }

        vector<pair<int, int>> eliminate_loops(vector<pair<int, int>> path, int r, int c) {
            if (path.empty()) return {};

            int hor_limit = c / 2;  // Horizontal movement limit
            int ver_limit = r / 2;  // Vertical movement limit

            vector<pair<int, int>> optimized_path;
            int move_count = 0;
            int last_idx = path[0].first, last_dir = path[0].second;

            for (size_t i = 0; i < path.size(); i++) {
                int cur_idx = path[i].first, cur_dir = path[i].second;

                if (cur_dir == last_dir && cur_idx == last_idx) {
                    move_count++;
                } else {
                    if (move_count > 0) {
                        if ((last_dir < 2 && move_count > hor_limit) ||  // Left/Right threshold
                            (last_dir >= 2 && move_count > ver_limit)) { // Up/Down threshold
                            int opposite_dir = (last_dir % 2 == 0) ? last_dir + 1 : last_dir - 1;
                            optimized_path.emplace_back(last_idx, opposite_dir);
                        } else {
                            for (int j = 0; j < move_count; j++) {
                                optimized_path.emplace_back(last_idx, last_dir);
                            }
                        }
                    }
                    last_idx = cur_idx;
                    last_dir = cur_dir;
                    move_count = 1;
                }
            }
            if (move_count > 0) {
                if ((last_dir < 2 && move_count > hor_limit) || (last_dir >= 2 && move_count > ver_limit)) {
                    int opposite_dir = (last_dir % 2 == 0) ? last_dir + 1 : last_dir - 1;
                    optimized_path.emplace_back(last_idx, opposite_dir);
                } else {
                    for (int j = 0; j < move_count; j++) {
                        optimized_path.emplace_back(last_idx, last_dir);
                    }
                }
            }

            return optimized_path;
        }

        vector<pair<int, int>> search(vector<vector<int>> init_config, const vector<vector<int>>& goal) {
            if (is_goal(init_config, goal)) {
                return {};
            }
            if (c < 1 || r < 1) {
                return {};
            }
            queue<tuple<vector<vector<int>>, vector<pair<int,int>>, int>> q;
            set<vector<vector<int>>> visited;
            vector<pair<int, int>> result;
            q.push(make_tuple(init_config, result, 0));
            while (!q.empty()) {
                tuple<vector<vector<int>>, vector<pair<int, int>>, int> state = q.front();
                q.pop();
                vector<vector<int>> config = get<0>(state);
                if (is_goal(config, goal)) {
                    cout << get<2>(state);
                    return get<1>(state);
                    // return eliminate_loops(state.second, r, c);
                }
                transition(state, &q, &visited);
            }
            return {};
        } 

        vector<pair<int, int>> usc_search(vector<vector<int>> init_config, const vector<vector<int>>& goal) {
            if (is_goal(init_config, goal)) return {};
            if (c < 1 || r < 1) return {};

            priority_queue<tuple<vector<vector<int>>, vector<pair<int, int>>, int>,
                        vector<tuple<vector<vector<int>>, vector<pair<int, int>>, int>>,
                        decltype(&pq_compare)> q(pq_compare);

            unordered_map<vector<vector<int>>, int, StateVectorHash> visited_cost; // Track shortest cost to each state
            vector<pair<int, int>> result;

            q.push(make_tuple(init_config, result, 0));
            visited_cost[init_config] = 0; // Initial state has cost 0

            while (!q.empty()) {
                tuple<vector<vector<int>>, vector<pair<int, int>>, int> state = q.top();
                q.pop();
                vector<vector<int>> config = get<0>(state);
                vector<pair<int, int>> path = get<1>(state);
                int cost = get<2>(state);
                if (is_goal(config, goal)) {
                    return path;
                }
                usc_transition(make_tuple(config, path, cost), q, visited_cost);
            }
            return {};
        }


        vector<pair<int, int>> astar_search(vector<vector<int>> init_config, const vector<vector<int>>& goal) {
            if (is_goal(init_config, goal)) return {};
            if (c < 1 || r < 1) return {};

            priority_queue<tuple<vector<vector<int>>, vector<pair<int, int>>, double>,
                        vector<tuple<vector<vector<int>>, vector<pair<int, int>>, double>>,
                        decltype(&astar_compare)> q(astar_compare);

            unordered_map<vector<vector<int>>, double, StateVectorHash> visited_cost; // Track shortest cost to each state
            vector<pair<int, int>> result;

            q.push(make_tuple(init_config, result, 0));
            visited_cost[init_config] = 0; // Initial state has cost 0

            while (!q.empty()) {
                tuple<vector<vector<int>>, vector<pair<int, int>>, double> state = q.top();
                q.pop();
                vector<vector<int>> config = get<0>(state);
                vector<pair<int, int>> path = get<1>(state);
                double cost = get<2>(state);
                if (is_goal(config, goal)) {
                    return path;
                }
                astar_transition(make_tuple(config, path, cost), q, visited_cost, goal);
            }
            return {};
        }

};

void show_progress() {
    while (true) {
        cout << "." << flush; // Print dot without newline
        this_thread::sleep_for(chrono::seconds(1));
    }
}
int main() {
    vector<vector<int>> init_config, goal_config;
    vector<pair<int, int>> solution;
    
    // Test Cases
    vector<pair<vector<vector<int>>, vector<vector<int>>>> test_cases = {
        {{{0, 2, 1, 0}, {0, 2, 1, 1}, {2, 1, 0, 2}}, {{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}}},
        {{{1, 1, 1}, {2, 2, 2}, {3, 3, 3}}, {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}}},
        {{{1, 0}, {0, 1}}, {{0, 0}, {1, 1}}},
        {{{0, 1, 2, 3}, {1, 2, 3, 0}, {2, 3, 0, 1}, {3, 0, 1, 2}}, {{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 2, 2, 2}, {3, 3, 3, 3}}},
        {{{0}, {1}, {2}, {3}}, {{3}, {0}, {1}, {2}}},
        {{{1, 2, 3, 0}, {0, 1, 2, 3}}, {{0, 1, 2, 3}, {0, 1, 2, 3}}}
    };

    vector<string> test_names = {
        "3x4 Cube: ",
        "Already Solved 3x3 Cube",
        "2x2 Cube",
        "4x4 Cube", 
        "Single column",
        "2x4"
    };

    for (size_t i = 0; i < test_cases.size(); i++) {
        init_config = test_cases[i].first;
        goal_config = test_cases[i].second;
        Cube cube(init_config.size(), init_config[0].size());

        cout << "Running " << test_names[i] << "..." << endl;
        auto start_time = high_resolution_clock::now();
        thread progress_thread(show_progress);

        solution = cube.astar_search(init_config, goal_config);
        auto end_time = high_resolution_clock::now();
        duration<double> elapsed = end_time - start_time;
        progress_thread.detach();

        cout << "\nTest: " << test_names[i] << endl;
        for (auto move : solution) {
            cout << "Row/Col: " << move.first << ", Direction: " << move.second << endl;
        }
        cout << "Moves required: " << solution.size() << endl;
        cout << "Time elapsed: " << elapsed.count() << " seconds\n\n";
    }

    return 0;
}