#include <bits/stdc++.h>

using namespace std;

class Cube {
    public:
        int r;
        int c;
        vector<vector<int>> config;
        vector<pair<int, int>> moves;

        Cube(int r, int c, vector<vector<int>> config) {
            this->r = r;
            this->c = c;
            this->config = config;
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
                int tmp = new_config[idx][c-1];
                for (int i = c-2; i >= 0; i--) {
                    int t = tmp;
                    tmp = new_config[idx][i];
                    new_config[idx][i] = t;
                }
                new_config[idx][c-1] = tmp;
                return new_config;
            }
            if (dir == 1) { // right
                int tmp = new_config[idx][0];
                for (int i = 1; i < this->c; i++) {
                    int t = tmp;
                    tmp = new_config[idx][i];
                    new_config[idx][i] = t;
                }
                new_config[idx][0] = tmp;
                return new_config;
            }
            if (dir == 2) { // up
                int tmp = new_config[r - 1][idx];
                for (int i = r - 2; i >= 0; i--) {
                    int t = tmp;
                    tmp = new_config[i][idx];
                    new_config[i][idx] = t;
                }
                new_config[r-1][idx] = tmp; 
                return new_config;
            }
            if (dir == 3) { // down
                int tmp = new_config[0][idx];
                for (int i = 0; i < r-1; i++) {
                    int t = tmp;
                    tmp = new_config[i][idx];
                    new_config[i][idx] = t;
                }
                new_config[0][idx] = tmp; 
                return new_config;
            }
        }

        bool is_goal(vector<vector<int>> config) {
            for (int i = 0; i < r; i++) {
                int init = config[i][0];
                for (int j = 1; j < c; j++) {
                    if (config[i][j] != init) {
                        return false;
                    }
                }
            }
            return true;
        }
        void transition(pair<vector<vector<int>>, vector<pair<int, int>>> state,
                int idx, int dir, queue<pair<vector<vector<int>>, 
                vector<pair<int, int>>>>* q, set<vector<vector<int>>>* visited) {
            vector<vector<int>> curr_config = state.first;
            vector<pair<int, int>> curr_path = state.second;
            for (auto move : moves) {
                vector<vector<int>> new_config = action(curr_config, idx, dir);
                if (visited->find(new_config) == visited->end()) {
                    visited->insert(new_config);
                    vector<pair<int, int>> new_path(curr_path);
                    new_path.push_back(move);
                    q->push(make_pair(new_config, new_path));
                }
            }
        }
};

int main() {}