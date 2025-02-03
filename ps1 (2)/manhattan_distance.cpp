#include <bits/stdc++.h>
using namespace std;
int r, c;
double heuristic(const vector<vector<int>>& curr, const vector<vector<int>>& goal) {
    vector<int> misplaced_idx(r, -1); // to keep track of last misplaced element, so to not using it again
    double cost = 0;
    for (int i = 0; i < r; i++) {
        misplaced_idx[i] = -1;
    }

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
                                && (r * p + q > misplaced_idx[g]) // its index should be larger than the index of the last tile of the same type being used to replace misplaced tile
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
    return cost / (r * c);
}

void test_heuristic(const vector<vector<int>>& curr, const vector<vector<int>>& goal) {
    double result = heuristic(curr, goal);
    cout << "Heuristic cost: " << result << endl;
}
int main() {
    r = 3;
    c = 4;

    test_heuristic(
        {
            {0, 1, 2, 3},
            {1, 2, 3, 0},
            {2, 3, 0, 1},
            {3, 0, 1, 2}
        }, 
        {
            {0, 0, 0, 0},
            {1, 1, 1, 1},
            {2, 2, 2, 2},
            {3, 3, 3, 3}
        }
    );
    test_heuristic(
        {
            {0, 0, 2, 1},
            {1, 0, 2, 1},
            {2, 1, 0, 2}
        }, 
        {
            {0, 0, 0, 0},
            {1, 1, 1, 1},
            {2, 2, 2, 2}
        }
    );

        test_heuristic(
        {
            {0, 1, 0},
            {1, 0, 1},
            {2, 2, 2}
        }, 
        {
            {0, 0, 0},
            {1, 1, 1},
            {2, 2, 2}
        }
    );
    return 0;
}