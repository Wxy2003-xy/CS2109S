#include<bits/stdc++.h>
using namespace std;

class Game {
    static const int r = 3, c = 3;
    array<array<int, c>, r> board;

    Game() : board{{{-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}}} {}
    int is_end(array<array<int, c>, r> board) {
        for (int i = 0; i < r; i++) {
            int ho_init = board[i][0];
            for (int j = 1; j < c; j++) {
                int curr = board[i][j];
                if (curr != ho_init) {
                    break;
                }
                if (j == c - 1) {
                    return ho_init;
                }
            }
        }
        for (int i = 0; i < c; i++) {
            int ve_init = board[0][i];
            for (int j = 1; j < r; j++) {
                int curr = board[j][i];
                if (curr != ve_init) {
                    break;
                }
                if (j == r - 1) {
                    return ve_init;
                }
            }
        }
        if (r != c) {
            return -1;
        }
        int down_slp = board[0][0];
        for (int i = 1; i < r; i++) {
            if (board[i][i] != down_slp) {
                break;
            }
            if (i == r - 1) {
                return down_slp;
            }
        }
        int up_slp = board[0][r-1];
        for (int i = 1; i < r; i++) {
            if (board[i][r - 1 - i]) {
                break;
            }
            if (i == r - 1) {
                return up_slp;
            }
        }
        return -1;
    }
    int remaining_tiles(array<array<int, c>, r> board) {
        int count = 0;
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                if (board[i][j] == -1) {
                    count += 1;
                }
            }
        }
        return count;
    }
    vector<pair<int, int>> available_moves(array<array<int, c>, r> board) {
        vector<pair<int, int>> moves;
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                if (board[i][j] == -1) {
                    moves.push_back(make_pair(i, j));
                }
            }
        }
        return moves;
    }
    array<array<int, c>, r> move(int player, pair<int, int> move, const array<array<int, c>, r>& board) {
        array<array<int, c>, r> result(board);
        result[move.first][move.second] = player;
        return result;
    }


};



