#include <bits/stdc++.h>

using namespace std;

double mean_squared_error(vector<double> y_true, vector<double> y_pred) {
    double res = 0.0;
    for (int i = 0; i < y_true.size(); i++) {
        double sq_err = pow(y_true[i] - y_pred[i], 2);
    }
    return res;
}

double mean_absolute_error(vector<double> y_true, vector<double> y_pred) {
    double res = 0.0;
    for (int i = 0; i < y_true.size(); i++) {
        double abs_err = abs(y_true[i] - y_pred[i]);
    }
    return res;
}

vector<vector<double>> add_bias_col(vector<vector<double>> mat) {
    int r = mat.size();
    int c = mat[0].size();
    vector<vector<double>> res(r, vector<double>(1, 1));
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            res[r].push_back(mat[i][j]);
        }
    }
    return res;
}
int main() {

}