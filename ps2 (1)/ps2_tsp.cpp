#include <bits/stdc++.h>
using namespace std;

vector<vector<int>> transition(const vector<int>& route) {
    vector<vector<int>> permutations; 
    for (size_t i = 0; i < route.size(); i++) {
        for (size_t j = i + 1; j < route.size();j++) {
            if (i == j) {
                continue;
            }
            vector<int> swap(route);
            int tmp = swap[i];
            swap[i] = swap[j];
            swap[j] = tmp;
            permutations.push_back(swap);
        }
    }
    return permutations;
}

static bool pq_compare(tuple<int, int, double>& a, tuple<int, int, double>& b) {
    return get<2>(a) <= get<2>(b);
}

double upper_bound(int cities, const vector<tuple<int, int, double>>& distances) {
    priority_queue<tuple<int, int, double>, vector<tuple<int, int, double>>, 
            decltype(&pq_compare)> pq(&pq_compare);
    for (int i = 0; i < distances.size(); i++) {
        pq.push(distances[i]);
    }
    double res = 0;
    for (int i = 0; i < cities; i++) {
        tuple<int, int, double> e = pq.top();
        pq.pop();
        res += get<2>(e);
    }
    return res;
}

double eval(int cities, const vector<tuple<int, int, double>>& distances, const vector<int>& route) {
    double score = upper_bound(cities, distances);
    for (int i = 0; i < route.size(); i++) {
        for (int j = 0; j < distances.size(); j++) {
            if (i == route.size() - 1) {
                if ((route[i] == get<0>(distances[j]) && route[0] == get<1>(distances[j]))  
                        || (route[i] == get<1>(distances[j]) && route[0] == get<0>(distances[j]))) {
                    score -= get<2>(distances[j]);
                    break;
                }
            } else {
                if ((route[i] == get<0>(distances[j]) && route[i + 1] == get<1>(distances[j]))  
                        || (route[i] == get<1>(distances[j]) && route[i + 1] == get<0>(distances[j]))) {
                    score -= get<2>(distances[j]);
                    break;
                }
            }
        }
    }
    return score;
}

vector<int> random_init(int cities) {
    vector<int> route(cities);
    iota(route.begin(), route.end(), 0);
    mt19937 rng(random_device{}());
    shuffle(route.begin(), route.end(), rng);
    return route;
}

vector<int> hill_climb(int cities, 
                       const vector<tuple<int, int, double>>& distances, 
                       vector<vector<int>> (*transition)(const vector<int>&),
                       double (*eval)(int, const vector<tuple<int, int, double>>&, const vector<int>&)) {
    vector<int> curr = random_init(cities);
    double curr_eval = eval(cities, distances, curr);
    while (true) {
        vector<vector<int>> permutations = transition(curr);
        vector<int> curr_best = permutations[0];
        double curr_best_eval = eval(cities, distances, curr_best);
        for (int i = 1; i < permutations.size(); i++) {
            double new_eval = eval(cities, distances, permutations[i]);
            if (curr_best_eval < new_eval) {
                curr_best = permutations[i];
                curr_best_eval = new_eval;
            }
        }
        if (curr_best_eval <= curr_eval) {
            return curr;
        }
        curr = curr_best;
        curr_eval = curr_best_eval;
    }
}

vector<int> stochastic_hill_climb(int repeats, int cities, 
                       const vector<tuple<int, int, double>>& distances, 
                       vector<vector<int>> (*transition)(const vector<int>&),
                       double (*eval)(int, const vector<tuple<int, int, double>>&, const vector<int>&)) {
    vector<int> init = hill_climb(cities, distances, transition, eval);
    double score = eval(cities, distances, init);
    for (int i = 0; i < repeats; i++) {
        vector<int> next = hill_climb(cities, distances, transition, eval);
        double new_score = eval(cities, distances, next);
        if (new_score > score) {
            init = next;
            score = new_score;
        }
    } 
    return init;
}
int main() {
    int cities = 5; // Number of cities
    vector<tuple<int, int, double>> distances = {
        {0, 1, 10.0}, {0, 2, 20.0}, {0, 3, 25.0}, {0, 4, 30.0},
        {1, 2, 15.0}, {1, 3, 35.0}, {1, 4, 40.0},
        {2, 3, 50.0}, {2, 4, 45.0},
        {3, 4, 55.0}
    };
    // 🔹 Run Hill Climbing
    vector<int> best_route = stochastic_hill_climb(1000000, cities, distances, transition, eval);
    double score = eval(cities, distances, best_route);
    // 🔹 Print the best found route
    cout << "Best Route Found: ";
    for (int city : best_route) {
        cout << city << " ";
    }
    cout << " (Back to start), score: "<< score << endl;

    return 0;
}
