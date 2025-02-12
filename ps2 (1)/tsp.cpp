#include <bits/stdc++.h>

using namespace std;

class Graph {
    public:
        int node_size;
        vector<int> nodes;
        vector<tuple<int, int, double>> edges;
        vector<vector<pair<int, double>>> adj;
        // undirected
        void add_edge(int u, int v, double w) {
            adj[u].push_back({v, w});
            adj[v].push_back({u, w});
        }
        Graph(int size) {
            node_size = size;
        }
        static bool weight_compare(const tuple<int, int, double> &a, const tuple<int, int, double> &b) {
            return get<2>(a) > get<2>(b);
        }
        vector<tuple<int, int, double>> MST() {
            vector<tuple<int, int, double>> res;
            vector<bool> visited(node_size, false);
            priority_queue<tuple<int, int, double>, vector<tuple<int, int, double>>, decltype(&weight_compare)> pq;
            for (int i = 0; i < node_size; i++) {
                pq.push(edges[i]);
            }
            
        }
};


int main() {
    Graph g(5);
    g.add_edge(0, 1, 1.0);
    g.add_edge(0, 2, 2.0);
    g.add_edge(1, 2, 1.5);
    g.add_edge(1, 3, 2.5);
    g.add_edge(2, 3, 1.0);
    g.add_edge(3, 4, 3.0);

    vector<tuple<int, int, double>> mst = g.MST();

    cout << "Minimum Spanning Tree edges:\n";
    for (auto& e : mst) {
        cout << get<0>(e) << " - " << get<1>(e) << " : " << get<2>(e) << "\n";
    }
    return 0;

}