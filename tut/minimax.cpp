#include <bits/stdc++.h>

using namespace std;
using namespace std::chrono;
// Generic Node Structure
template <typename T>
struct Node {
    T value;
    vector<Node<T>*> children;

    Node(T val) : value(val) {}
};

// Generic Minimax Function (With Debug Prints)
template <typename T>
T minimax(Node<T>* node, bool isMax, int depth = 0) {
    // Print current node evaluation
                                                            cout << string(depth * 2, ' ') << "Evaluating node with value: " << node->value << endl;
    if (node->children.empty()) {
                                                            cout << string(depth * 2, ' ') << "Leaf node reached, returning: " << node->value << endl;
        return node->value;
    }
    if (isMax) {
        T best = numeric_limits<T>::min();
                                                            cout << string(depth * 2, ' ') << "Maximizing..." << endl;
        for (Node<T>* child : node->children) {
            T childValue = minimax(child, false, depth + 1);
            best = max(best, childValue);
                                                            cout << string(depth * 2, ' ') << "Updated best value (Max): " << best << endl;
        }
        return best;
    } else {
        T best = numeric_limits<T>::max();
                                                            cout << string(depth * 2, ' ') << "Minimizing..." << endl;
        for (Node<T>* child : node->children) {
            T childValue = minimax(child, true, depth + 1);
            best = min(best, childValue);
                                                            cout << string(depth * 2, ' ') << "Updated best value (Min): " << best << endl;
        }
        return best;
    }
}

// Generic Minimax with Alpha-Beta Pruning (With Debug Prints)
template<typename T>
T minimax_alpha_beta(Node<T>* node, T alpha, T beta, bool isMax, int depth = 0) {
                                                            cout << string(depth * 2, ' ') << "Evaluating node with value: " << node->value << " (α=" << alpha << ", β=" << beta << ")" << endl;

    if (node->children.empty()) {
                                                            cout << string(depth * 2, ' ') << "Leaf node reached, returning: " << node->value << endl;
        return node->value;
    }

    if (isMax) {
        T best = numeric_limits<T>::min();
                                                            cout << string(depth * 2, ' ') << "Maximizing..." << endl;
        for (Node<T>* child : node->children) {
            T childValue = minimax_alpha_beta(child, alpha, beta, false, depth + 1);
            best = max(best, childValue);
            alpha = max(alpha, best);
                                                            cout << string(depth * 2, ' ') << "Updated best (Max): " << best << " (α=" << alpha << ", β=" << beta << ")" << endl;
            if (beta <= alpha) {
                                                            cout << string(depth * 2, ' ') << "Pruning branch (α=" << alpha << ", β=" << beta << ")" << endl;
                break;  // Pruning
            }
        }
        return best;
    } else {
        T best = numeric_limits<T>::max();
                                                            cout << string(depth * 2, ' ') << "Minimizing..." << endl;
        for (Node<T>* child : node->children) {
            T childValue = minimax_alpha_beta(child, alpha, beta, true, depth + 1);
            best = min(best, childValue);
            beta = min(beta, best);
                                                            cout << string(depth * 2, ' ') << "Updated best (Min): " << best << " (α=" << alpha << ", β=" << beta << ")" << endl;
            if (beta <= alpha) {
                                                            cout << string(depth * 2, ' ') << "Pruning branch (α=" << alpha << ", β=" << beta << ")" << endl;
                break;  // Pruning
            }
        }
        return best;
    }
}

// Demo: Create a Generic Minimax Tree
int main() {
    // sample tree
    Node<int>* root = new Node<int>(0);
    root->children = {
        new Node<int>(0),
        new Node<int>(0),
        new Node<int>(0)
    };

    root->children[0]->children = { new Node<int>(3), new Node<int>(5), new Node<int>(2) };
    root->children[1]->children = { new Node<int>(7), new Node<int>(9), new Node<int>(8) };
    root->children[2]->children = { new Node<int>(4), new Node<int>(6), new Node<int>(1) };
    // root of minimax tree B
    Node<int>* root_1 = new Node<int>(0);
    root_1->children = {new Node<int>(0), new Node<int>(0)};
        root_1->children[0]->children = {new Node<int>(10), new Node<int>(0), new Node<int>(0)};

        root_1->children[1]->children = {new Node<int>(0), new Node<int>(0)};
            root_1->children[0]->children[1]->children = {new Node<int>(12), new Node<int>(0)};
            root_1->children[0]->children[2]->children = {new Node<int>(9), new Node<int>(7), new Node<int>(3)};

            root_1->children[1]->children[0]->children = {new Node<int>(0), new Node<int>(3), new Node<int>(8)};
            root_1->children[1]->children[1]->children = {new Node<int>(4), new Node<int>(2)};
    
    // node of minimax tree D
    Node<int>* root_2 = new Node<int>(0);

    root_2->children = {new Node<int>(0)};


                                                            cout << "\n==== Minimax Execution ====" << endl;
    auto start1 = high_resolution_clock::now();
    int result1 = minimax(root_1, true);
    auto stop1 = high_resolution_clock::now();
    auto duration1 = duration_cast<microseconds>(stop1 - start1);
                                                            cout << "Final Minimax result: " << result1 << endl;
                                                            cout << "Time taken without Alpha-Beta pruning: " << duration1.count() << " microseconds" << endl;

                                                            cout << "\n==== Minimax with Alpha-Beta Pruning Execution ====" << endl;
    auto start2 = high_resolution_clock::now();
    int result2 = minimax_alpha_beta(root_1, numeric_limits<int>::min(), numeric_limits<int>::max(), true);
    auto stop2 = high_resolution_clock::now();
    auto duration2 = duration_cast<microseconds>(stop2 - start2);
                                                            cout << "Final Minimax with Alpha-Beta result: " << result2 << endl;
                                                            cout << "Time taken with Alpha-Beta pruning: " << duration2.count() << " microseconds" << endl;

    // Cleanup memory
    for (Node<int>* child : root_1->children) {
        for (Node<int>* grandchild : child->children) {
            delete grandchild;
        }
        delete child;
    }
    delete root;

    return 0;
}