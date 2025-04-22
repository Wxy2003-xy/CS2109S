import numpy as np
def compute_Info_boolean(pos, neg):
    pos_prob = np.divide(pos, (pos + neg), dtype=np.float128)
    neg_prob = np.divide(neg, (pos + neg), dtype=np.float128)
    return - pos_prob * np.log2(pos_prob) - neg_prob * np.log2(neg_prob)

def entropy(pos, neg):
    total = pos + neg
    if total == 0: 
        return 0
    pos_prob = np.divide(pos, total, dtype=np.float128)
    neg_prob = np.divide(neg, total, dtype=np.float128)
    entropy = 0
    if pos_prob > 0:
        entropy -= pos_prob * np.log2(pos_prob)
    if neg_prob > 0:
        entropy -= neg_prob * np.log2(neg_prob)

    return entropy

def compute_IG_boolean(parent_pos, parent_neg, left_pos, left_neg, right_pos, right_neg):
    parent_entropy = entropy(parent_pos, parent_neg)
    total_parent = parent_pos + parent_neg
    total_left = left_pos + left_neg
    total_right = right_pos + right_neg
    left_weight = np.divide(total_left, total_parent, dtype=np.float128) if total_left > 0 else 0
    right_weight = np.divide(total_right, total_parent, dtype=np.float128) if total_right > 0 else 0
    weighted_child_entropy = left_weight * entropy(left_pos, left_neg) + right_weight * entropy(right_pos, right_neg)
    return parent_entropy - weighted_child_entropy


# x0 = np.matrix([[1], [1], [1], [1]])
# x1 = np.matrix([[6], [8], [12], [2]])
# x2 = np.matrix([[4], [5], [9], [1]])
# x3 = np.matrix([[11], [9], [25], [3]])
# y = np.matrix([[20], [1], [3], [7]])
# X = np.hstack([x0, x1, x2, x3])
# Xt = np.transpose(X)
# w = np.linalg.inv(Xt @ X) @ Xt @ y
# print(w)

