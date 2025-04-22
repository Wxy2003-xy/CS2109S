import numpy as np 
from collections import defaultdict
bids = [
    [ 101, 500 ],
    [ 99, 1000 ],
    [ 98, 2200 ],
    [ 101, 1500 ],
    [ 100, 1500 ]
]
asks = [
    [ 102, 5000 ],
    [ 99, 1500 ],
    [ 100, 1000 ],
    [ 98.5, 2000 ],
    [ 99, 1500 ]
]
bids = np.asanyarray(bids)
asks = np.asanyarray(asks)

bids_dict= defaultdict(float)
for price, shares in bids:
    bids_dict[price] += shares
bids = np.array(list(bids_dict.items()))
bids = bids[bids[:, 0].argsort()]

asks_dict = defaultdict(float)
for price, shares in asks:
    asks_dict[price] += shares
asks = np.array(list(asks_dict.items()))
bids.T[[0,1]] = bids.T[[1,0]]

# no_b = bids.shape[0]
# no_a = asks.shape[0]
# bids.T[[0, 1]] = bids.T[[1, 0]]
zeros = np.zeros((bids.shape[0], 1))
bids = np.column_stack((bids, zeros)) 
zeros = np.zeros((asks.shape[0], 1))
asks = np.column_stack((zeros, asks))
# bids = bids[bids[:, 1].argsort()]
# asks = asks[asks[:, 1].argsort()]
print(bids, asks)

# i = 0
# j = 0
# arr = [0,0,0]
# count = 0
# while i < no_b and j < no_a:
#     if bids[i][1] < asks[j][1]:
#         arr = np.row_stack((arr, bids[i]))
#         i+=1
#         continue
#     if bids[i][1] >= asks[j][1]:
#         arr = np.row_stack((arr, asks[j]))
#         j+=1
#         continue
# arr = arr[1:]
# arr.T[0] = np.cumsum(arr.T[0])
# arr.T[2] = (np.cumsum(np.flip(arr.T[2])))
# print(arr)
