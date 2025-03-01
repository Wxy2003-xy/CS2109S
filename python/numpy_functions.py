import numpy as np

# Declare and initialize array/matrix
    # 1D
arr = np.array([1, 2, 3])
    # 2D
arr_2D = np.array([[1, 2], [3, 4]])
    # 3D
arr_3D = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])


# Pattern repetition: np.tile(<array>, <times to repeat>)
arr_rep = np.tile(arr, 3)
arr_2Drep = np.tile(arr_2D, 3)
arr_2Drep_2D = np.tile(arr_2D, (2, 2))

print(arr_rep) # -> [1 2 3 1 2 3 1 2 3]
print(arr_2Drep) # -> [[1 2 1 2 1 2]
                #      [3 4 3 4 3 4]]
print(arr_2Drep_2D) # -> [[1 2 1 2]
                    #     [3 4 3 4]
                    #     [1 2 1 2]
                    #     [3 4 3 4]]
# Element-wise repeat: np.repeat(<array>, <time to repeat>, axis=<0|1|...>)
arr_ele_rep = np.repeat(arr, 3)
print(arr_ele_rep)  # -> [1 1 1 2 2 2 3 3 3]


# cumulative operations: np.cumprod(<array>, axis)
arr_cumprod = np.cumprod(arr)
print(arr_cumprod)  # -> [1 2 6]
arr_2D_cumprod = np.cumprod(arr_2D, axis=0)
print(arr_2D_cumprod) # -> [[1 2]
                        #   [3 8]]
arr_2D_cumprod1 = np.cumprod(arr_2D, axis=1)
print(arr_2D_cumprod1) # -> [[ 1  2]
                          #  [ 3 12]]


arr_X = np.array([[1], [2], [3]])
poly = np.cumprod(np.tile(arr_X, 3), axis=1)
print(poly)