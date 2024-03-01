import numpy as np

import queue

# a = queue.Queue(maxsize=5)

# a.put(np.full((1,2), 6))

# a.put(np.full((1,2), 5))

# a.put(np.full((1,2), 4))


# elements_in_queque = []
# while not a.empty():
#     element = a.get()
#     elements_in_queque.append(element)
# for element in elements_in_queque:
#     a.put(element)

# x = np.vstack(elements_in_queque[::-1])



# #maxsize-现在有的states拼接成 feature_dims * maxsize 维度的二维矩阵
# for lack_num in range(8 - x.shape[0]):
#     x = np.vstack((x, np.array([0, 0])))

# print(x)

lack_id_col = np.full((8,1), 999)
lack_data_matrix = np.full((8, 6), 0)
lack_timestamp = np.full((8,1), -1)

result_history_matrix = np.concatenate((lack_id_col, lack_data_matrix, lack_timestamp), axis=1)

print(result_history_matrix)
print(result_history_matrix[0][0])
print(result_history_matrix.shape)

