import tensorflow as tf
import numpy as np

# create a dataset of 10 numbers
dataset = tf.data.Dataset.range(10)
for val in dataset:
    print(val.numpy())
'''
0
1
2
3
4
5
6
7
8
9
'''
print()

# split into 10 sliding windows with at max 5 elements
dataset = dataset.window(5, shift=1)
for window_dataset in dataset:
    for val in window_dataset:
        print(val.numpy(), end=" ")
    print()
'''
0 1 2 3 4
1 2 3 4 5
2 3 4 5 6
3 4 5 6 7
4 5 6 7 8
5 6 7 8 9
6 7 8 9
7 8 9
8 9
9
'''
print()

# only choose the windows that have exactly 5 elements
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
for window_dataset in dataset:
    for val in window_dataset:
        print(val.numpy(), end=" ")
    print()
'''
0 1 2 3 4 
1 2 3 4 5 
2 3 4 5 6 
3 4 5 6 7 
4 5 6 7 8 
5 6 7 8 9 
'''
print()

# create a list of each of these
dataset = dataset.flat_map(lambda window: window.batch(5))
for window in dataset:
    print(window.numpy())
'''
[0 1 2 3 4]
[1 2 3 4 5]
[2 3 4 5 6]
[3 4 5 6 7]
[4 5 6 7 8]
[5 6 7 8 9]
'''
print()

# create the features and labels
# 4 items are features, the last item is the label
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
for x, y in dataset:
    print(x.numpy(), y.numpy())
'''
[0 1 2 3] [4]
[1 2 3 4] [5]
[2 3 4 5] [6]
[3 4 5 6] [7]
[4 5 6 7] [8]
[5 6 7 8] [9]
'''
print()

# shuffle the data
dataset = dataset.shuffle(buffer_size=10)
for x, y in dataset:
    print(x.numpy(), y.numpy())
'''
[5 6 7 8] [9]
[4 5 6 7] [8]
[0 1 2 3] [4]
[2 3 4 5] [6]
[3 4 5 6] [7]
[1 2 3 4] [5]
'''
print()

# group into batches of 2
dataset = dataset.batch(2).prefetch(1)
for x, y in dataset:
  print("x = ", x.numpy())
  print("y = ", y.numpy())
'''
x =  [[0 1 2 3] [3 4 5 6]]
y =  [[4] [7]]

x =  [[4 5 6 7] [5 6 7 8]]
y =  [[8] [9]]

x =  [[1 2 3 4] [2 3 4 5]]
y =  [[5] [6]]
'''