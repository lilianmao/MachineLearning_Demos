import numpy as np

list1 = [3, 5, -4, -1, 0, -2, -6]
list2 = sorted(list1, key=lambda x: abs(x))
print list2

list3 = np.minimum(list1, 2)
print list3