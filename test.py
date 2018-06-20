a = [1,2,3,4,5]
b = [12,14,14,15,15]

# for x,y in zip(enumerate(a),enumerate(b)):
#     print(type(x))
#     print(y)
#     print(x,y)


import numpy as np
x=np.random.random(10)
y=np.random.random(10)
z=np.random.random(10)

# print(x,y)

#方法一：根据公式求解
d1=np.sqrt(np.sum(np.square(x-y)))

#方法二：根据scipy库求解
from scipy.spatial.distance import pdist,squareform
X=np.vstack([x,y])
X=np.vstack([X,z])

# print(X)
x = np.array([[0, 1], [1, 0], [2, 0]])
d2=pdist(X,'euclidean')
z = squareform(d2)


# print(np.mean(z,axis=1)[-1])


print( {
    'epoch': 1,
    'average_loss': 1,
    'correct': 2,
    'total': 2,
    'accuracy': 100. * 1 / 1
})