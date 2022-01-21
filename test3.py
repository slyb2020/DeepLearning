import numpy as np
vectors = np.random.rand(10,1)-0.5
print(vectors)
vectors = np.array([1 if d > 0 else 0 for d in vectors]).reshape(-1,1)
print(vectors)