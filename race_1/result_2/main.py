import numpy as np

x = np.linspace(0, 2, 1000)

print(np.frompyfunc(lambda i: i + 1, 1, 1)(x))
