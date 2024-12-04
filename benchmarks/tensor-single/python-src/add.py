import numpy as np

import time

N = 1000
dim = 3
ndim = 10

if __name__ == '__main__':
	shape = [dim for i in range(ndim)]

	completion_time = 0
	for i in range(N):
		a = np.random.uniform(-1, 1, shape)
		b = np.random.uniform(-1, 1, shape)

		start_time = time.time()
		c = a + b
		completion_time += time.time() - start_time

	completion_time /= N
	print(f"python\t{completion_time*1000000:.0f}\xB5s\tadd")
