import numpy as np

import time

N = 1000
dim = 100000

if __name__ == '__main__':
	a = np.random.uniform(-1, 1, [dim])
	b = np.random.uniform(-1, 1, [dim])

	completion_time = 0
	for i in range(N):
		start_time = time.time()
		c = np.dot(a, b)
		completion_time += time.time() - start_time

	completion_time /= N
	print(f"python\t{completion_time*1000000:.0f}\xB5s\tdot")
