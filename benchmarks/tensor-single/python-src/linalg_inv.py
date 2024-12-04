import numpy as np

import time

N = 1000
dim = 50

if __name__ == '__main__':
	a = np.random.uniform(-1, 1, [dim, dim])

	completion_time = 0
	for i in range(N):
		start_time = time.time()
		b = np.linalg.inv(a)
		completion_time += time.time() - start_time

	completion_time /= N
	print(f"python\t{completion_time*1000000:.0f}\xB5s\tlinalg_inv")
