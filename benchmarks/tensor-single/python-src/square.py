import numpy as np

import time

N = 10000
dim = 200

if __name__ == '__main__':

	completion_time = 0
	for i in range(N):
		a = np.random.uniform(-1, 1, [dim, dim])

		start_time = time.time()
		b = np.square(a)
		completion_time += time.time() - start_time

	completion_time /= N
	print(f"python\t{completion_time*1000000:.0f}\xB5s\tsquare")
