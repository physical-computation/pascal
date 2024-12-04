import numpy as np

import time

N = 1000
dim = 300

if __name__ == "__main__":

    completion_time = 0
    for i in range(N):
        a = np.random.uniform(-1, 1, [dim, dim])

        start_time = time.time()
        b = np.reciprocal(a)
        completion_time += time.time() - start_time
    completion_time /= N
    print(f"python\t{completion_time*1000000:.0f}\xB5s\treciprocal")
