import numpy as np

SAMPLES_LOCATION = "samples"

SAMPLE_FILES = [
	"samples/pascal.csv",
	"samples/pytorch.csv"
]

if __name__ == '__main__':
	pascal_samples = np.genfromtxt(SAMPLE_FILES[0], delimiter=',')
	python_samples = np.genfromtxt(SAMPLE_FILES[1], delimiter=',')

	print(f"pascal_mu: \t{np.mean(pascal_samples)},\t\t pascal_sigma: \t{np.std(pascal_samples)}")
	print(f"python_mu: \t{np.mean(python_samples)},\t\t python_sigma: \t{np.std(python_samples)}")