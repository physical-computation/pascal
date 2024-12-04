import json
import numpy as np
from matplotlib import pyplot as plt

# Load the data
with open("results/results.json") as f:
	data = json.load(f)

# Plot the data
def plot(data: dict[str, float], output_dir: str) -> None:
	fig = plt.figure()

	plt.scatter(data["x"], data["y"], marker="x", color="red")
	plt.plot(data["x_new"], data["means"], color="black")
	plt.fill_between(data["x_new"], data["means"] - 3 * np.sqrt(data["variances"]), data["means"] + 3 * np.sqrt(data["variances"]), color='gray', alpha=.4, label='+/- 3 std')

	plt.xlabel("Input")
	plt.ylabel("Output")
	plt.title("Simple Gaussian Process Regression")

	plt.savefig(output_dir + "/plot.pdf", format="pdf")


if __name__ == '__main__':
	plot(data, "plots")