import torch

x = torch.tensor([-1.85830188, 6.49783171, -5.28644911, -8.64860012, 7.26167869, -6.80747631, 3.05319221, 2.13969528, 6.08561453, -7.7652272, 5.04327755, -7.63837908, -6.58659673, 1.69170154, -3.86355285, -2.94409609, -5.44211817, -7.32981358, -9.10707192, 8.81635902, 5.04970648, 2.49813017, 5.37001898, -8.07558233, -3.13279303, -9.19288847, -9.32337246, 7.48993695, 5.70166202, -9.41504528], dtype=torch.double).reshape([10, 3])

y = torch.tensor([1.04437514, 0.97880481, 0.06629005, 1.05698838, 0.71818224, 0.01391952, -0.04546158, 1.09317289, 0.0722612, 0.96746236], dtype=torch.double).reshape([10, 1])

def sigmoid(x):
	return 1 / (1 + torch.exp(-x))

def forward(x, w0, w1):
	l1 = sigmoid(torch.matmul(x, w0))
	l2 = sigmoid(torch.matmul(l1, w1))

	return l2

def loss(y, y_pred):
	return torch.mean((y - y_pred) ** 2)

if __name__ == '__main__':
	n = 20
	learning_rate = 2.0

	w0 = torch.tensor([3.285286, -0.723522, 0.217484, -1.068676, 0.806425, -1.374203, 0.594250, -2.378225, 0.891999, 1.264324, -0.511385, 0.932220, -0.569741, -1.240200, -0.399840, 0.033192, 1.563455, -0.369499, 0.038533, -0.088953, 1.662209, 0.016993, 0.830817, 0.717879, -1.485082, -0.722157, -0.264169, 0.716676, 0.960650, 0.405095], dtype=torch.double, requires_grad=True).reshape([3, 10])
	w0.retain_grad()

	w1 = torch.tensor([-0.447488, -0.483253, -0.163882, 1.369771, 0.293023, -0.878082, 1.372940, -1.099247, 0.331679, 0.093458], dtype=torch.double, requires_grad=True).reshape([10, 1])
	w1.retain_grad()

	torch.Tensor()

	print_string = ""
	for i in range(n):
		y_pred = forward(x, w0, w1)
		loss_val = loss(y, y_pred)
		print_string += loss_val.data.item().__str__() + ", "

		loss_val.backward()

		w0.data = w0.data - learning_rate * w0.grad.data
		w1.data = w1.data - learning_rate * w1.grad.data

		w0.grad.data.zero_()
		w1.grad.data.zero_()

	print(print_string[:-2])

