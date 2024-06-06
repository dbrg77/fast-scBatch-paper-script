import time
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from tqdm import trange

"""
Check all parameters passed to the function.
"""
def _check_params(
	X, D, batch, k, c, p, EPOCHS, lr, corr_method, cluster_method, device
) -> None:
	if not isinstance(X, pd.DataFrame):
		raise TypeError("X must be a pandas DataFrame.")
	if not isinstance(D, pd.DataFrame):
		raise TypeError("D must be a pandas DataFrame.")
	if not isinstance(batch, pd.DataFrame):
		raise TypeError("batch must be a pandas DataFrame.")
	if D.shape != (X.shape[1], X.shape[1]):
		raise ValueError(f"D must be a square matrix with shape ({X.shape[1]}, {X.shape[1]}).")
	if batch.shape != (X.shape[1], 1):
		raise ValueError(f"batch must be a column vector with shape ({X.shape[1]}, 1).")
	if not isinstance(k, int) or k < 1:
		raise TypeError("k must be a positive integer.")
	if not isinstance(c, int) or c < 1:
		raise TypeError("c must be a positive integer.")
	if not isinstance(p, float) or p < 0 or p > 1:
		raise TypeError("p must be a float between 0 and 1.")
	if not isinstance(EPOCHS, tuple) or len(EPOCHS) != 3:
		raise TypeError("EPOCHS must be a tuple of 3 integers.")
	if not all(isinstance(i, int) and i >= 0 for i in EPOCHS):
		raise ValueError("EPOCHS must be a tuple of 3 non-negative integers.")
	if not isinstance(lr, tuple) or len(lr) != 3:
		raise TypeError("lr must be a tuple of 3 floats.")
	if not all(isinstance(i, float) and i > 0 for i in lr):
		raise ValueError("lr must be a tuple of 3 positive floats.")
	if not isinstance(corr_method, str):
		raise TypeError("corr_method must be a string.")
	if corr_method not in ("pearson"):
		raise NotImplementedError("corr_method currently supports: \'pearson\'.")
	if not isinstance(cluster_method, str):
		raise TypeError("cluster_method must be a string.")
	if cluster_method not in ("spectral"):
		raise NotImplementedError("cluster_method currently supports: \'spectral\'.")
	if not isinstance(device, str):
		raise TypeError("device must be a string.")

"""
The network structure.
"""
# @torch.compile
class Network(nn.Module):
	def __init__(self, _alpha, _beta, m, n, k, device) -> None:
		super(Network, self).__init__()
		self.m, self.n, self.k = m, n, k
		self.device = device
		self.alpha = nn.Parameter(_alpha)
		self.beta = nn.Parameter(_beta)
	def forward(self, A, X, mode) -> torch.Tensor:
		if mode == 'all': # Stage 3: normal update
			return torch.matmul(A, self.alpha) + torch.mul(self.beta, X)
		elif type(mode) == int: # Stage 2: randomly reduce dimension
			keys = torch.tensor(random.sample(range(self.m), mode), device=self.device)
			A_ = torch.index_select(A, dim=0, index=keys)
			X_ = torch.index_select(X, dim=0, index=keys)
			return torch.matmul(A_, self.alpha) + torch.mul(self.beta, X_)
		else: # Stage 1: choose one in each cluster to update
			X_ = torch.index_select(X, dim=1, index=mode)
			alpha_ = torch.index_select(self.alpha, dim=1, index=mode)
			beta_ = torch.index_select(self.beta, dim=1, index=mode)
			return torch.matmul(A, alpha_) + torch.mul(beta_, X_)

"""
Loss function.
@param Y: the correlation matrix given by the model.
@param std: the justified correlation matrix.
@return: the loss value.
"""
# @torch.compile
def loss(Y, std, method = "pearson") -> torch.Tensor:
	if method == "pearson":
		corr = torch.corrcoef(Y.T)
		return torch.norm(corr - std, p='fro')
	else:
		raise NotImplementedError("loss function currently supports: \'pearson\'.")

def solver(
	X: pd.DataFrame,
	D: pd.DataFrame,
	batch: pd.DataFrame,
	k: int = 10,
	c: int = 10,
	p: float = 0.15,
	EPOCHS = (30, 90, 80),
	lr = (0.002, 0.004, 0.008),
	corr_method: str = "pearson",
	cluster_method: str = "spectral",
	device: str = "cuda" if torch.cuda.is_available() else "cpu",
	random_state: int = 42,
	verbose: bool = True
) -> pd.DataFrame:
# Initialize
	if verbose:
		print(f"Initalizing on \'{device}\'...", end="\t")
	try:
		_check_params(X, D, batch, k, c, p, EPOCHS, lr, corr_method, cluster_method, device)
	except Exception as e:
		raise e
	torch.manual_seed(random_state)
	np.random.seed(random_state)
	random.seed(random_state)
	if verbose:
		print("Done.")

# Preprocessing
	if verbose:
		print(f"Preprocessing data of shape {X.shape} and performing PCA...", end="\t")
	m, n = X.shape
	initial_columns = X.columns
	batch.columns = ["Batch"]
	X = X.T.join(batch).sort_values(by=["Batch"]).drop(columns=["Batch"]).T
	indices_name, columns_name = X.index, X.columns
	D = D.loc[columns_name, columns_name]
	X = torch.tensor(X.values, dtype=torch.float32, device=device)
	D = torch.tensor(D.values, dtype=torch.float32, device=device)
	batch = batch.groupby("Batch").size().values.tolist()
	batch = [i for i in batch if i != 0]
	X_split = torch.split(X, batch, dim=1)
	U, V = [], []
	for i in range(len(batch)):
		u, s, v = map(
			lambda x: torch.tensor(x, device=device),
			PCA(n_components=k)._fit(X_split[i].cpu().numpy())
		)
		U.append(torch.matmul(u, torch.diag(s)))
		v = torch.cat([
			torch.zeros(i*k, batch[i], device=device),
			v.to(device),
			torch.zeros((len(batch)-i-1)*k, batch[i], device=device)
		], dim=0)
		V.append(v)
	A = torch.cat(U, dim=1)
	B = torch.cat(V, dim=1)
	model = Network(torch.zeros_like(B), torch.ones(1, n), m, n, k, device).to(device)
	optimizer1 = torch.optim.SGD(model.parameters(), lr=lr[0])
	optimizer2 = torch.optim.Adam(model.parameters(), lr=lr[1])
	optimizer3 = torch.optim.Adam(model.parameters(), lr=lr[2])
	if verbose:
		print("Done.")

# Training
	if verbose:
		print(f"Training...")
	clock = time.time()
# Stage 1: choose one in each cluster to update
	# Here only spectral clustering is implemented.
	if verbose:
		print(f"Stage 1: choose one in each cluster to update...")
	label = [
		SpectralClustering(n_clusters=min(c, batch[i]), random_state=random_state).\
			fit_predict(V[i].T.cpu().numpy()) + c * i
		for i in range(len(batch))
	]
	label = torch.tensor(np.concatenate(label), dtype=torch.int32, device=device)
	cluster = [[] for i in range(len(label.unique()))]
	for i in range(n):
		cluster[label[i]].append(i)

	losses = []
	for epoch in (trange(EPOCHS[0]) if verbose else range(EPOCHS[0])):
		optimizer1.zero_grad()
		keys = torch.tensor([random.choice(cluster[i]) for i in range(len(cluster))], device=device)
		D_ = torch.index_select(torch.index_select(D, dim=0, index=keys), dim=1, index=keys)
		Y = model(A, X, keys)
		loss_val = loss(Y, D_, corr_method)
		# losses.append(loss_val.item())
		loss_val.backward()
		# You may check the loss value on the whole dataset if preferred.
		"""
		model.eval()
		with torch.no_grad():
			Y = model(A, X, 'all')
			loss_val = loss(Y, D)
			losses.append(loss_val.item())
		model.train()
		"""
		model.alpha.grad = torch.index_select(
			model.alpha.grad, dim=1,
			index=torch.index_select(keys, dim=0, index=label)
		) + model.alpha.grad * min(lr[2]/lr[0] - 1, 5)
		model.beta.grad = torch.index_select(
			model.beta.grad, dim=1,
			index=torch.index_select(keys, dim=0, index=label)
		) + model.beta.grad * min(lr[2]/lr[0] - 1, 5)
		optimizer1.step()

# Stage 2: randomly reduce dimension
	if verbose:
		print(f"Stage 2: randomly reduce dimension...")
	for epoch in (trange(EPOCHS[1]) if verbose else range(EPOCHS[1])):
		optimizer2.zero_grad()
		Y = model(A, X, int(m*p))
		loss_val = loss(Y, D, corr_method)
		losses.append(loss_val.item())
		loss_val.backward()
		optimizer2.step()

# Stage 3: normal update
	if verbose:
		print(f"Stage 3: normal update...")
	for epoch in (trange(EPOCHS[2]) if verbose else range(EPOCHS[2])):
		optimizer3.zero_grad()
		Y = model(A, X, 'all')
		loss_val = loss(Y, D, corr_method)
		losses.append(loss_val.item())
		loss_val.backward()
		optimizer3.step()
# End of training
	if verbose:
		plt.plot(np.arange(len(losses)), losses, label="loss")
		plt.legend()
		plt.show()
		print(f"Training finished in {time.time()-clock:.2f} seconds.")

# Postprocessing
	if verbose:
		print(f"Calculating final result...", end="\t")
	model.eval()
	with torch.no_grad():
		Y = model(A, X, 'all')
		# print(f"loss = {loss(Y, D, corr_method).item():.4f}.", end="\t")
		Y = pd.DataFrame(Y.cpu().numpy(), index=indices_name, columns=columns_name)
		Y = Y[initial_columns]
	if verbose:
		print("Done.")
		print(f"fast_scBatch finished. The corrected data is returned.")
	return Y