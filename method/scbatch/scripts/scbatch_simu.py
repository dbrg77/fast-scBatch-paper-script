import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import anndata
from tqdm import trange

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Network(nn.Module):
	def __init__(self) -> None:
		super(Network, self).__init__()
		self.alpha = nn.Parameter(torch.eye(n))
		self.beta = nn.Parameter(torch.zeros(p, n))
	def forward(self, X):
		return torch.matmul(X, self.alpha) + self.beta
def loss(Y, std):
	corr = torch.corrcoef(Y.T)
	return torch.norm(corr - std, p='fro')

for i in trange(1, 11):
	cell = anndata.read_h5ad(f"./simulation/data/data{i}.h5ad")
	batch = cell.obs[["Batch"]].copy()
	cells = cell.to_df().T
	corr = pd.read_csv(f"./method/quantnorm/results/quantnorm_simu{i}.csv", index_col=0)
	corr.columns = cells.columns
	corr.index = cells.columns
	p, n = cells.shape
	cells = cells.values
	corr = corr.values
	X = torch.from_numpy(cells).float().to(device)
	D = torch.from_numpy(corr).float().to(device)

	model = Network().to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
	EPOCHS = 300
	losses = []
	for epoch in range(EPOCHS):
		optimizer.zero_grad()
		Y = model(X)
		loss_val = loss(Y, D)
		losses.append(loss_val.item())
		loss_val.backward()
		optimizer.step()
	
	# plt.plot(np.arange(EPOCHS), losses, label="loss")
	# plt.legend()
	# plt.show()
	
	model.eval()
	Y = model(X)
	adata = anndata.AnnData(X=Y.cpu().detach().numpy().T, obs=cell.obs, var=cell.var)
	adata.write(f"./method/scbatch/results/scbatch_simu{i}.h5ad")