import os
import numpy as np
import pandas as pd

def preprocess_data():
	# Load human cell data
	human1 = pd.read_csv("./realdata/pancreas/raw/GSM2230757_human1_umifm_counts.csv", index_col=0)
	human2 = pd.read_csv("./realdata/pancreas/raw/GSM2230758_human2_umifm_counts.csv", index_col=0)
	human3 = pd.read_csv("./realdata/pancreas/raw/GSM2230759_human3_umifm_counts.csv", index_col=0)
	human4 = pd.read_csv("./realdata/pancreas/raw/GSM2230760_human4_umifm_counts.csv", index_col=0)

	# Load mouse cell data
	# mouse1 = pd.read_csv("./realdata/pancreas/raw/GSM2230761_mouse1_umifm_counts.csv", index_col=0)
	# mouse2 = pd.read_csv("./realdata/pancreas/raw/GSM2230762_mouse2_umifm_counts.csv", index_col=0)

	# drop unrelated columns (PISD is duplicated in mouse data)
	human1 = human1.drop(["barcode"], axis=1)
	human2 = human2.drop(["barcode"], axis=1)
	human3 = human3.drop(["barcode"], axis=1)
	human4 = human4.drop(["barcode"], axis=1)
	# mouse1 = mouse1.drop(["barcode", "PISD"], axis=1)
	# mouse2 = mouse2.drop(["barcode", "PISD"], axis=1)
	# mouse1.columns = [x.upper() if x != "assigned_cluster" else x for x in mouse1.columns]
	# mouse2.columns = [x.upper() if x != "assigned_cluster" else x for x in mouse2.columns]

	# add batch information
	human1["batch"] = 1
	human2["batch"] = 2
	human3["batch"] = 3
	human4["batch"] = 4
	# mouse1["batch"] = 5
	# mouse2["batch"] = 6

	# merge all data
	# res = pd.concat([human1, human2, human3, human4, mouse1, mouse2], axis=0).fillna(0)
	res = pd.concat([human1, human2, human3, human4], axis=0).fillna(0)
	res = res.loc[:, (res != 0).any(axis=0)]

	# data label split
	label =  res["assigned_cluster"]
	batch = res["batch"]
	res = res.drop(["assigned_cluster", "batch"], axis=1)

	# log-normalize data
	res = np.log1p(res)

	return res, batch, label

dat, bat, lab = preprocess_data()
# dat.to_csv("./realdata/pancreas/data.csv")
# bat.to_csv("./realdata/pancreas/batch.csv")
# lab.to_csv("./realdata/pancreas/celltype.csv")

import scanpy as sc

adata = sc.AnnData(dat)
adata.obs["batch"] = bat
adata.obs["celltype"] = lab
print(adata)
sc.pp.neighbors(adata, use_rep='X', n_neighbors=10, metric='cosine')
sc.tl.umap(adata)
sc.pl.umap(adata, color=["batch", "celltype"], wspace=0.5, save="_pancreas.pdf")

adata.write("./realdata/pancreas/data.h5ad")