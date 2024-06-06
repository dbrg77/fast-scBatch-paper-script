import numpy as np
import pandas as pd
import scanpy as sc

def preprocessing():
	dat = pd.read_csv('./realdata/usoskin/raw/data.csv', index_col=0).T
	bat = pd.read_csv('./realdata/usoskin/raw/batch.csv', index_col=0)
	cty = pd.read_csv('./realdata/usoskin/raw/cell_type.csv', index_col=0)
	bat.columns = ['batch']
	cty.columns = ['celltype']
	bat.index = dat.index
	cty.index = dat.index

	adata = sc.AnnData(dat)
	adata.obs['batch'] = bat
	adata.obs['celltype'] = cty

	adata.write('./realdata/usoskin/data.h5ad')

	sc.pp.neighbors(adata, n_neighbors=5, method='cosine')
	sc.tl.umap(adata)
	sc.pl.umap(adata, color=['batch', 'celltype'], save='_usoskin.pdf')

if __name__ == '__main__':
	preprocessing()