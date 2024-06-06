library(limma)
library(reticulate)
use_condaenv("torch")
anndata <- import("anndata", convert = FALSE)

for(i in 1:30){
	dat <- anndata$read_h5ad(paste0("./simulation/data/data", i, ".h5ad"))
	batch <- py_to_r(dat$obs)$Batch
	batch <- as.numeric(gsub("Batch", "", batch))
	res <- t(py_to_r(dat$X))
	res <- removeBatchEffect(res, batch=batch)
	adat <- anndata$AnnData(X = t(res), obs = py_to_r(dat$obs), var = py_to_r(dat$var))
	adat$write_h5ad(paste0("./method/limma/results/limma_simu", i, ".h5ad"), compression = "gzip")
	print(i)
}