library(batchelor)
library(reticulate)
use_condaenv("torch")
anndata <- import("anndata", convert = FALSE)

for(i in 1:30){
	dat <- anndata$read_h5ad(paste0("./simulation/data/data", i, ".h5ad"))
	batch <- py_to_r(dat$obs)$Batch
	batch <- as.numeric(gsub("Batch","",batch))
	res <- t(py_to_r(dat$X))
	res <- mnnCorrect(res, batch=batch, k=20)
	res <- assays(res)$corrected
	adat <- anndata$AnnData(X = t(res), obs = py_to_r(dat$obs), var = py_to_r(dat$var))
	adat$write_h5ad(paste0("./method/mnn/results/mnn_simu", i, ".h5ad"), compression = "gzip")
	print(i)
}