library(sva)
library(reticulate)
use_condaenv("torch")
anndata <- import("anndata", convert = FALSE)
scipy <- import("scipy", convert = FALSE)

for(i in 1:30){
	dat <- anndata$read_h5ad(paste0("./simulation/data/data",i,".h5ad"))
	batch <- py_to_r(dat$obs)$Batch
	batch <- as.numeric(gsub("Batch","",batch))
	res <- t(py_to_r(dat$X))
	res <- ComBat(res, batch)
	dat <- anndata$AnnData(X = t(res), obs = py_to_r(dat$obs), var = py_to_r(dat$var))
	dat$write_h5ad(paste0("./method/combat/results/combat_simu",i,".h5ad"))
	print(i)
}