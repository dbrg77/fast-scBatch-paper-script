library(QuantNorm)
library(reticulate)
use_condaenv("torch")
anndata <- import("anndata", convert = FALSE)

for(i in 11:30){
	use_condaenv("torch")
	anndata <- import("anndata", convert = FALSE)
	dat <- anndata$read_h5ad(paste0("./simulation/data/data", i, ".h5ad"))
	batch <- py_to_r(dat$obs)$Batch
	batch <- as.numeric(gsub("Batch", "", batch))
	res <- t(py_to_r(dat$X))
	ccorr <- 1 - QuantNorm(
		res, batch,
		logdat=F, method='row/column',
		cor_method='pearson', max=10
	)
	write.csv(ccorr, paste0("./method/quantnorm/results/quantnorm_simu", i, ".csv"))
	print(i)
}