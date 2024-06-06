library(QuantNorm)
library(reticulate)
use_condaenv("torch")
anndata <- import("anndata", convert = FALSE)

dat <- anndata$read_h5ad("./realdata/pancreas/data.h5ad")
batch <- py_to_r(dat$obs)$batch
batch <- as.numeric(batch)
res <- t(py_to_r(dat$X))
ccorr <- 1 - QuantNorm(
	res, batch,
	logdat=F, method='row/column',
	cor_method='pearson', max=8
)
write.csv(ccorr, paste0("./method/quantnorm/results/quantnorm_pancreas.csv"))