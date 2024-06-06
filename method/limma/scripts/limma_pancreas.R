library(limma)
library(reticulate)
use_condaenv("torch")
anndata <- import("anndata", convert = FALSE)

dat <- anndata$read_h5ad("./realdata/pancreas/data.h5ad")
batch <- py_to_r(dat$obs)$batch
batch <- as.numeric(batch)
res <- t(py_to_r(dat$X))
res <- removeBatchEffect(res, batch=batch)
adat <- anndata$AnnData(X = t(res), obs = py_to_r(dat$obs), var = py_to_r(dat$var))
adat$write_h5ad("./method/limma/results/limma_pancreas.h5ad", compression = "gzip")