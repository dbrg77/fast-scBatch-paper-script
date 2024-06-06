library(batchelor)
library(reticulate)
library(parallel)
use_condaenv("torch")
anndata <- import("anndata", convert = FALSE)

dat <- anndata$read_h5ad("./realdata/pancreas/data.h5ad")
batch <- py_to_r(dat$obs)$batch
batch <- as.numeric(batch)
res <- t(py_to_r(dat$X))
res <- mnnCorrect(res, batch=batch, k=20)
res <- assays(res)$corrected
dat <- anndata$AnnData(X = t(res), obs = py_to_r(dat$obs), var = py_to_r(dat$var))
dat$write_h5ad("./method/mnn/results/mnn_pancreas.h5ad", compression = "gzip")