library(sva)
library(reticulate)
use_condaenv("torch")
anndata <- import("anndata", convert = FALSE)
scipy <- import("scipy", convert = FALSE)

dat <- anndata$read_h5ad("./realdata/usoskin/data.h5ad")
batch <- py_to_r(dat$obs)$batch
batch <- as.numeric(batch)
res <- t(py_to_r(dat$X))
res <- ComBat(res, batch)

dat <- anndata$AnnData(X = t(res), obs = py_to_r(dat$obs), var = py_to_r(dat$var))
dat$write_h5ad("./method/combat/results/combat_usoskin.h5ad", compression = "gzip")