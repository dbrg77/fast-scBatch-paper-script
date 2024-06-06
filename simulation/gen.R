library(splatter)
library(scater)
library(reticulate)

use_condaenv("torch")
anndata <- import("anndata", convert = FALSE)

# Simulate data for three stages:
# n = 360, n = 600, n=960
# users should modify the n and p to simulate data manually.

n <- 360
p <- 4000
seeds <- sample(1:10000,30)

# The range of i should also be adjusted manually.

for(i in 30:30){
	sim.groups <- splatSimulate(
		nGenes = p, batchCells = c(n/3,n/4,n/4,n/6),
		seed = seeds[i], group.prob = c(0.4,0.3,0.2,0.1),
		de.prob = c(0.05,0.05,0.05,0.05), batch.facLoc = c(0.1,0.25,0.05,0.15), 
		batch.facScale = c(0.03,0.2,0.4,0.12), method = "groups", verbose = F)
	sim.groups <- scater::logNormCounts(sim.groups)
	adata <- anndata$AnnData(
		X = t(assays(sim.groups)$logcounts),
		obs = data.frame(colData(sim.groups)),
		var = data.frame(rowData(sim.groups))
	)
	anndata$AnnData$write(adata, paste0("./simulation/data/data",i,".h5ad"))
}