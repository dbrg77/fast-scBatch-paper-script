library(splatter)
library(scater)
library(reticulate)

use_condaenv("torch")
anndata <- import("anndata", convert = FALSE)

n_set = c(240, 360, 600, 840, 960,
		  1440, 2160, 3000, 4020, 5100,
		  6000, 7020
)
p_set = c(5000, 5000, 5000, 5000, 5000,
		  5000, 5000, 5000, 5000, 5000,
		  10000, 10000
)
seeds <- sample(1:10000,12)

for(i in 1:12){
	n <- n_set[i]
	p <- p_set[i]
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
	anndata$AnnData$write(adata, paste0("./simulation/sample/sample",i,".h5ad"))
}