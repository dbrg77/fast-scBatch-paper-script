# Fast-scBatch (paper scripts)

The scripts used in the fast-scBatch paper are stored here.

### Programming Environment

The scripts are suggested to run in a conda evnvironment. For the scripts examples in this repository, we assume the environment name is `torch`.

To run the scripts, the following programming languages/tools should be installed:
* Python (suggested version: 3.11.7)
* R (suggested version: 4.4.0)
* Jupyter Notebook

For the languages listed above, the following packages/libraries and their dependencies should be installed:

**Python**
* `NumPy`
* `Pandas`
* `SciPy`
* `Matplotlib`
* `Seaborn`
* `PyTorch`
* `Scikit-Learn`
* `ScanPy`
* `AnnData`
* `Tqdm`
* `scDML`

**R**
* `reticulate`
* `sva`
* `limma`
* `batchelor`
* `QuantNorm`

### Real Datasets Downloading

Since the original datasets are too large, we cannot directly upload them to the Github repository.

We only keep the folder structure and users can download the datasets locally back to this folder.

The mouse neoron datasets can be downloaded from [Gene Expression Omnibus - Series GSE59739](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE59739).

The human pancreas datasets can be downloaded from [Gene Expression Omnibus - Series GSE84133](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84133).

The datasets are expected to be loaded in the directory `./$dataset_name/raw/`.
