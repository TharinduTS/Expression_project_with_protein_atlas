# Expression_project_with_protein_atlas

I am exploring the possibility of using The Human Protein Atlas datasets to find cell type specific gene expression
There methods look very promising and organized
I downloaded cluster data from 
```url
https://www.proteinatlas.org/humanproteome/single+cell/single+cell+type/data
```
load python with needed modules
```
module load gcc arrow
module load python

python -m venv ~/envs/scanpy
source ~/envs/scanpy/bin/activate

python
```
Inside python
```
import pandas as pd
import numpy as np

# Load the TSV file into cell_cluster_data
file_path = "rna_single_cell_cluster.tsv"
cell_cluster_data = pd.read_csv(file_path, sep="\t")

#*******I AM ADDING THIS TO SHORTEN THE DATASET FOR TESTING PURPOSES. COMMENT THIS OUT LATER*****
#cell_cluster_data= cell_cluster_data.head(5)  # Limit to first 5 rows for now

# 1) Aggregate cluster rows to get one expression per (Gene × Cell type)
agg_df = (
    cell_cluster_data
    .groupby(['Gene', 'Gene name', 'Cell type'], as_index=False)
    .agg(nCPM=('nCPM', 'mean'))  # or 'sum' if you prefer total expression
)

# 2) Compute average-of-others and enrichment score, one row per (Gene × Cell type)
#    avg_other = (sum of nCPM for the gene - this cell type's nCPM) / (number of cell types - 1)
gene_sums = agg_df.groupby('Gene')['nCPM'].transform('sum')
gene_counts = agg_df.groupby('Gene')['nCPM'].transform('count')
avg_other = (gene_sums - agg_df['nCPM']) / (gene_counts - 1)

# Avoid division by zero (if a gene is present in only one cell type or others avg is 0)
avg_other = avg_other.replace([np.inf, -np.inf], np.nan)

agg_df['Enrichment score'] = np.where(avg_other > 0, agg_df['nCPM'] / avg_other, np.inf)

# Optional: if you prefer to drop rows where avg_other is NaN (e.g., gene seen in 1 cell type only)
# agg_df = agg_df.dropna(subset=['Enrichment score'])

# Example: show enrichment for ENSG00000000003 (TSPAN6) in ovarian stromal cells
example = agg_df[
    (agg_df['Gene'] == 'ENSG00000000003') &
    (agg_df['Cell type'] == 'ovarian stromal cells')
]
print(example)

# Peek at the first few rows to confirm there's only one row per (Gene × Cell type)
print(agg_df.head())
```



