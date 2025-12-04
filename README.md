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

# --- Aggregation to one row per (Gene × Cell type) with mean nCPM ---
agg_df = (
    cell_cluster_data
    .groupby(['Gene', 'Gene name', 'Cell type'], as_index=False)
    .agg(nCPM=('nCPM', 'mean'))
)

# --- Count how many clusters contributed to each (Gene × Cell type) ---
# Prefer counting unique cluster IDs if a 'Cluster' column exists; otherwise use row count.
if 'Cluster' in cell_cluster_data.columns:
    cluster_counts = (
        cell_cluster_data
        .groupby(['Gene', 'Gene name', 'Cell type'])['Cluster']
        .nunique()
        .reset_index(name='clusters_used')
    )
else:
    cluster_counts = (
        cell_cluster_data
        .groupby(['Gene', 'Gene name', 'Cell type'])
        .size()
        .reset_index(name='clusters_used')
    )

# --- Merge counts into agg_df ---
agg_df = agg_df.merge(cluster_counts, on=['Gene', 'Gene name', 'Cell type'], how='left')

# --- (Optional) If you want the cell type label to include the count for easy viewing ---
#agg_df['Cell type (with clusters)'] = (
#    agg_df['Cell type'].astype(str) + ' (' + agg_df['clusters_used'].astype(int).astype(str) + ' clusters)'
#)

# --- enrichment score calculation  ---
gene_sums = agg_df.groupby('Gene')['nCPM'].transform('sum')
gene_counts = agg_df.groupby('Gene')['nCPM'].transform('count')
avg_other = (gene_sums - agg_df['nCPM']) / (gene_counts - 1)

# Avoid division by zero or inf
avg_other = avg_other.replace([np.inf, -np.inf], np.nan)
agg_df['Enrichment score'] = np.where(avg_other > 0, agg_df['nCPM'] / avg_other, np.inf)

# Rename column nCPM to avg_nCPM
agg_df = agg_df.rename(columns={'nCPM': 'avg_nCPM'})

# Peek
print(agg_df.head())
```
double checking output by checking rows for different cell types
```

# Define the gene and cell type you want to filter
target_gene = "TSPAN6"       # Replace with your gene symbol or name
target_cell_type = "adipocytes"  # Replace with your cell type

# Filter rows matching both conditions
filtered_df = cell_cluster_data[
    (cell_cluster_data['Gene name'] == target_gene) &
    (cell_cluster_data['Cell type'] == target_cell_type)
]

# Display the result
filtered_df
```

