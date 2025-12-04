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

#checking for non numeric values in nCPM to avoid errors

# Convert to numeric, invalid entries become NaN
numeric_series = pd.to_numeric(agg_df['avg_nCPM'], errors='coerce')
# Find unique non-numeric values
non_numeric_values = agg_df.loc[numeric_series.isna(), 'avg_nCPM'].unique()
# Check and print message
if len(non_numeric_values) > 0:
    print(f"\033[31mNon numeric values found in avg_nCPM: {non_numeric_values}\033[0m")
else:
    print("\033[32mGOOD DATA nCPM. NO Non numeric\033[0m")


#****This is a function to drop rows with genes that do not express in any cell types ***
def drop_genes_with_no_expression(agg_df, expr_col=None, treat_nan_as_zero=False):
    """
    Remove rows for genes whose expression is zero across all cell types.
    Parameters
    ----------
    agg_df : pd.DataFrame
        Aggregated DataFrame with one row per (Gene × Cell type).
    expr_col : str or None
        Column with aggregated expression (defaults to 'avg_nCPM' if present, else 'nCPM').
    treat_nan_as_zero : bool
        If True, NaN values are treated as zero when determining 'no expression'.
        If False, genes with NaN-only expressions will NOT be classified as zero; they are kept.
    Returns
    -------
    filtered_df : pd.DataFrame
        agg_df with rows for zero-expression genes dropped.
    dropped_genes : list
        List of gene identifiers that were removed.
    """
    # Pick expression column
    if expr_col is None:
        expr_col = 'avg_nCPM' if 'avg_nCPM' in agg_df.columns else 'nCPM'
    # Work on a copy
    df = agg_df.copy()
    # Ensure numeric and clean infs
    df[expr_col] = pd.to_numeric(df[expr_col], errors='coerce')
    df[expr_col] = df[expr_col].replace([np.inf, -np.inf], np.nan)
    # Optionally treat NaN as zero for the test
    expr_for_test = df[expr_col].fillna(0) if treat_nan_as_zero else df[expr_col]
    # Compute per-gene max expression across all cell types
    gene_max = expr_for_test.groupby(df['Gene']).max()
    # Genes where max == 0 → zero across all rows
    genes_all_zero = gene_max[gene_max == 0].index.tolist()
    # Filter out those genes
    filtered_df = df[~df['Gene'].isin(genes_all_zero)].copy()
    return filtered_df, genes_all_zero

# ---- Example usage ----
# Run this BEFORE enrichment calculation (recommended), or after if needed.
agg_df_filtered, dropped = drop_genes_with_no_expression(agg_df, expr_col=None, treat_nan_as_zero=False)

print(f"\033[31mDropped {len(dropped)} row(s) with zero expression across all cell types.\033[0m")

# Count unique gene names after dropping genes with no expression
n_unique_genes = agg_df['Gene'].dropna().nunique()

# Print in green
print(f"\033[32mNumber of unique genes remaining: {n_unique_genes}\033[0m")

# Count unique (non-null) values in 'Cell type'
n_unique = agg_df['Cell type'].dropna().nunique()

# Create a sorted list of unique (non-null) cell types
unique_cell_types = sorted(agg_df['Cell type'].dropna().unique().tolist())

# Save to TSV (one column named 'Cell type')
pd.Series(unique_cell_types, name='Cell type').to_csv('unique_cell_types.tsv', sep='\t', index=False)

print(f"\033[32mNumber of unique cell types: {n_unique}\033[0m")

# overwrite agg_df:
agg_df = agg_df_filtered

# (Optional) Save the cleaned agg_df to TSV
agg_df.to_csv('all_gene_cell_enrichment_data.cleaned.tsv', sep='\t', index=False)

#****

print("\033[33mCalculating Enrichment Scores....\033[0m")

# --- enrichment score calculation  ---
# gene sums gives me the sum of all nCPM values across all cell types for each gene
gene_sums = agg_df.groupby('Gene')['nCPM'].transform('sum')
#gene counts gives me how many cell types entries it has
gene_counts = agg_df.groupby('Gene')['nCPM'].transform('count')
#average nCPM across all other cell types for the same gene
avg_other = (gene_sums - agg_df['nCPM']) / (gene_counts - 1)

# Avoid division by zero or inf
avg_other = avg_other.replace([np.inf, -np.inf], np.nan)
agg_df['Enrichment score'] = np.where(avg_other > 0, agg_df['nCPM'] / avg_other, np.inf)

print("\033[33mDone calculating\033[0m")

# Rename column nCPM to avg_nCPM
agg_df = agg_df.rename(columns={'nCPM': 'avg_nCPM'})



#******
# --- Choose the expression column (use avg_nCPM if you renamed it) ---
expr_col = 'avg_nCPM' if 'avg_nCPM' in agg_df.columns else 'nCPM'

# --- Build/confirm the single-cell-type flag ---
gene_celltype_counts = agg_df.groupby('Gene')['Cell type'].transform('nunique')
agg_df['single_cell_type_gene'] = (gene_celltype_counts == 1)

min_count = min(gene_celltype_counts)
print(f"\033[33mNumber of cell types for the gene expressed in the least amount of genes: {min_count}\033[0m")

# Show rows for single-cell-type genes ---
single_cell_rows = agg_df[agg_df['single_cell_type_gene']].copy()

if not single_rows.empty:
    n_genes = single_rows['Gene'].nunique()
    print(f"\033[32mFound {n_genes} gene(s) expressed in exactly one cell type.\033[0m")
    print(single_rows.to_string(index=False))
    single_cell_rows.to_csv('single_cell_type_gene_rows.tsv', sep='\t', index=False)
else:
    print("\033[33mNo genes were found that are only expressed in one cell type\033[0m")

#******

# Save agg_df to a TSV file
agg_df.to_csv('all_gene_cell_enrichment_data.tsv', sep='\t', index=False)

print("\033[33mFiles saved: all_gene_cell_enrichment_data.tsv\033[0m")

#selecting rows with highest enrichment values
#****this is the function
def top_percent_global(df, score_col='Enrichment score', pct=0.05, include_infinite=True):
    """
    Returns the top `pct` fraction of rows by `score_col` across the whole DataFrame.
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (e.g., agg_df).
    score_col : str
        Column name containing enrichment scores.
    pct : float
        Fraction for percentile cutoff (e.g., 0.05 = top 5%).
    include_infinite : bool
        If True, treat np.inf scores as very large values (included in top).
        If False, exclude ∞ scores.
    """
    work = df.copy()
    if include_infinite:
        # Map inf to a very large finite number so they rank at the top
        work['__score_finite__'] = work[score_col].replace(np.inf, np.finfo(float).max)
    else:
        # Replace inf/-inf with NaN and drop them
        work['__score_finite__'] = work[score_col].replace([np.inf, -np.inf], np.nan)
    # Drop rows without a usable score
    work = work.dropna(subset=['__score_finite__'])
# Compute percentile threshold (e.g., 95th percentile for pct=0.05)
    threshold = work['__score_finite__'].quantile(1 - pct)
    # Select rows at or above threshold
    top_df = work[work['__score_finite__'] >= threshold].copy()
    top_df = top_df.sort_values(by=score_col, ascending=False)
    # Clean up helper column
    top_df = top_df.drop(columns=['__score_finite__'])
    return top_df, threshold

# Example usage (top 0.05%)
#*** Change percentile here
top5_df, thresh = top_percent_global(agg_df, score_col='Enrichment score', pct=0.0005, include_infinite=True)
print(f"Global top 5% threshold: {thresh}")
print(top5_df[['Gene', 'Gene name', 'Cell type', 'Enrichment score', 'clusters_used']])


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

