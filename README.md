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

# Rename column nCPM to avg_nCPM
agg_df = agg_df.rename(columns={'nCPM': 'avg_nCPM'})

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

# Count unique (non-null) values in 'Cell type'
n_unique = agg_df['Cell type'].dropna().nunique()

# Create a sorted list of unique (non-null) cell types
unique_cell_types = sorted(agg_df['Cell type'].dropna().unique().tolist())

# Save to TSV (one column named 'Cell type')
pd.Series(unique_cell_types, name='Cell type').to_csv('unique_cell_types.tsv', sep='\t', index=False)

print(f"\033[32mNumber of unique cell types: {n_unique}\033[0m")

# overwrite agg_df:
agg_df = agg_df_filtered

# Count unique gene names after dropping genes with no expression
n_unique_genes = agg_df['Gene'].dropna().nunique()

# Print in green
print(f"\033[32mNumber of unique genes remaining: {n_unique_genes}\033[0m")

#****

print("\033[33mCalculating Enrichment Scores....\033[0m")

# --- enrichment score calculation  ---
def add_enrichment(
    agg_df: pd.DataFrame,
    gene_col: str = "Gene",
    value_col: str = "nCPM",
    out_col: str = "Enrichment score",
    min_background: float = 1e-3,   # minimum allowed background (denominator)
    min_expression: float = 0.0,    # minimum required numerator (nCPM) to compute enrichment
    min_count: int = 2,             # require at least this many cell-type entries per gene
    pseudocount: float | None = None,  # optional stabilizer added to background (and/or numerator)
    clip_max: float | None = None      # optional: cap extreme enrichment values
):
    """
    Compute enrichment per row as nCPM / average(nCPM of other cell types for the same gene),
    with safeguards to avoid infinities and noise from tiny denominators.

    Parameters
    ----------
    agg_df : DataFrame with at least [gene_col, value_col]
    gene_col : Name of the gene column
    value_col : Name of the expression column (e.g., 'nCPM')
    out_col : Name for the output enrichment column
    min_background : Minimum allowed background mean (denominator); values below are raised to this
    min_expression : Minimum required numerator; rows below become NaN
    min_count : Require at least `min_count` rows for a given gene to compute enrichment
    pseudocount : If provided, added to background (and optionally numerator) to stabilize low values
    clip_max : If provided, cap (winsorize) extreme enrichment at this value

    Returns
    -------
    df : DataFrame copy with an added enrichment column
    """
    df = agg_df.copy()
    # Ensure numeric; coerce invalids to NaN
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    # Group-level sums and counts
    gene_sums   = df.groupby(gene_col)[value_col].transform("sum")
    gene_counts = df.groupby(gene_col)[value_col].transform("count")
    # Average of "other" cell types = (sum - current) / (count - 1)
    denom_counts = gene_counts - 1
    avg_other = (gene_sums - df[value_col]) / denom_counts
    # Invalid if there is no "other" (i.e., count <= 1)
    avg_other = avg_other.mask(denom_counts <= 0, np.nan)
    # Optional stabilization via pseudocount on the denominator (and numerator if desired)
    if pseudocount is not None:
        # You can also add pseudocount to numerator; uncomment if desired:
        # df[value_col] = df[value_col] + pseudocount
        avg_other = avg_other + pseudocount
    # Enforce minimum background: raise small denominators to min_background
    # (keeps finite ratios instead of inf; if you'd rather flag them, change logic below)
    denom = np.maximum(avg_other, min_background)
    # Enforce minimum expression: rows below threshold are set to NaN (not artificially inflated)
    numer = df[value_col].where(df[value_col] >= min_expression, np.nan)
    # Safe divide (NaN when denom <= 0 or numer is NaN)
    df[out_col] = np.divide(
        numer, denom,
        out=np.full(df.shape[0], np.nan, dtype=float),
        where=(denom > 0)
    )
    # If avg_other itself is NaN (e.g., insufficient counts), keep NaN
    df.loc[avg_other.isna(), out_col] = np.nan
    # Optional: cap extreme ratios to reduce leverage of outliers
    if clip_max is not None:
        df[out_col] = df[out_col].clip(upper=clip_max)
    return df


# --- Compute enrichment with sensible safeguards ---
agg_df = add_enrichment(
    agg_df,
    gene_col="Gene",
    value_col="avg_nCPM",
    out_col="Enrichment score",
    min_background=1e-3,     # lift very small denominators
    min_expression=0.0,      # require >= this to compute enrichment
    min_count=2,             # at least 2 entries per gene
    pseudocount=None,        # try setting to e.g. 0.01 for stabilization
    clip_max=None            # e.g., set to 100 to cap extreme ratios
)
#****

print("\033[33mDone calculating\033[0m")

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

if not single_cell_rows.empty:
    n_genes = single_cell_rows['Gene'].nunique()
    print(f"\033[32mFound {n_genes} gene(s) expressed in exactly one cell type.\033[0m")
    print(single_cell_rows.to_string(index=False))
    single_cell_rows.to_csv('single_cell_type_gene_rows.tsv', sep='\t', index=False)
else:
    print("\033[33mNo genes were found that are only expressed in one cell type\033[0m")

#******
#ssort agg_df by enrichment
agg_df=agg_df.sort_values(by="Enrichment score", ascending=False)

# Save agg_df to a TSV file
agg_df.to_csv('all_gene_cell_enrichment_data.tsv', sep='\t', index=False)
print("\033[33mFiles saved: all_gene_cell_enrichment_data.tsv\033[0m")

# select top x rows

# Sort by Enrichment score in descending order and take top 100
top100 = agg_df.head(100)

# Save to TSV file
top100.to_csv("top100_enrichment.tsv", sep="\t", index=False)

print("Saved top 100 rows to top100_enrichment.tsv")

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
Finally you can make an interactive html plot with the following python script
```py

import argparse
import json
import pandas as pd
import plotly.graph_objects as go

# ---------- Data Loading ----------
def load_tsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    # Clean column names
    df.columns = [c.strip() for c in df.columns]
    expected = [
        "Gene", "Gene name", "Cell type", "avg_nCPM",
        "clusters_used", "Enrichment score", "single_cell_type_gene"
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # Coerce numeric columns
    for col in ["avg_nCPM", "clusters_used", "Enrichment score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Label: use Gene name, fallback to Gene
    df["Label"] = df["Gene name"].fillna(df["Gene"])

    # Drop rows missing critical fields
    df = df.dropna(subset=["Label", "Enrichment score"])

    return df

# ---------- Figure Building ----------
def build_fig(
    df: pd.DataFrame,
    top_n: int = 100,
    use_log: bool = True,
    orientation: str = "v",        # 'v' (vertical bars) or 'h' (horizontal)
    log_dtick: str | None = None,  # e.g., "D1" or "D2" for log minor ticks
    initial_zoom: int | None = None,  # initial number of bars to show (zoomed view)
):
    """
    Build the Plotly figure and a JSON payload for robust client-side filtering.
    initial_zoom: if provided, sets initial axis range to show only the first N bars.
    """
    import plotly.express as px

    # Filter if using log scale (must be > 0)
    if use_log:
        df = df[df["Enrichment score"] > 0]

    # Sort by enrichment and select top N
    df_plot = df.sort_values("Enrichment score", ascending=False).head(top_n).copy()

    # Prepare customdata for click handler (keeps full row for details)
    detail_cols = [
        "Gene", "Gene name", "Cell type", "avg_nCPM",
        "clusters_used", "Enrichment score", "single_cell_type_gene"
    ]
    customdata = df_plot[detail_cols].values

    # Stable color mapping (sorted for reproducibility)
    colors = px.colors.qualitative.Safe
    ctypes = sorted(df_plot["Cell type"].astype(str).unique())
    cmap = {ct: colors[i % len(colors)] for i, ct in enumerate(ctypes)}
    bar_colors = df_plot["Cell type"].astype(str).map(cmap)

    # Build bar chart
    if orientation == "v":
        x_vals = df_plot["Label"]
        y_vals = df_plot["Enrichment score"]
        bar = go.Bar(
            x=x_vals,
            y=y_vals,
            customdata=customdata,
            marker_color=bar_colors,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Enrichment: %{y:.2f}<br>"
                "Cell type: %{customdata[2]}<br>"
                "avg_nCPM: %{customdata[3]}<br>"
                "clusters_used: %{customdata[4]}<br>"
                "<extra></extra>"
            ),
        )
    else:
        x_vals = df_plot["Enrichment score"]
        y_vals = df_plot["Label"]
        bar = go.Bar(
            x=x_vals,
            y=y_vals,
            orientation="h",
            customdata=customdata,
            marker_color=bar_colors,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Enrichment: %{x:.2f}<br>"
                "Cell type: %{customdata[2]}<br>"
                "avg_nCPM: %{customdata[3]}<br>"
                "clusters_used: %{customdata[4]}<br>"
                "<extra></extra>"
            ),
        )

    fig = go.Figure(data=[bar])

    # Layout
    fig.update_layout(
        title="Gene Enrichment",
        xaxis_title=("Enrichment score" if orientation == "h" else "Gene name"),
        yaxis_title=("Gene name" if orientation == "h" else "Enrichment score"),
        template="plotly_white",
        height=700,
        bargap=0.2,
        margin=dict(l=60, r=30, t=60, b=120),
        hovermode="closest",
        legend_title_text="Cell type",
    )

    # Axes tweaks
    if orientation == "v":
        fig.update_xaxes(
            tickangle=-45,
            automargin=True,
            categoryorder="array",
            categoryarray=list(df_plot["Label"])
        )
        if use_log:
            kwargs = dict(type="log", title="Enrichment score (log scale)")
            if log_dtick in ("D1", "D2"):
                kwargs["dtick"] = log_dtick
            fig.update_yaxes(**kwargs)
        else:
            fig.update_yaxes(title="Enrichment score")
    else:
        fig.update_yaxes(
            automargin=True,
            categoryorder="array",
            categoryarray=list(df_plot["Label"])
        )
        if use_log:
            kwargs = dict(type="log", title="Enrichment score (log scale)")
            if log_dtick in ("D1", "D2"):
                kwargs["dtick"] = log_dtick
            fig.update_xaxes(**kwargs)
        else:
            fig.update_xaxes(title="Enrichment score")

    # Initial zoom (index-based range on the categorical axis)
    if initial_zoom is not None:
        zoom_n = max(1, min(int(initial_zoom), len(df_plot)))
        # Category axis range uses index positions: [-0.5, zoom_n - 0.5]
        if orientation == "v":
            fig.update_xaxes(range=[-0.5, zoom_n - 0.5])
        else:
            fig.update_yaxes(range=[-0.5, zoom_n - 0.5])

    # Build JSON payload for robust client-side filtering
    rows = df_plot[[
        "Gene", "Gene name", "Cell type", "avg_nCPM",
        "clusters_used", "Enrichment score", "single_cell_type_gene", "Label"
    ]].to_dict(orient="records")

    payload = {
        "rows": rows,
        "colors": cmap,        # cell type -> color
        "orientation": orientation,
        "detail_cols": ["Gene","Gene name","Cell type","avg_nCPM","clusters_used","Enrichment score","single_cell_type_gene"],
        "label_col": "Label",  # for X/Y labels
        "enrich_col": "Enrichment score"
    }

    return fig, payload

# ---------- Enhanced HTML (Details + Filters + Search) ----------
DETAILS_SNIPPET = """
<style>
#controls {
  font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
  margin: 12px 0; display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
}
#controls label { font-size: 14px; color: #333; }
#controls select, #controls input { padding: 6px 8px; font-size: 14px; }
#controls button { padding: 6px 10px; font-size: 14px; cursor: pointer; }
#count-info { font-size: 13px; color: #666; margin-left: 8px; }
#details-panel {
  font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
  margin-top: 12px; padding: 12px;
  border: 1px solid #ddd; border-radius: 8px; background: #fafafa;
  white-space: pre-wrap; font-size: 14px;
}
#details-panel table { border-collapse: collapse; width: 100%; }
#details-panel td { padding: 4px 8px; border-bottom: 1px solid #eee; vertical-align: top; }
#details-panel td:first-child { font-weight: 600; color: #333; width: 220px; }
.plotly-graph-div { width: 100% !important; }
</style>

<div id="controls">
  <label>Cell type:
    <select id="celltype-filter"><option value="__all__">All cell types</option></select>
  </label>
  <label style="margin-left:12px;">Gene search:
    <input type="text" id="gene-search" placeholder="e.g., LALBA">
  </label>
  <button id="reset-filters" title="Clear filters">Reset</button>
  <span id="count-info"></span>
</div>

<div id="details-panel"><em>Click a bar to see full row details here.</em></div>

<!-- The Python side will inject a <script id="chart-data" type="application/json">...</script> before this script -->
<script>
(function(){
  // Wait until the Plotly figure exists and is initialized
  function initOnceReady() {
    var gd = document.querySelector('div.plotly-graph-div') || document.querySelector('.js-plotly-plot');
    var dataTag = document.getElementById('chart-data');
    if (!gd || !gd.data || !gd.data.length || !dataTag) { setTimeout(initOnceReady, 50); return; }

    // Parse embedded JSON payload from Python
    var payload = {};
    try { payload = JSON.parse(dataTag.textContent); } catch(e) { console.error("JSON parse error:", e); return; }
    var rows = Array.isArray(payload.rows) ? payload.rows : [];
    var colors = payload.colors || {};
    var orientation = payload.orientation || 'v';
    var detailCols = payload.detail_cols || ["Gene","Gene name","Cell type","avg_nCPM","clusters_used","Enrichment score","single_cell_type_gene"];
    var labelCol = payload.label_col || "Label";
    var enrichCol = payload.enrich_col || "Enrichment score";

    // Controls
    var sel = document.getElementById('celltype-filter');
    var searchBox = document.getElementById('gene-search');
    var resetBtn = document.getElementById('reset-filters');
    var panel = document.getElementById('details-panel');
    var countInfo = document.getElementById('count-info');

    // Number formatter
    function fmtNumber(val, maxDigits=2) {
      if (val === null || val === undefined || val === "" || isNaN(val)) return String(val ?? "");
      var num = Number(val);
      if (!isFinite(num)) return String(val);
      return num.toLocaleString('en-US', { maximumFractionDigits: maxDigits });
    }

    // Build base arrays from rows (authoritative source)
    var baseLabels = rows.map(function(r){ return r[labelCol]; });
    var baseEnrich = rows.map(function(r){ return r[enrichCol]; });
    var baseCustom = rows.map(function(r){
      return detailCols.map(function(k){ return r[k]; });
    });
    var baseColors = rows.map(function(r){
      var ct = (r["Cell type"] == null ? "" : String(r["Cell type"]).trim());
      return colors[ct] || '#636EFA';
    });

    var N = rows.length;

    // Populate unique cell types into dropdown
    var cellTypesSet = {};
    for (var i = 0; i < N; i++) {
      var ct = rows[i]["Cell type"];
      if (ct != null) {
        ct = String(ct).trim();
        if (ct.length) cellTypesSet[ct] = true;
      }
    }
    var cellTypes = Object.keys(cellTypesSet).sort();
    cellTypes.forEach(function(ct){
      var opt = document.createElement('option');
      opt.value = ct; opt.textContent = ct;
      sel.appendChild(opt);
    });

    // Details panel render
    function renderDetails(customdata) {
      if (!customdata || !customdata.length) return;
      var html = "<table>";
      for (var i = 0; i < detailCols.length; i++) {
        var key = detailCols[i];
        var val = customdata[i];
        if (key === "avg_nCPM" || key === "Enrichment score") { val = fmtNumber(val, 2); }
        html += "<tr><td>" + key + "</td><td>" + (val === null ? "" : val) + "</td></tr>";
      }
      html += "</table>";
      panel.innerHTML = html;
      panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    // Filtering
    function applyFilter() {
      var selectedCT = sel.value;
      var q = (searchBox.value || "").trim().toLowerCase();

      var fx = [], fy = [], fcd = [], fcol = [];
      for (var i = 0; i < N; i++) {
        var row = rows[i];
        var ct = (row["Cell type"] == null ? "" : String(row["Cell type"]).trim());
        var geneName = String(row["Gene name"] || "").toLowerCase();
        var labelLower = String(row[labelCol] || "").toLowerCase();

        var ctOk = (selectedCT === "__all__") || (selectedCT === ct);
        var qOk = (!q) || geneName.includes(q) || labelLower.includes(q);

        if (ctOk && qOk) {
          if (orientation === 'h') {
            fx.push(row[enrichCol]);  // enrichment on x
            fy.push(row[labelCol]);   // labels on y
          } else {
            fx.push(row[labelCol]);   // labels on x
            fy.push(row[enrichCol]);  // enrichment on y
          }
          fcd.push(detailCols.map(function(k){ return row[k]; }));
          fcol.push(colors[ct] || '#636EFA');
        }
      }

      // Restyle robustly
      var update = { x: [fx], y: [fy], customdata: [fcd], marker: [{ color: fcol }] };
      Plotly.restyle(gd, update, [0]);

      // Keep category order aligned with filtered labels
      var relayout = {};
      if (orientation === 'h') {
        relayout['yaxis.categoryorder'] = 'array';
        relayout['yaxis.categoryarray'] = fy;
      } else {
        relayout['xaxis.categoryorder'] = 'array';
        relayout['xaxis.categoryarray'] = fx;
      }
      Plotly.relayout(gd, relayout);

      countInfo.textContent = (fx.length) + " shown of " + N;
    }

    // Wire events (no initial restyle; plot stays intact)
    sel.addEventListener('change', applyFilter);
    searchBox.addEventListener('input', applyFilter);
    resetBtn.addEventListener('click', function(){
      sel.value = "__all__";
      searchBox.value = "";
      applyFilter();
    });

    // Initial count
    countInfo.textContent = N + " shown of " + N;

    // Click → details
    var clickHandler = function(evt) {
      var e = evt && evt.points ? evt : (evt && evt.detail ? evt.detail : null);
      if (!e || !e.points || !e.points.length) return;
      var p = e.points[0];
      renderDetails(p.customdata);
    };
    if (typeof gd.on === 'function') gd.on('plotly_click', clickHandler);
    else gd.addEventListener('plotly_click', clickHandler);
  }

  // Start readiness check
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initOnceReady);
  } else {
    initOnceReady();
  }
})();
</script>
"""

# ---------- HEAD meta for mobile ----------
HEAD_SNIPPET = """<meta name="viewport" content="width=device-width, initial-scale=1">"""

# ---------- HTML Saving (standards-compliant, embeds JSON payload) ----------
def save_html(
    fig: go.Figure,
    out_path: str,
    payload: dict,
    self_contained: bool = False,
    lang_code: str = "en"
):
    """
    Save a robust, standards-compliant HTML:
      - Ensures <!DOCTYPE html> at the very beginning (No Quirks Mode).
      - Adds lang="..." on <html>.
      - Injects viewport meta just after <head>.
      - Embeds a JSON payload with rows & colors for reliable client-side filtering.
      - Appends DETAILS_SNIPPET before </body>.
    """
    include_js = True if self_contained else "cdn"
    html = fig.to_html(full_html=True, include_plotlyjs=include_js)

    import re

    # 1) Ensure <!DOCTYPE html> at the very beginning
    html = html.lstrip("\ufeff\r\n\t ")
    if not re.match(r"(?is)^<!doctype\s+html>", html):
        html = "<!DOCTYPE html>\n" + html

    # 2) Ensure <html lang="...">
    if re.search(r"(?is)<html(?:\s[^>]*)?>", html):
        if not re.search(r"(?is)<html[^>]*\blang\s*=", html):
            html = re.sub(r"(?is)<html(\s*)>", f'<html lang="{lang_code}">', html, count=1)
    else:
        html = f"<!DOCTYPE html>\n<html lang=\"{lang_code}\">\n{html}\n</html>"

    # 3) Inject <meta name="viewport"> immediately after <head>
    if re.search(r"(?is)<head\s*>", html):
        html = re.sub(r"(?is)<head\s*>", "<head>\n" + HEAD_SNIPPET + "\n", html, count=1)
    else:
        html = re.sub(
            r"(?is)(<html[^>]*>)",
            r"\1\n<head>\n" + HEAD_SNIPPET + "\n</head>\n",
            html,
            count=1
        )

    # 4) Embed JSON payload (safe single <script> with application/json)
    json_str = json.dumps(payload, ensure_ascii=False)
    data_block = f'<script id="chart-data" type="application/json">{json_str}</script>\n'

    # Place the data block just before DETAILS_SNIPPET and </body>
    if re.search(r"(?is)</body\s*>", html):
        html = re.sub(r"(?is)</body\s*>", data_block + DETAILS_SNIPPET + "\n</body>", html, count=1)
    else:
        if not re.search(r"(?is)<body[^>]*>", html):
            html = re.sub(r"(?is)</head\s*>", "</head>\n<body>\n", html, count=1)
        html = html + "\n" + data_block + DETAILS_SNIPPET + "\n</body>\n</html>"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Make interactive gene enrichment chart (TSV → HTML) with details, cell-type filter, and gene search")
    ap.add_argument("--file", "-f", required=True, help="Input TSV file (e.g., top100_enrichment.tsv)")
    ap.add_argument("--out", "-o", default="interactive_markers.html", help="Output HTML file")
    ap.add_argument("--top", "-n", type=int, default=100, help="Top N genes by enrichment")
    ap.add_argument("--log", action="store_true", help="Use log scale on enrichment axis")
    ap.add_argument("--linear", action="store_true", help="Use linear scale on enrichment axis")
    ap.add_argument("--horizontal", action="store_true", help="Use horizontal bars (better for long labels)")
    ap.add_argument("--self-contained", action="store_true", help="Embed plotly.js for fully offline HTML")
    ap.add_argument("--log-digits", choices=["D1","D2"], help="Log-axis minor digits: D1 (all) or D2 (2 & 5)")
    ap.add_argument("--lang", default="en", help="HTML lang attribute (e.g., 'en', 'en-CA')")
    ap.add_argument("--initial-zoom", type=int, default=100, help="Initial number of bars to show on load (zoomed view)")
    args = ap.parse_args()

    if args.log and args.linear:
        raise SystemExit("Choose either --log or --linear (not both).")
    use_log = True if args.log or (not args.linear) else False
    orientation = "h" if args.horizontal else "v"

    df = load_tsv(args.file)
    fig, payload = build_fig(
        df,
        top_n=args.top,
        use_log=use_log,
        orientation=orientation,
        log_dtick=args.log_digits,
        initial_zoom=args.initial_zoom,  # NEW: apply initial zoom range
    )
    save_html(fig, args.out, payload, self_contained=args.self_contained, lang_code=args.lang)

    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()
```
RUN it like this
-f -input file
-0 -output file
-n no of columns/gene-enrichment combos to process
--log log transforms the scale
--self-contained -makes it usable offline
--initial-zoom 100 -set initial zoom level to this many columns
```
 python plot_interactive.py -f all_gene_cell_enrichment_data.tsv -o interactive_markers.html -n 1000 --log --self-contained
```
