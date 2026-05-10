import scanpy as sc
import squidpy as sq
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import anndata as ad
from matplotlib.colors import ListedColormap, to_rgb
import skimage
from matplotlib.colors import TwoSlopeNorm

''' Plots clusters and ssGSEA scores on the tissue. The images are saved.'''

# plot the scores per cluster
def score_per_cluster(adata, ssgsea_df, gene_set,  # column name in ssgsea_df
    res: str = 'hires', colour: str = '', agg: str = 'mean', title: str = '', save_path: str = '.'):

    # --- Get image + coordinates - setup ---
    library_id = list(adata.uns['spatial'].keys())[0]
    image = adata.uns['spatial'][library_id]['images'][res] # res is lowres or hires
    spatial_coords = adata.obsm['spatial']
    sf = adata.uns['spatial'][library_id]['scalefactors'][f'tissue_{res}_scalef']
    coords = spatial_coords * sf

    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # --- Align data ---
    ssgsea_df.index = adata.obs.index
    df = ssgsea_df.loc[adata.obs_names].copy()
    df['cluster'] = adata.obs[colour].values

    # --- Aggregate per cluster ---
    if agg == 'median':
        cluster_scores = df.groupby('cluster')[gene_set].median()
    else:
        cluster_scores = df.groupby('cluster')[gene_set].mean()

    # --- Map back to spots ---
    spot_scores = adata.obs[colour].map(cluster_scores)

    # --- Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=200)

    ax.imshow(image)

    sc = ax.scatter(coords[:, 0],  coords[:, 1], c=spot_scores, cmap='bwr', s=2, edgecolor='none')

    ax.set_title(title if title else f"{gene_set}\n({agg} NES per cluster)")
    ax.axis("off")

    # Colorbar
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Score")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()   


# plot the NES scores on the tissue
def ssgsea_scores(adata, ssgsea_df, gene_set,  # column name in ssgsea_df
    res: str = 'hires', cmap: str = 'bwr', save_path: str = '.'):

    library_id = list(adata.uns['spatial'].keys())[0]

    # --- Get image + coordinates ---
    images = adata.uns['spatial'][library_id]['images']
    spatial_coords = adata.obsm['spatial']

    if res == 'hires':
        image = images['hires']
        sf = adata.uns['spatial'][library_id]['scalefactors']['tissue_hires_scalef']
    else:
        image = images['lowres']
        sf = adata.uns['spatial'][library_id]['scalefactors']['tissue_lowres_scalef']

    coords = spatial_coords * sf

    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    # --- Align data ---
    ssgsea_df.index = adata.obs.index
    df = ssgsea_df.loc[adata.obs_names].copy()
        
    spot_scores_raw = df[gene_set]
    
    # --- Shared color normalization ---
    vmin = min(spot_scores_raw)
    vmax = max(spot_scores_raw)
    vcenter = ((vmax - vmin) / 2) + vmin    
    
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    
    # --- Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=200)

    ax.imshow(image)

    sc = ax.scatter(coords[:, 0], coords[:, 1], c=spot_scores_raw, cmap=cmap, norm=norm, s=2, edgecolor='none')
    ax.set_title(f"{gene_set}")
    ax.axis("off")

    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("NES")                   
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()  


# plots the cluster back to the tissue. 
def ssgsea_clusters(adata, gene_set, res: str='hires', colour: str = 'multires_clst', k: int = 20, alpha: float=1.0,
                    title: str = '', save_path: str = '.'):
    
    # --- Get image + coordinates - setup ---
    library_id = list(adata.uns['spatial'].keys())[0]
    image = adata.uns['spatial'][library_id]['images'][res] # res is lowres or hires
    spatial_coords = adata.obsm['spatial']
    sf = adata.uns['spatial'][library_id]['scalefactors'][f'tissue_{res}_scalef']
    coords = spatial_coords * sf

    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    complete_palette = ['#1f77b4', '#ff7f0e', '#279e68', '#d62728', '#aa40fc', '#8c564b', '#e377c2', '#b5bd61', '#17becf', '#aec7e8',
                 '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#dbdb8d', '#9edae5', '#ad494a', '#8c6d31',
                '#7f7f7f', '#c7c7c7', '#2ca02c', '#9467bd', '#8c9cff', '#00d9ff', '#a0ff87', '#ff6b9c', '#ceb301',  '#4d66e8', '#00cc99']

    custom_colors = complete_palette[:k]
    custom_cmap = ListedColormap(custom_colors)

        # categorical colormap
    #cmap = cm.get_cmap("tab20", n_clusters)
    # cluster_colors = {cl: custom_cmap(i) for i, cl in enumerate(unique_clusters)}

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=200)
    ax.imshow(image)
    sq.pl.spatial_scatter(adata, color=colour, img_res_key=res, size=1.2, cmap=custom_cmap, alpha=alpha, ax=ax, dpi=200)
    ax.set_title(gene_set)
    ax.axis("off")

    fig.suptitle(title)
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path, dpi=200)  # Save before showing
    plt.show() 



def show_results(ssgsea_df, gene_set, results_adata_dir: str = "multimodal_results",
                    output_dir: str = "multimodal_results/clustering_plots",):
    """
    Visualize clustering results for all analysis output from clustering.cluster().    
    The function works with saved AnnData objects in the results directory.
    Parameters
    ----------
    results_dir : str
        Directory containing subdirectories with the multimodal results
    output_dir : str
        Directory to save clustering visualizations
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results_adata_path = Path(results_adata_dir)
    
    # Find all AnnData files and corresponding cluster files
    adata_files = list(results_adata_path.glob("adata_*.h5ad"))
    
    print(f"Found {len(adata_files)} AnnData files")
    
    for adata_file in adata_files:
        adata = sc.read_h5ad(adata_file)
        base_name = adata_file.stem.replace("adata_", "")           
        params = parse_parameters_from_name(base_name)   # Parse parameters from base_name
        print(params)
    
        cluster_cols = []
        k_values = []
        
        for col in adata.obs.columns:
            if 'clusters_' in col:
                _, k = col.split(sep='_')
                k = int(k)
                k_values.append(k)
                cluster_cols.append(col)
                print(f'{base_name}, k={k}')
                score_per_cluster(adata, ssgsea_df, gene_set, colour=col, save_path=output_path / f"cluster_NES_{base_name}_k={k}.png")  # plot the mean score per cluster
                ssgsea_clusters(adata, gene_set, colour=col, k=k, # title=f'σ={params['sigma']}, k={k}, weights={params['weights']}', 
                                save_path=output_path / f"clusters_{base_name}_k={k}.png")

    #print('Original ssGSEA NES scores mapped to tissue.')
    #ssgsea_scores(adata, ssgsea_df, gene_set, colour=col, save_path=f"ssgsea_NES_{gene_set}.png")
                
    print(f"Clustering visualization complete! Results saved to: {output_path}")


def parse_parameters_from_name(base_name: str) -> dict:
    """ Extract parameters from filename base name."""
    params = {}    
    # Extract sigma (e.g., "s0" -> 0)
    sigma_part = base_name.split('_')[0]
    params['sigma'] = int(list(sigma_part)[-1])  # Remove 's' prefix
    
    # Extract modality (rest of the name)
    modality_parts = base_name.split('_')[2:]
    
    weights_parts = [part for part in modality_parts if part.startswith('weights')]
    if weights_parts:
        ind = modality_parts.index(*weights_parts)
        # Extract weights values
        params['weights'] = [float(modality_parts[ind+1]), float(modality_parts[ind+2])]
    
    # Reconstruct modality name
    params['modality'] = '_'.join(modality_parts)
    
    return params