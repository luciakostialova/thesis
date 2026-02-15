""" Performs agglomerative hierarchical clustering on precomputed distance matrix. Visualises the cluster outputs for a directory. """ 

import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad
import plot_clusters
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering

def cluster(adata: ad.AnnData, metric: str = 'precomputed', linkage: str = 'complete', k: int = 20, x_pca: bool = False):
    """ 
    Performs agglomerative hierarchical clustering on the precomputed multimodal distance metric. 
    Ward linnkage cannot be used and may not be reasonable either way, as the multimodal distance is not in Euclidean space.
    Cluster labels are saved in adata. 
    Parameters
    ----------
    adata : AnnData
        ST data, should contain the multimodal distance.
    metric: str = 'precomputed'
        metric to use for clustering, default is 'precomputed'. Otherwise any metric can be used that is compatible with sklearn.cluster.
    linkage: str = 'complete'
        linkage for clustering, default is complete.
    k: int = 20
        number of clusters to compute
    x_pca: bool = False
        It is possible to cluster the principal components if computed. 
    """
    agc = AgglomerativeClustering(n_clusters=k, metric=metric, linkage=linkage)
    if x_pca:
        cluster_labels = agc.fit_predict(adata.obsm['X_pca'])
    else:
        cluster_labels = agc.fit_predict(adata.obsp['multimodal_distance'])

    fname = 'clusters_' + str(k)        
    adata.obs[fname] = cluster_labels
    adata.obs[fname] = adata.obs[fname].astype('category') 

def show_results(results_adata_dir: str = "multimodal_results",
                    output_dir: str = "multimodal_results/clustering_plots"):
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
    
        cluster_cols = []
        k_values = []
        
        for col in adata.obs.columns:
            if 'clusters_' in col:
                _, k = col.split(sep='_')
                k = int(k)
                k_values.append(k)
                cluster_cols.append(col)
                plot_clusters.plot_cluster_boundaries(adata, colour=col, k=k,
                    title=f"σ={params['sigma']}, {params['modality']}, k={k}",
                    save_path=output_path / f"boundaries_{base_name}_k={k}.png")
                plot_clusters.plot_clusters(adata, colour=col, k=k, save_path=output_path / f"clusters_{base_name}_k={k}.png")
                
    
    print(f"Clustering visualization complete! Results saved to: {output_path}")

def output(results_adata_dir: str = "multimodal_results",
                      results_clust_dir: str = "multimodal_results/clustering_results",
                      output_dir: str = "multimodal_results/clustering_plots"):
    """
    Visualize clustering results for all analysis output from R.    
    Parameters
    ----------
    results_dir : str
        Directory containing the multimodal results
    output_dir : str
        Directory to save clustering visualizations
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results_adata_path = Path(results_adata_dir)
    results_clust_path = Path(results_clust_dir)
    
    # Find all AnnData files and corresponding cluster files
    adata_files = list(results_adata_path.glob("adata_*.h5ad"))
    cluster_files = list(results_clust_path.glob("*_clusters_k*.csv"))
    
    print(f"Found {len(adata_files)} AnnData files and {len(cluster_files)} cluster files")
    
    # Group files by base name pattern
    analyses = {}
    
    for adata_file in adata_files:
        # Extract base name (e.g., "s0_segmentation_gene_expression" from "adata_s0_segmentation_gene_expression.h5ad")
        base_name = adata_file.stem.replace("adata_", "")
        
        # Find matching cluster files
        matching_clusters = []
        for cluster_file in cluster_files:
            # Extract k value and base pattern from cluster filename
            # Expected pattern: "distance_s0_segmentation_gene_expression_clusters_k5.csv"
            if f"distance_{base_name}__clusters_k" in cluster_file.name:
                # Extract k value
                k_value = int(cluster_file.stem.split("_k")[-1])
                matching_clusters.append((k_value, cluster_file))
        
        if matching_clusters:
            analyses[base_name] = {
                'adata_path': adata_file,
                'clusters': matching_clusters
            }
    
    print(f"Found {len(analyses)} analyses with cluster results")
    
    if not analyses:
        print("No analyses with cluster results found!")
        return
    
    # Process each analysis
    all_results = []
    
    for base_name, analysis_info in analyses.items():
        print(f"Processing: {base_name}")
        
        try:
            # Load AnnData
            adata = sc.read_h5ad(analysis_info['adata_path'])
            
            # Parse parameters from base_name
            params = parse_parameters_from_name(base_name)
            
            # Create individual plots for each k value
            for k_value, cluster_file in analysis_info['clusters']:
                # Load cluster labels
                clusters_df = pd.read_csv(cluster_file, index_col=0)
                clusters_df = clusters_df.reset_index(drop=True)
                cluster_col = f"clusters_k{k_value}"
                
                # Add clusters to AnnData
                adata.obs[cluster_col] = clusters_df['clusters'].values #.astype('category')
                
                # Create individual plot
                fig = plot_clusters.plot_cluster_boundaries(
                    adata, 
                    colour=cluster_col,
                    k=k_value,
                    title=f"σ={params['sigma']}, {params['modality']}, k={k_value}",
                    save_path=output_path / f"clusters_{base_name}_k{k_value}.png"
                )
                
                # Store results for combined plot
                all_results.append({
                    'adata': adata.copy(),
                    'base_name': base_name,
                    'sigma': params['sigma'],
                    'modality': params['modality'],
                    'k': k_value,
                    'cluster_col': cluster_col,
                    'weights': params.get('weights', None)
                })
                
                print(f"  - Created plot for k={k_value}")
                
        except Exception as e:
            print(f"Error processing {base_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create combined comparison plot
    if all_results:
        pass
        # create_combined_comparison_plot(all_results, output_path)
        # create_modality_comparison_plots(all_results, output_path)
    
    print(f"Clustering visualization complete! Results saved to: {output_path}")


def parse_parameters_from_name(base_name: str) -> dict:
    """ Extract parameters from filename base name."""
    params = {}    
    # Extract sigma (e.g., "s0" -> 0)
    sigma_part = base_name.split('_')[0]
    params['sigma'] = int(list(sigma_part)[-1])  # Remove 's' prefix
    
    # Extract modality (rest of the name)
    modality_parts = base_name.split('_')[1:]
    
    weights_parts = [part for part in modality_parts if part.startswith('weights')]
    if weights_parts:
        ind = modality_parts.index(*weights_parts)
        # Extract weights values
        params['weights'] = [float(modality_parts[ind+1]), float(modality_parts[ind+2])]
    
    # Reconstruct modality name
    params['modality'] = '_'.join(modality_parts)
    
    return params

def create_combined_comparison_plot(all_results: list, output_path: Path):
    """Create a large comparison plot with all analyses."""
    n_results = len(all_results)
    if n_results == 0:
        return
    
    # Group by k value for better organization
    k_values = sorted(set([result['k'] for result in all_results]))
    
    # Create subplot grid: rows = k values, columns = analyses per k
    analyses_by_k = {}
    for k in k_values:
        analyses_by_k[k] = [r for r in all_results if r['k'] == k]
    
    n_rows = len(k_values)
    n_cols = max(len(analyses_by_k[k]) for k in k_values)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, k in enumerate(k_values):
        analyses_for_k = analyses_by_k[k]
        for j, result in enumerate(analyses_for_k):
            if j < n_cols:
                ax = axes[i, j] if n_rows > 1 else axes[j]
                
                # Get coordinates and clusters
                adata = result['adata']
                if 'spatial' in adata.obsm:
                    coords = adata.obsm['spatial']
                else:
                    coords = adata.obs[['array_col', 'array_row']].values
                
                clusters = adata.obs[result['cluster_col']].cat.codes.values
                
                # Create scatter plot
                scatter = ax.scatter(coords[:, 0], coords[:, 1], 
                                   c=clusters, cmap='tab20', s=5, alpha=0.8)
                
                # Create title
                title = f"σ={result['sigma']} {result['modality']}"
                if result['weights']:
                    title += f"\nw={result['weights']}"
                ax.set_title(title, fontsize=8)
                ax.axis('equal')
                ax.axis('off')
    
    # Remove empty subplots
    for i in range(n_rows):
        for j in range(len(analyses_by_k[k_values[i]]), n_cols):
            if n_rows > 1:
                axes[i, j].axis('off')
            else:
                axes[j].axis('off')
    
    plt.suptitle(f"Clustering Comparison (k values: {', '.join(map(str, k_values))})", 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / "clustering_comparison_all.png", dpi=300)
    plt.close()

def create_modality_comparison_plots(all_results: list, output_path: Path):
    """Create separate comparison plots for each modality."""
    modalities = sorted(set([result['modality'] for result in all_results]))
    k_values = sorted(set([result['k'] for result in all_results]))
    
    for modality in modalities:
        modality_results = [r for r in all_results if r['modality'] == modality]
        
        if not modality_results:
            continue
        
        # Group by k value
        n_rows = len(k_values)
        sigmas = sorted(set([r['sigma'] for r in modality_results]))
        n_cols = len(sigmas)
        
        if n_cols == 0:
            continue
            
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, k in enumerate(k_values):
            for j, sigma in enumerate(sigmas):
                # Find result for this k and sigma
                result = None
                for r in modality_results:
                    if r['k'] == k and r['sigma'] == sigma:
                        result = r
                        break
                
                ax = axes[i, j] if n_rows > 1 else axes[j]
                
                if result:
                    adata = result['adata']
                    if 'spatial' in adata.obsm:
                        coords = adata.obsm['spatial']
                    else:
                        coords = adata.obs[['array_col', 'array_row']].values
                    
                    clusters = adata.obs[result['cluster_col']].cat.codes.values
                    
                    scatter = ax.scatter(coords[:, 0], coords[:, 1], 
                                       c=clusters, cmap='tab20', s=5, alpha=0.8)
                    
                    title = f"k={k}, σ={sigma}"
                    if result['weights']:
                        title += f"\nw={result['weights']}"
                    ax.set_title(title, fontsize=8)
                else:
                    ax.text(0.5, 0.5, "Not available", 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"k={k}, σ={sigma}", fontsize=8)
                
                ax.axis('equal')
                ax.axis('off')
        
        plt.suptitle(f"Modality: {modality}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / f"clustering_comparison_{modality}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()