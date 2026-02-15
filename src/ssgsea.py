''' 
ssGSEA input is prepared, the distance is computed, some statistics are visualised. 
'''### Returns ssGSEA distance, prepares input for ssgsea, reutrns some statistics
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import squidpy as sq
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler


def prepare_ssGSEA_input(adata: ad.AnnData, filename: str='', preprocess: bool = True):
    ''' The function preprocesses the gene count matrix, and prepares a .gct file for ssGSEA to use in R. 
    Only the filename is necessary to add. The filename automatically gains the .gct extension. '''
    
    gene_ids = adata.var.index
    barcodes = adata.obs.index

    if preprocess:
        sc.pp.normalize_total(adata, inplace=True)  # Normalize each cell by total counts over all genes, so that every cell has the same total count after normalization
        sc.pp.log1p(adata) # Logarithmize the data matrix
        sc.pp.scale(adata, max_value=10)
        # sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)

    df = pd.DataFrame(adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X)
    df.columns = gene_ids
    df.index = barcodes
    df = df.T
    df.insert(loc=0, column='Description', value=gene_ids)

    print(f'ssGSEA input file saved as {filename}.gct.')
    df.to_csv(filename + '.gct', sep='\t')

def _get_geneset_distance(adata, geneset_path, weights):
    ''' The Euclidean distance is applied to the gene set normalised enrichment scores.
    The distance matrix is saved to adata. '''
    scaler = MinMaxScaler()
    
    # load ssgsea results from R 
    ssgsea_out = pd.read_csv(Path(geneset_path), sep=',', index_col=0)
    ssgsea_out = ssgsea_out.T
    ssgsea_out.index = ssgsea_out.index.str.replace(".", "-", regex=False)

    # check if shape fits
    #if adata.obs.index == ssgsea_out.index:
     #   pass

    ssgsea = ad.AnnData(ssgsea_out)
    ssgsea.obs_names = ssgsea_out.index
    ssgsea.var_names = ssgsea_out.columns
    ssgsea.obsm['spatial'] = adata.obsm['spatial']
    ssgsea.uns['spatial'] = adata.uns['spatial']

    # check if shape fits
    #if ssgsea.X.shape == (adata.X.shape[0], 50):
     #   pass # should be #spots x # H gene sets

    # compute Euclidean distance
    ssgsea_dist = pdist(ssgsea.X, 'cosine')
    ssgsea_dist = squareform(ssgsea_dist)
    ssgsea_dist_scaled = scaler.fit_transform(ssgsea_dist)
           
    adata.obsp['ssgsea_dist'] = ssgsea_dist
    adata.obsp['ssgsea_dist_scaled'] = ssgsea_dist_scaled

    if weights is not None:
        ssgsea_dist_scaled *= weights[1]
        
    return ssgsea_dist_scaled, ssgsea 


def ssgsea_output_statistics(ssgsea: ad.AnnData):
    ''' Leiden clustering of the NES is plotted. Distribution of the ssGSEA NES is plotted. '''
    sc.pp.pca(ssgsea)
    sc.pp.neighbors(ssgsea)
    sc.tl.umap(ssgsea)
    sc.tl.leiden(ssgsea)

    sq.pl.spatial_scatter(ssgsea, color="leiden", size=40, shape=None)
    # plt.savefig('leiden.png', dpi=300)

    print(f'Minimum and maximum ssGSEA Normalized Enrichment Scores: min = {np.min(ssgsea.X.flatten())} max = {np.max(ssgsea.X.flatten())}')
    
    fig, ax = plt.subplots(1,2, figsize=(10, 5))
    
    # Histogram 1: NES values
    ax[0].hist(ssgsea.X.flatten(), bins=80, alpha=0.7, color='skyblue', edgecolor='black')
    ax[0].set_xlabel('Distances')
    ax[0].set_ylabel('Frequency')
    ax[0].set_title('Distribution of ssGSEA Normalized Enrichment Scores')
    ax[0].grid(True, alpha=0.3)
    
    ax[1].hist(ssgsea.X.flatten(), bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    ax[1].set_xlabel('Distances')
    ax[1].set_ylabel('Frequency')
    ax[1].set_title('Distribution of ssGSEA Normalized Enrichment Scores')
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
