import pandas as pd
import anndata as ad
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, Literal

''' The distance matrices derived from gene expression levels and gene set enrichment scores are computed.
The distance matrices are saved in adata. The weights are not applied at this stage.'''

def _get_geneset_distance(adata: ad.AnnData, ssgsea_df: pd.DataFrame, distance: Optional[Literal["euclidean"]] = None):
    ''' The (Euclidean) distance is applied to the gene set normalised enrichment scores.
    The distance matrix is saved to adata. '''
    scaler = MinMaxScaler()

    ssgsea_dist = pdist(ssgsea_df.to_numpy().reshape(-1,1), metric=distance)  # for 1 set / variable
    ssgsea_dist = squareform(ssgsea_dist)
    ssgsea_dist_scaled = scaler.fit_transform(ssgsea_dist)
           
    adata.obsp[f'ssgsea_{distance}_dist'] = ssgsea_dist
    adata.obsp[f'ssgsea_{distance}_dist_scaled'] = ssgsea_dist_scaled

    return ssgsea_dist_scaled, ssgsea_dist 



def _get_geneexpr_distance(adata: ad.AnnData, distance: Optional[Literal["euclidean"]] = None, pca: bool = True):
    """
    Distance matrix for gene expression level data (original feature matrix) is calculated. 
    The user can choose whether to use dimensionality reduction (PCA) and which distance metric to use.
    Possible distance metrics include all as in scipy.spatial.distance.pdist(). Default is euclidean.
    The distance matrix is scaled to fit the interval [0,1].
    The distance matrices are saved in adata.
    Parameters
    ----------
    adata : AnnData
    weights: Tuple[float, float] | None = None
        weights used for weighing the distances
    distance: Optional[Literal["euclidean"]] = None
        distance used to compute gene expression distance metric.
    pca: bool = True
        compute on principal components    
    """
    
    scaler = MinMaxScaler()

    if pca:
        dist = pdist(adata.obsm['X_pca'], metric=distance)

    else:
        dist = pdist(adata.X.toarray(), metric=distance)
    
    
    geneexpr_dist = squareform(dist)
    geneexpr_dist_scaled = scaler.fit_transform(geneexpr_dist)  # Scale the distance matrix
    
    adata.obsp[f'geneexpr_{distance}_dist'] = geneexpr_dist
    adata.obsp[f'geneexpr_{distance}_dist_scaled'] = geneexpr_dist_scaled

    return geneexpr_dist_scaled, geneexpr_dist
