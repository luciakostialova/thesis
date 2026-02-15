import matplotlib.pyplot as plt
import numpy as np
import anndata as ad
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform
from typing import Optional, Tuple, Literal

def _get_geneexpr_distance(adata: ad.AnnData, weights: Tuple[float, float] | None = None, 
                           distance: Optional[Literal["euclidean"]] = None, pca: bool = True):
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
    
    FIX: add options for correlation. 
    """
    
    scaler = MinMaxScaler()

    if pca:
        dist = pdist(adata.obsm['X_pca'], metric=distance)

    else:
        dist = pdist(adata.X.toarray(), metric=distance)
    
    
    geneexpr_dist = squareform(dist)
    geneexpr_dist_scaled = scaler.fit_transform(geneexpr_dist)  # Scale the distance matrix
    
    # Store in adata
    adata.obsp['geneexpr_dist'] = geneexpr_dist
    adata.obsp['geneexpr_dist_scaled'] = geneexpr_dist_scaled

    if weights is not None:
        geneexpr_dist_scaled *= weights[1]
    
    return geneexpr_dist_scaled





# geneexpr_corr = np.corrcoef(adata.X.toarray())
# geneexpr_dist = 1 - geneexpr_corr
# geneexpr_dist_scaled = scaler.fit_transform(geneexpr_dist)
# adata.obsp['geneexpr_correlation'] = geneexpr_corr