import pandas as pd
import numpy as np
from typing import Optional, Literal
import anndata as ad
from pathlib import Path
from typing import Optional, Tuple, Dict, Literal, List

import image_segmentation
import distances
import clustering

def multimodal_distance(    
    adata_img_segmentation: ad.AnnData,
    adata_gene_expr: ad.AnnData,
    ssgsea_df: pd.DataFrame = None,
    modality_combinations: Tuple[bool, bool] = None, 
    distance_gexpr: Optional[Literal["correlation"]] = None,
    use_pca: bool = True,
    distance_gset: Optional[Literal["euclidean"]] = None,
    superpixel: str = 'slic',
    sigmas: List[int] = [0, 1, 2, 3], 
    weight_combinations: List[Tuple[float, float]]  | None = None,
    cluster: bool = True,
    linkage: Optional[Literal["complete"]] = None,
    k: List[int] = [8, 16, 20, 32],
    output_dir: str = "multimodal_results"
    ):

    '''
    Run multimodal analysis combining image segmentation, gene expression, and gene set enrichment.
    
    Parameters
    ----------
    adata_img_segmentation : AnnData
        Segmented image data with distance matrices stored in .obsp
    adata_gene_expr : AnnData, optional
        Gene expression data (required if geneexpr_level or geneset_level is True)
    ssgsea_df : pd.DataFrame, optional
        Gene set enrichment analysis scores (required if geneset_level is True). Currently only one gene set is supported.
    gene_set : str, optional
        Gene set name/identifier. Currently only enrichment scores for one gene set are supported.
    modality_combinations : Tuple[bool, bool]
        Tuple of (geneexpr_level, geneset_level) indicating which modalities to include.
        Examples:
            (True, False)   - segmentation + gene expression
            (False, True)   - segmentation + gene sets
            (True, True)    - all three modalities
    distance_gexpr : str, optional
        Distance metric for gene expression ('correlation' is default)
    use_pca : bool, default True
        Whether to use PCA for gene expression distance calculation
    distance_gset : str, optional
        Distance metric for gene sets ('euclidean' is default)
    superpixel : str, default 'slic'
        Superpixel segmentation method. Methods 'slic' and 'seed' are available.
    sigmas : List[int], default [0, 1, 2, 3]
        List of sigma values used in previous segmentation
    weight_combinations : List[Tuple[float, float]], optional
        List of weight tuples (seg_weight, modality_weight) for combining distances.
        If None, defaults to [(1, 1)] - no weighting. If all 3 modalities are used - no weighting is applied default.
    cluster : bool, default True
        Whether to perform hierarchical clustering after distance calculation
    linkage : str, optional
        Linkage method for clustering ('complete' is default)
    k : List[int], default [8, 16, 20, 32]
        List of cluster numbers k to compute if cluster=True
    output_dir : str, default "multimodal_results"
        Directory path to save results
    '''

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    if weight_combinations is None:
        weight_combinations = [(1,1)]

    adata_gexpr = adata_gene_expr.copy()

    if modality_combinations is None:
        raise ValueError("modality_combinations must be provided")

    geneexpr_level, geneset_level = modality_combinations

    if geneset_level:
        ssgsea_dist_scaled, ssgsea_dist = distances._get_geneset_distance(adata_gexpr, ssgsea_df, distance_gset)

    if geneexpr_level:
        geneexpr_dist_scaled, geneexpr_dist = distances._get_geneexpr_distance(adata_gexpr, distance_gexpr, use_pca)
          
    for sigma in sigmas:

        sigma_dir = Path(f'{output_dir}/s{sigma}_output')
        sigma_dir.mkdir(exist_ok=True)

        segmentation_key = f'segmentation_{superpixel}_sigma{sigma}_dist_scaled'

        segmentation_distance = adata_img_segmentation.obsp[segmentation_key]

        for weight in weight_combinations:
            processed_adata = adata_gene_expr.copy()

            if geneexpr_level and not geneset_level:
                distance_matrix = segmentation_distance * weight[0] + geneexpr_dist_scaled * weight[1]

            elif geneset_level and not geneexpr_level:
                distance_matrix = segmentation_distance * weight[0] + ssgsea_dist_scaled * weight[1]

            elif geneexpr_level and geneset_level:
                distance_matrix = segmentation_distance + geneexpr_dist_scaled + ssgsea_dist_scaled

            else:
                distance_matrix = segmentation_distance
                
            processed_adata.obsp['multimodal_distance'] = distance_matrix

            if cluster:
                for k_val in k:
                    clustering.cluster(processed_adata, linkage=linkage, k=k_val)
            
            modality_name = get_modality_name(geneexpr_level, geneset_level, weight)
            base_name = f"s{sigma}_{modality_name}"
            adata_path = sigma_dir / f"adata_{base_name}.h5ad"
            processed_adata.write(adata_path)

     
def get_modality_name(geneexpr_level: bool, 
                      geneset_level: bool, 
                      weights: Optional[Tuple[float, float]] = None) -> str:
    """ Generate descriptive modality name for filename. """
    
    base_name = ""
    
    if geneexpr_level and geneset_level:
        base_name = "all_modalities"
    elif geneexpr_level:
        base_name = "segmentation_gene_expression"
    elif geneset_level:
        base_name = "segmentation_gene_sets"
    else:
        base_name = "segmentation_only"
    
    # Add weight information only if weights are explicitly provided
    if weights is not None:
        w1, w2 = weights
        weight_suffix = f"_weights_{w1}_{w2}"
    else:
        weight_suffix = ""
    
    return base_name + weight_suffix   

def run_img_segmentation(
    adata: ad.AnnData,
    output_dir: str = "segmentation_results",
    sigmas: List[int] = [0, 1, 2, 3],
    superpixel: str = "slic",
    res: str = "hires"
    ) -> ad.AnnData:
    ''' Run image segmentation for the given adata and save the segmentation labels and distances to adata.'''

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    adata_copy = adata.copy()  # save all the segmentations and distances to adata_copy

    for sigma in sigmas:
        print(f"Running superpixel segmentation for sigma={sigma}.")
        
        coords, image = setup_segmentation(adata=adata_copy, res=res)

        image_segmentation.perform_segmentation(adata_copy, image, coords, sigma, superpixel)

        image_segmentation.get_segmentation_distance(adata=adata_copy, coords=coords, sigma=sigma, superpixel=superpixel)

        print(f"Segmentation distance saved to adata.")

    save_path = output_path / f"adata_img_segmentation_{superpixel}.h5ad"
    adata_copy.write_h5ad(save_path)

    return adata_copy


def setup_segmentation(adata: ad.AnnData, res: str = 'hires'):
    ''' Setup for the analysis: Image data, coordinates, scalefactors are loaded.'''
    
    # images
    library_id = list(adata.uns['spatial'].keys())[0]
    images = adata.uns['spatial'][library_id]['images']

    # scalefactors
    hires_sf = adata.uns['spatial'][library_id]['scalefactors']['tissue_hires_scalef']
    lowres_sf = adata.uns['spatial'][library_id]['scalefactors']['tissue_lowres_scalef']

    spatial_coords = adata.obsm['spatial']
    
    if res == 'hires':
        image = images['hires']
        coords = spatial_coords * hires_sf
    else:
        image = images['lowres']
        coords = spatial_coords * lowres_sf

    return coords, image