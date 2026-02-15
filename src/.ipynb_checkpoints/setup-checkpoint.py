import numpy as np
import json
import anndata as ad
from pathlib import Path
from typing import Optional, Tuple, Dict, Literal, List
import matplotlib.pyplot as plt
import scanpy as sc

import geneexpr_distance
import image_segmentation
import clustering
from plot_stat import plot_statistics, plot_distance_statistics
from ssgsea import prepare_ssGSEA_input, _get_geneset_distance, ssgsea_output_statistics



def run_multimodal_pipeline(
    adata: ad.AnnData,
    output_dir: str = "multimodal_results",
    superpixel: Optional[Literal["slic", "seed"]] = None,
    sigmas: List[int] = [0, 1, 2, 3],
    distance: Optional[Literal["euclidean"]] = None,
    pca: bool = True,
    modality_combinations: List[Tuple[bool, bool]] = None,
    weight_combinations: List[Tuple[float, float]]  | None = None,
    library_id: str = None,
    res: str = 'hires',
    geneset_path: Optional[Path] = None,
    preprocess: bool = True,
    cluster: bool = True,
    linkage: Optional[Literal["complete"]] = None, 
    k: int = None
):
    '''
    Run comprehensive multimodal analysis using `multimodal_distance`.
    
    Parameters
    ----------
    adata : AnnData
        Spatial transcriptomics data
    output_dir : str
        Directory to save results
    sigmas : List[int]
        Sigma values for smoothing
    modality_combinations : List[Tuple[bool, bool]]
        List of (geneexpr_level, geneset_level) combinations
         modality_combinations = [
            (True, False),   # segmentation + gene expression
            (False, True),   # segmentation + gene sets
            (True, True)     # all three
        ]
    weights : List[(0.5, 0.5), (0.25, 0.75), (0.75, 0.25)]  | None = None
        List of weights for distance matrix, optional. Can be set for just 2 modalities.
    library_id : str
        Library identifier
    res : str
        Resolution ('hires' or 'lowres')
    geneset_path : Path
        Path to gene set enrichment scores
    '''
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Set default library_id if not provided
    if library_id is None:
        library_id = list(adata.uns['spatial'].keys())[0]
    
    # Default modality combinations if none are set
    if modality_combinations is None:
        modality_combinations = [
            (True, False),   # segmentation + gene expression
            (False, True),   # segmentation + gene sets
            (True, True)     # all three
        ]

    # Default weight combinations - if None, no weights will be applied
    if weight_combinations is None:
        weights = None
    
    results = {}
    
    # Cache for segmentation distance matrices to avoid recomputation
    segmentation_cache = {}

    # for all resolution levels and modality levels:
    for sigma in sigmas:
        
        sigma_dir = Path(f'{output_dir}/s{sigma}_output')
        sigma_dir.mkdir(exist_ok=True)
        
        for i, (geneexpr_level, geneset_level) in enumerate(modality_combinations):
            # Determine if we should use weights
            # Only apply weights when we have exactly one omics modality + segmentation AND weight_combinations is provided 
            should_use_weights = (geneexpr_level != geneset_level) and weight_combinations is not None
            
            if should_use_weights:
                weights_list = weight_combinations
                weight_names = [f"w{w1}_{w2}" for w1, w2 in weight_combinations]
            else:
                weights_list = [None]
                weight_names = ["no_weights"]
            
            for weights, weight_name in zip(weights_list, weight_names):
                modality_name = get_modality_name(geneexpr_level, geneset_level, weights)                
                print(f"Processing: sigma={sigma}, {modality_name}, weights={weights}")
            
                try:
                    # Calculate multimodal distance with segmentation caching
                    result = multimodal_distance(
                        adata=adata.copy(),  # Work on copy to avoid conflicts
                        weights=weights,
                        superpixel=superpixel,
                        geneexpr_level=geneexpr_level,
                        geneset_level=geneset_level,
                        geneset_path=geneset_path,
                        library_id=library_id,
                        res=res,
                        sigma=sigma,
                        segmentation_cache=segmentation_cache,  # Pass the cache
                        preprocess = preprocess
                    )
                    
                    processed_adata, ssgsea = result

                    if cluster:
                        clustering.cluster(processed_adata, linkage='complete', k=k)

                    # Generate file paths
                    base_name = f"s{sigma}_{modality_name}"
                    # fig_path = output_path / f"boundaries_{base_name}.png"
                    # distance_path = output_path / f"distance_{base_name}.npy"
                    distance_csv = sigma_dir / f"distance_{base_name}.csv"
                    metadata_path = sigma_dir / f"metadata_{base_name}.json"
                    adata_path = sigma_dir / f"adata_{base_name}.h5ad"
                                   
                    # Save distance matrix
                    multimodal_dist = processed_adata.obsp['multimodal_distance']
                    # np.save(distance_path, multimodal_dist)
                    np.savetxt(distance_csv, multimodal_dist, delimiter="\t")
                    
                    # Save the AnnData object
                    processed_adata.write(adata_path)
                    
                    # Save metadata
                    metadata = {
                        'sigma': sigma,
                        'modality': modality_name,
                        'geneexpr_level': geneexpr_level,
                        'geneset_level': geneset_level,
                        'weights': weights,
                        'library_id': library_id,
                        'resolution': res,
                        'timestamp': np.datetime64('now').astype(str),
                        'distance_matrix_shape': multimodal_dist.shape,
                        'adata_shape': f"{processed_adata.n_obs} x {processed_adata.n_vars}",
                        'files': {
                            # 'figure': str(fig_path.name),
                            # 'distance_matrix': str(distance_path.name),
                            'distance_csv': str(distance_csv.name),
                            'anndata': str(adata_path.name)
                        }
                    }
                    
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    # Store results
                    key = f"sigma_{sigma}_{modality_name}"
                    results[key] = {
                        'adata': processed_adata,
                        'distance_matrix': multimodal_dist,
                        'metadata': metadata,
                        # 'figure_path': fig_path,
                        'adata_path': adata_path,
                        'files': metadata['files']
                    }
                    
                    print(f"Completed: sigma={sigma}, {modality_name}, weights={weights}")
                    print(f"Saved AnnData to: {adata_path}")
                    
                except Exception as e:
                    print(f"Failed: sigma={sigma}, {modality_name}, weights={weights}")
                    print(f"   Error: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    return results

def setup_analysis(adata, library_id, res, weights: Tuple[float, float] | None=None, preprocess: bool=True):
    ''' Setup for the analysis: Image data, coordinates, scalefactors are loaded. The gene count matrix is preprocessed. '''
    
    # images
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

    if weights is not None:
        adata.uns['spatial'][library_id]['distances'] = {}
        adata.uns['spatial'][library_id]['distances']['weights'] = list(weights)
        
    if preprocess:
        sc.pp.normalize_total(adata, inplace=True)  # Normalize each cell by total counts over all genes, so that every cell has the same total count after normalization
        sc.pp.log1p(adata) # Logarithmize the data matrix
        sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000) 
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        sc.tl.leiden(adata)

    return coords, image

def multimodal_distance(
    adata: ad.AnnData,
    weights: Tuple[float, float] | None = None,
    superpixel: Optional[Literal["slic", "seed"]] = None,
    distance: str = 'euclidean',
    pca: bool = True,
    geneexpr_level: bool = True,
    geneset_level: bool = False,
    geneset_path: Path | None = None,
    library_id: str | None = None,
    res: str = 'hires',
    sigma: int = 0,
    segmentation_cache: Dict = None,
    preprocess: bool = True
):
    """ 
    The function receives as input the AnnData object. 
    The output should be the multimodal distance matrix, containing 'morphology' and omics information.
    The user can choose whether to include gene expression, gene set or both kinds of information.
    In case of including gene set information, the user should include the GSEA Enrichment Scores given by R (GSVA).
    For gene expression, a correlation-based distance is used. For ES, Euclidean distance. For the morphology, 
    a segmentation algorithm is used: SLIC or SEED. The resulting distance matrices are scaled and added together. 
    Optionally, basic statistics concerning the distance and omics information is printed.
    
    Parameters
    ----------
    adata
        The Anndata object.
    weights
        Weights for combining distance matrices
    superpixel
        the segmentation algorithm to use for generating superpixels
    distance
        the distance metric to use for gene expression 
    pca: 
        whether to use PCs from gene expression
    geneexpr_level
        Whether to include gene expression distance
    geneset_level
        Whether to include gene set enrichment distance
    geneset_path
        The path to the Enrichment Score file.
    segmentation_cache: Dict = None,
        Dictionary to cache segmentation distance matrices by sigma
    """    
    
    # Initialize cache if not provided
    if segmentation_cache is None:
        segmentation_cache = {}
    
    # Create cache key for this sigma
    cache_key = f"sigma_{sigma}"
    
    # Check if we already have segmentation for this sigma
    if cache_key in segmentation_cache:
        print(f"Using cached segmentation for sigma={sigma}")
        cached_segmentation_data = segmentation_cache[cache_key]
        
        # Restore segmentation labels and distance matrix
        adata.uns['spatial'][library_id]['segmentation'] = cached_segmentation_data['segmentation_info']
        distance_matrix = cached_segmentation_data['segmentation_distance'].copy()
        
    else:
        # images
        print(f'Setting up the analysis...')
        coords, image = setup_analysis(adata, library_id, res, weights, preprocess)

        # segmentation
        print(f'Performing SLIC segmentation for sigma={sigma}.')
        image_segmentation._perform_segmentation(adata, image, coords, library_id, res, sigma, superpixel)

        # distance matrix for segmentation labeling
        print(f'Computing segmentation distance matrix for sigma={sigma}.')
        distance_matrix = image_segmentation._get_segmentation_distance(adata, coords, library_id, weights)
        
        # Cache the segmentation results for this sigma
        segmentation_cache[cache_key] = {
            'segmentation_info': adata.uns['spatial'][library_id]['segmentation'].copy(),
            'segmentation_distance': distance_matrix.copy()
        }
        print(f"Cached segmentation for sigma={sigma}")
    
    # gene expression level
    if geneexpr_level:
        print(f'Computing gene expression distance matrix.')
        geneexpr_dist = geneexpr_distance._get_geneexpr_distance(adata, weights, distance, pca)
        distance_matrix += geneexpr_dist

    # gene set level
    ssgsea = None
    if geneset_level:        
        print(f'Computing Enrichment Score distance matrix')
        ssgsea_dist_scaled, ssgsea = _get_geneset_distance(adata, geneset_path, weights)
        distance_matrix += ssgsea_dist_scaled        

    print(f'Multimodal distance is ready.')
    adata.obsp['multimodal_distance'] = distance_matrix

    return (adata, ssgsea)


def get_modality_name(geneexpr_level: bool, geneset_level: bool, weights: Optional[Tuple[float, float]] = None) -> str:
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
        weight_suffix = ""  # No weight suffix when no weights are applied
    
    return base_name + weight_suffix