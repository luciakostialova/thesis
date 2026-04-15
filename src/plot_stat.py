''' Plots distributions for modality values and distance matrices. '''
import numpy as np
import matplotlib.pyplot as plt

def plot_statistics(adata, library_id):
    ''' Plots gene expression values, gene expression correlation values for each spot, the unscaled correlation distance and unscaled segmentation distance. '''
    
    sigma = adata.uns['spatial'][library_id]['segmentation']['sigma']
    geneexpr_corr = adata.obsp['geneexpr_correlation']
    geneexpr_dist = adata.obsp['geneexpr_dist']
    segmentation_dist = adata.obsp['segmentation_dist']
    
    print(f'Minimum and maximum segmentation distances (sigma={sigma}): min = {np.min(segmentation_dist.flatten())} max = {np.max(segmentation_dist.flatten())}')
    print(f'Minimum and maximum gene expression values: min = {np.min(adata.X.toarray().flatten())} max = {np.max(adata.X.toarray().flatten())}')
    print(f'Minimum and maximum gene expression correlation values: min = {np.min(geneexpr_corr.flatten())} max = {np.max(geneexpr_corr.flatten())}')
    print(f'Minimum and maximum gene expression correlation distance values: min = {np.min(geneexpr_dist.flatten())} max = {np.max(geneexpr_dist.flatten())}')
    
    fig, ax = plt.subplots(2,2, figsize=(15, 10))
    
    ax[0,0].hist(adata.X.toarray().flatten(), bins=100, alpha=0.7, color='skyblue', edgecolor='black')
    ax[0,0].set_xlabel('Values')
    ax[0,0].set_ylabel('Frequency')
    ax[0,0].set_title('Distribution of Gene Expression Values')
    ax[0,0].grid(True, alpha=0.3)

    mask = np.triu(np.ones_like(geneexpr_corr, dtype=bool), k=1)
    correlation_values = geneexpr_corr[mask].flatten()
    
    ax[0,1].hist(correlation_values.flatten(), bins=100, alpha=0.7, color='lightcoral', edgecolor='black')
    ax[0,1].set_xlabel('Pearson Correlation Coefficient')
    ax[0,1].set_ylabel('Frequency')
    ax[0,1].set_title('Distribution of Gene Expression Correlation')
    ax[0,1].grid(True, alpha=0.3)
    
    ax[1,0].hist(geneexpr_dist.flatten(), bins=100, alpha=0.7, color='skyblue', edgecolor='black')
    ax[1,0].set_xlabel('Distance')
    ax[1,0].set_ylabel('Frequency')
    ax[1,0].set_title('Distribution of Gene Expression Correlation Distance')
    ax[1,0].grid(True, alpha=0.3)
    
    ax[1,1].hist(segmentation_dist.flatten(), bins=100, alpha=0.7, color='lightcoral', edgecolor='black')
    ax[1,1].set_xlabel('Distance')
    ax[1,1].set_ylabel('Frequency')
    ax[1,1].set_title(f'Distribution of Segmentation Distance Matrix, sigma={sigma}')
    ax[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_distance_statistics(adata, library_id, geneset_level: bool = False):
    ''' The scaled distance values distributions are plotted for all modalities. Some statisticsa are printed. '''
    geneexpr_dist_scaled = adata.obsp['geneexpr_dist_scaled']
    segmentation_dist_scaled = adata.obsp['segmentation_dist_scaled']
    distance_matrix = adata.obsp['multimodal_distance']    
    sigma = adata.uns['spatial'][library_id]['segmentation']['sigma']
    if geneset_level:
        ssgsea_dist_scaled = adata.obsp['ssgsea_dist_scaled']
    else:
        ssgsea_dist_scaled = np.zeros(adata.shape)

    print(f'All matrices are scaled in the range [0,1].')
    print(f'Minimum and maximum in gene expression distance matrix: min = {np.min(geneexpr_dist_scaled)} max = {np.max(geneexpr_dist_scaled)}')
    print(f'Minimum and maximum in superpixel distance matrix: min = {np.min(segmentation_dist_scaled)} max = {np.max(segmentation_dist_scaled)}')
    print(f'Minimum and maximum in ssGSEA distance matrix: min = {np.min(ssgsea_dist_scaled)} max = {np.max(ssgsea_dist_scaled)}')
    print(f'Statistics summary for multimodal distance:\n Maximum: {np.max(distance_matrix)} minimum: {np.min(distance_matrix)} average: {np.mean(distance_matrix)} median:  {np.median(distance_matrix)}')
    
    fig, ax = plt.subplots(2,2, figsize=(15, 10))
    
    ax[0,0].hist(geneexpr_dist_scaled.flatten(), bins=100, alpha=0.7, color='skyblue', edgecolor='black')
    ax[0,0].set_xlabel('Distance')
    ax[0,0].set_ylabel('Frequency')
    ax[0,0].set_title('Distribution of Gene Expression Distance Matrix')
    ax[0,0].grid(True, alpha=0.3)
    
    ax[0,1].hist(segmentation_dist_scaled.flatten(), bins=100, alpha=0.7, color='lightcoral', edgecolor='black')
    ax[0,1].set_xlabel('Distance')
    ax[0,1].set_ylabel('Frequency')
    ax[0,1].set_title('Distribution of Segmentation Distance Matrix')
    ax[0,1].grid(True, alpha=0.3)
    
    ax[1,0].hist(ssgsea_dist_scaled.flatten(), bins=100, alpha=0.7, color='skyblue', edgecolor='black')
    ax[1,0].set_xlabel('Distance')
    ax[1,0].set_ylabel('Frequency')
    ax[1,0].set_title('Distribution of ssGSEA Euclidean Distance Matrix')
    ax[1,0].grid(True, alpha=0.3)
    
    ax[1,1].hist(distance_matrix.flatten(), bins=100, alpha=0.7, color='lightcoral', edgecolor='black')
    ax[1,1].set_xlabel('Distance')
    ax[1,1].set_ylabel('Frequency')
    ax[1,1].set_title(f'Distribution of Multimodal Distance Matrix, sigma={sigma}')
    ax[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    