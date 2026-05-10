import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgb
from typing import List
import squidpy as sq
from skimage import io, segmentation
from skimage.draw import disk
import skimage
from scipy import ndimage as ndi

""" Plots the clusters and cluster boundaries on the tissue. The images are saved.""" 

def thesis_clusters(adata: ad.AnnData,
                    res: str='hires',
                    colour: str = 'multires_clst',
                    k: int = 20,
                    alpha: float=0.0,
                    lw: float=0.9,
                    title: str = '', 
                    save_path: str = '.'):

    ''' Main function for plotting cluster visualisations for thesis. 
    Parameters
    ----------
    adata : AnnData
        ST data, should contain the clusters labels
    colour: str = 'multires_clst'
        cluster label found in adata.obs.columns
    k: int = 20
        Number of clusters, default is 20
    alpha : float, default=0.0
        Transparency level
    lw : float, default=0.9
        Line width of spatial spot boundarie
    title: str, default=''
        Title displayed above the plot
    save_path: str = '.'
        Path to save image
    '''
    library_id = list(adata.uns['spatial'].keys())[0]

    # images
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

    height, width = image.shape[:2]
        
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]

    clusters = adata.obs[colour].astype("category").cat.codes.values

    spot_diameter = (adata.uns["spatial"][library_id]["scalefactors"]["spot_diameter_fullres"] * sf)
    radius = int(spot_diameter) * 1.1  # can be tweaked

    # 3. Rasterize into label image
    label_image = -1 * np.ones((height, width), dtype=int)

    for (x, y), cl in zip(coords, clusters):
        rr, cc = disk((y, x), radius, shape=label_image.shape)
        label_image[rr, cc] = cl
    
    # Optional: smooth small gaps between spots
    label_image = ndi.grey_closing(label_image, size=(3, 3))
    
    cluster_polygons = {}
    
    for cl in np.unique(clusters):
        mask = label_image == cl
        
        # find_contours expects float image
        contours = skimage.measure.find_contours(mask.astype(float), level=0.5)
    
        # Convert (row,col) → (x,y)
        polygons = [np.flip(contour, axis=1) for contour in contours]
    
        cluster_polygons[int(cl)] = polygons

    unique_clusters = sorted(cluster_polygons.keys())
    n_clusters = len(unique_clusters)

    complete_palette = ['#1f77b4', '#ff7f0e', '#279e68', '#d62728', '#aa40fc', '#8c564b', '#e377c2', '#b5bd61', '#17becf', '#aec7e8',
                 '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#dbdb8d', '#9edae5', '#ad494a', '#8c6d31',
                '#7f7f7f', '#c7c7c7', '#2ca02c', '#9467bd', '#8c9cff', '#00d9ff', '#a0ff87', '#ff6b9c', '#ceb301',  '#4d66e8', '#00cc99']

    custom_colors = complete_palette[:k]
    custom_cmap = ListedColormap(custom_colors)
    cluster_colors = {cl: custom_cmap(i) for i, cl in enumerate(unique_clusters)}

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=200)

    sq.pl.spatial_scatter(adata, color=colour, img_res_key='hires', size=1, cmap=custom_cmap, alpha=1, ax=axes[0], dpi=200)
    axes[0].set_title("Tissue Overlay")
    axes[0].axis("off")

    axes[1].imshow(image)
    axes[1].set_title("Cluster Boundaries")
    for cl, polygons in cluster_polygons.items():
        fill_color = cluster_colors[cl]

        for poly in polygons:
            axes[1].fill(poly[:, 0], poly[:, 1], color=fill_color, alpha=alpha, zorder=2)

            axes[1].plot(poly[:, 0], poly[:, 1], color="black", linewidth=lw, zorder=2)
    
    
    axes[1].axis("off")
    
    plt.tight_layout()
    fig.suptitle(title)
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path, dpi=200)  # Save before showing
    plt.show() 


def plot_clusters(adata, colour: str = 'multires_clst', k: int = 20, save_path: str = '.'):
    """ 
    Plots the clusters on the tissue with spatial_scatter and saves the image. Currently not used.
    Parameters
    ----------
    adata : AnnData
        ST data, should contain the clusters labels
    colour: str = 'multires_clst'
        cluster label found in adata.obs.columns
    k: int = 20
        Number of clusters, default is 20
    save_path: str = '.'
        Path to save image
    """ 
    
    complete_palette = ['#1f77b4', '#ff7f0e', '#279e68', '#d62728', '#aa40fc', '#8c564b', '#e377c2', '#b5bd61', '#17becf', '#aec7e8',
                 '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#dbdb8d', '#9edae5', '#ad494a', '#8c6d31',
                '#7f7f7f', '#c7c7c7', '#2ca02c', '#9467bd', '#8c9cff', '#00d9ff', '#a0ff87', '#ff6b9c', '#ceb301',  '#4d66e8', '#00cc99']

    custom_colors = complete_palette[:k]
    custom_cmap = ListedColormap(custom_colors)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sq.pl.spatial_scatter(adata, color=colour, size=50, shape=None, ax=axes[0], cmap=custom_cmap)
    axes[0].set_title("Spatial Clusters")
    
    sq.pl.spatial_scatter(adata, color=colour, img_res_key='hires', size=1, cmap=custom_cmap, alpha=0.8, ax=axes[1], dpi=200)
    axes[1].set_title("Tissue Overlay")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path, dpi=150)  # Save before showing
    plt.show() 

def plot_cluster_boundaries(adata: ad.AnnData, 
                            res: str='hires', 
                            colour: str = 'multires_clst', 
                            k: int = 20, 
                            alpha: float=0.0, 
                            lw: float=0.9,
                            title: str = '', 
                            save_path: str = '.'):
    """ 
    Plots the cluster boundaries on the original tissue image and saves the image. Currently not used.
    Parameters
    ----------
    adata : AnnData
        ST data, should contain the clusters labels
    library_id: str = ''
        library_id of the AnnData object
    res: str='hires'
    colour: str = 'multires_clst'
        cluster label found in adata.obs.columns
    k: int = 20
        Number of clusters, default is 20
    title: str = ''
    save_path: str = '.'
        Path to save image
    """ 
    
    library_id = list(adata.uns['spatial'].keys())[0]

    # images
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

    height, width = image.shape[:2]
        
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]

    clusters = adata.obs[colour].astype("category").cat.codes.values

    spot_diameter = (adata.uns["spatial"][library_id]["scalefactors"]["spot_diameter_fullres"] * sf)
    radius = int(spot_diameter) * 1.1  # can be tweaked

    # 3. Rasterize into label image
    label_image = -1 * np.ones((height, width), dtype=int)

    for (x, y), cl in zip(coords, clusters):
        rr, cc = disk((y, x), radius, shape=label_image.shape)
        label_image[rr, cc] = cl
    
    # Optional: smooth small gaps between spots
    label_image = ndi.grey_closing(label_image, size=(3, 3))
    
    cluster_polygons = {}
    
    for cl in np.unique(clusters):
        mask = label_image == cl
        
        # find_contours expects float image
        contours = skimage.measure.find_contours(mask.astype(float), level=0.5)
    
        # Convert (row,col) → (x,y)
        polygons = [np.flip(contour, axis=1) for contour in contours]
    
        cluster_polygons[int(cl)] = polygons

    unique_clusters = sorted(cluster_polygons.keys())
    n_clusters = len(unique_clusters)

    complete_palette = ['#1f77b4', '#ff7f0e', '#279e68', '#d62728', '#aa40fc', '#8c564b', '#e377c2', '#b5bd61', '#17becf', '#aec7e8',
                 '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#dbdb8d', '#9edae5', '#ad494a', '#8c6d31',
                '#7f7f7f', '#c7c7c7', '#2ca02c', '#9467bd', '#8c9cff', '#00d9ff', '#a0ff87', '#ff6b9c', '#ceb301',  '#4d66e8', '#00cc99']

    custom_colors = complete_palette[:k]
    custom_cmap = ListedColormap(custom_colors)

    cluster_colors = {cl: custom_cmap(i) for i, cl in enumerate(unique_clusters)}

    plt.figure(figsize=(6, 6), dpi=150)
    plt.imshow(image)

    for cl, polygons in cluster_polygons.items():
        fill_color = cluster_colors[cl]

        for poly in polygons:
            plt.fill(poly[:, 0], poly[:, 1], color=fill_color, alpha=alpha, zorder=2)

            plt.plot(poly[:, 0], poly[:, 1], color="black", linewidth=lw, zorder=2)

    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)  # Save before showing
    plt.show()