""" Plots the clusters and cluster boundaries on the tissue. The images are saved.""" 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgb
from typing import List
import squidpy as sq
from skimage import io, segmentation
from scipy import ndimage as ndi


def plot_clusters(adata, colour: str = 'multires_clst', k: int = 20, save_path: str = '.'):
    """ 
    Plots the clusters on the tissue with spatial_scatter and saves the image. 
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

def plot_cluster_boundaries(adata, library_id: str = '', colour: str = 'multires_clst', k: int = 20, boundary_colour: List[int] | None=None, 
                           title: str = '', save_path: str = '.'):
    """ 
    Plots the cluster boundaries on the original tissue image and saves the image.
    Parameters
    ----------
    adata : AnnData
        ST data, should contain the clusters labels
    library_id: str = ''
        library_id of the AnnData object
    colour: str = 'multires_clst'
        cluster label found in adata.obs.columns
    k: int = 20
        Number of clusters, default is 20
    boundary_colour: List[int] | None=None
        colour of boundaries printed
    title: str = ''
    save_path: str = '.'
        Path to save image
    """ 

    if boundary_colour is None:
        boundary_colour = [0,0,0]

    if library_id == '':
        library_id = list(adata.uns['spatial'].keys())[0]

    hires_image = adata.uns['spatial'][library_id]['images']['hires']

    complete_palette = ['#1f77b4', '#ff7f0e', '#279e68', '#d62728', '#aa40fc', '#8c564b', '#e377c2', '#b5bd61', '#17becf', '#aec7e8',
                 '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#dbdb8d', '#9edae5', '#ad494a', '#8c6d31',
                '#7f7f7f', '#c7c7c7', '#2ca02c', '#9467bd', '#8c9cff', '#00d9ff', '#a0ff87', '#ff6b9c', '#ceb301',  '#4d66e8', '#00cc99']

    custom_colors = complete_palette[:k]

    custom_cmap = ListedColormap(custom_colors)

        
    sq.pl.spatial_scatter(adata, color=colour, img=True, img_res_key='hires', size=1.9, cmap=custom_cmap, figsize=(6,6), dpi=327,
        title='', legend_loc=None, frameon=False, colorbar=None, axis_label='', save='multires_clst.png')

    sq.pl.spatial_scatter(adata, color=colour, img=True, img_res_key='hires', size=1.9, cmap=custom_cmap, figsize=(6,6), dpi=327,
        title='', legend_loc=None, frameon=False, colorbar=None, axis_label='', save='no_clst.png')

    #plt.clf()

    # Load your image
    image = io.imread("figures/multires_clst.png")
    hires_image = io.imread("figures/no_clst.png")

    # Crop the larger image to match hires image shape
    if image.shape[0] > hires_image.shape[0]:
        image = image[:hires_image.shape[0], :, :]
    if image.shape[1] > hires_image.shape[1]:
        image = image[:, :hires_image.shape[1], :]
    
    image = image[:,:,:3]
    hires_image = hires_image[:,:,:3]
    image_q = (image // 8) * 8
    image_float = image / 255.0  # convert to 0–1 range
    cluster_rgbs = np.array([to_rgb(c) for c in custom_colors])
    
    # Convert to labels (by unique colors), Each unique color will be treated as a region
    _, labels = np.unique(image_q.reshape(-1, image_q.shape[2]), axis=0, return_inverse=True)
    labels = labels.reshape(image_q.shape[:2])
        
    # Find region boundaries
    boundaries = segmentation.find_boundaries(labels, mode='thick')
    # Thicken boundaries (3 pixels wide), if needed
    boundaries_thick = ndi.binary_dilation(boundaries, iterations=1)

    # Overlay black boundaries on the original image
    outlined = hires_image.copy()
    outlined[boundaries] = boundary_colour
      
    # --- Compute color distances and classify pixels. For each pixel, find its distance to all cluster colors.
    dists = np.linalg.norm(image_float[..., None, :] - cluster_rgbs[None, None, :, :], axis=-1)
    min_dist = np.min(dists, axis=-1)
    
    # --- Build mask --- If a pixel is close enough to a known color, mark it as "cluster"
    mask_colored = min_dist < 0.08  # adjust tolerance if needed
    mask_background = ~mask_colored

    # Overlay black boundaries on the original image
    outlined = hires_image.copy()
    overlap = mask_colored & boundaries_thick
    outlined[overlap] = boundary_colour
    
    plt.figure(dpi=150)
    plt.title(title)
    plt.imshow(outlined)
    plt.axis('off')
    
    # Save FIRST, then show
    plt.savefig(save_path, dpi=150)  # Save before showing
    plt.show() 