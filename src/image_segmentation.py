"""
Image segmentation is performed, the adjacent distance is computed and saved to adata.
There are two algorithms to choose from that perform the image segmentation.
The first is SLIC. details
The second algorithm is SEED. details.

FIX: add algorithm details to comments; set superpixel params as voluntary params to set in main function.
CHECK for params passed through the functions if necessary.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad
import skimage
from skimage.filters import gaussian
from skimage.util import img_as_float
from skimage.segmentation import slic, mark_boundaries
from skimage.draw import draw
from sklearn.preprocessing import MinMaxScaler
from typing import Optional, Literal, Tuple
import cv2

def _perform_segmentation(adata: ad.AnnData, image, coords, sigma: int = 0, superpixel: Optional[Literal["slic", "seed"]] = None):   
    """ 
    Image segmentation is performed by SLIC or SEED superpixel segmentation algorithm. 
    The parameters are set to fit the specific BRCA image.
    The image is smoothed if sigma is set to be larger than 0.
    The image boundaries are printed on the smoothed image. Segmentation labels are saved in adata.
    If segmentation labels are already saved in adata, only the image is printed.
    Parameters
    ----------
    adata: AnnData
    library_id: str = None
        Library identifier
    res: str = 'hires'
        Resolution ('hires' or 'lowres')
    sigma: int = 0
        standard deviation for Gaussian blurring, default is no smoothing (0).
    superpixel: Optional[Literal["slic", "seed"]] = None
        algorithm for superpixel segmentation
    image: 
        image to be segmented
    coords: 
        coordinates of spatial barcodes on the tissue
    """
    segmentation_key = f'segmentation_{superpixel}_sigma{sigma}'
    library_id = list(adata.uns['spatial'].keys())[0]

    if segmentation_key not in adata.uns['spatial'][library_id].keys():
        height, width, channels = image.shape

        # --- Gaussian smoothing, RGB ---
        if sigma > 0:
            smoothed = gaussian(image, sigma=sigma, channel_axis=-1, preserve_range=True).astype(np.uint8)
        else:
            smoothed = image.copy()
    
        if superpixel == 'seed':
            # params
            num_superpixels = 400
            num_levels = 4  # Number of block levels. The more levels, the more accurate is the segmentation
            prior = 3  # enable 3x3 shape smoothing term if >0. A larger value leads to smoother shapes. prior must be in the range [0, 5]. 
            num_histogram_bins = 15
            num_iterations = 40
            double_step	= True  # If true, iterate each block level twice for higher accuracy.
            
            smoothed_bgr = cv2.cvtColor(smoothed, cv2.COLOR_RGB2BGR)
    
            seeds = cv2.ximgproc.createSuperpixelSEEDS(width,  height, channels, num_superpixels, 
                                                       num_levels, prior, num_histogram_bins, double_step)
            seeds.iterate(smoothed_bgr, num_iterations)
            labels = seeds.getLabels()
            n_labels = seeds.getNumberOfSuperpixels()
        
            mask = seeds.getLabelContourMask(thick_line=True)        
            overlay_bgr = smoothed_bgr.copy()
            overlay_bgr[mask == 255] = [0, 255, 0]
        
            # Convert back BGR → RGB for plotting
            overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
            print('SEED segmentation')
    
        elif superpixel == 'slic':
            # params
            num_superpixels = 400    # The (approximate) number of labels in the segmented output image.
            compactness = 20.0      
            # color proximity and space proximity tradeoff. Higher values give more weight to space proximity, superpixels are more square/cubic.
            max_num_iter = 80  # Maximum number of iterations of k-means.
            enforce_connectivity = True  # Whether the generated segments are connected or not
            
            # SLIC segmentation
            labels = slic(smoothed, n_segments=num_superpixels, compactness=compactness, max_num_iter=max_num_iter, 
                  start_label=1, channel_axis=-1, enforce_connectivity=enforce_connectivity)
            # Overlay boundaries
            overlay = mark_boundaries(smoothed,labels, color=(0, 1, 0), mode="thick")
            n_labels = len(np.unique(labels))
            print('SLIC segmentation')

        
        plt.figure(figsize=(8, 8))
        plt.imshow(overlay)
        plt.title(f"σ = {sigma} | superpixel = {n_labels}")
        plt.axis('off')
    
        row, col = coords[:,1].astype(int), coords[:,0].astype(int)
    
        adata.obs[segmentation_key] = labels[row, col]
        adata.uns['spatial'][library_id][segmentation_key] = {}
        adata.uns['spatial'][library_id][segmentation_key]['segmentation_labels'] = labels
        adata.uns['spatial'][library_id][segmentation_key]['sigma'] = sigma

    else:  
        labels = adata.uns['spatial'][library_id][segmentation_key]['segmentation_labels']
        sigma = adata.uns['spatial'][library_id][segmentation_key]['sigma']
        print(f'Image segmentation has already been done.')
        print(f'Number of SLIC segments, sigma={sigma} : {len(np.unique(labels))}')
        # filtered_img = gaussian(image, sigma=sigma)
        # img_flt = img_as_float(filtered_img)
        # plt.figure(figsize=(8, 8))
        # plt.imshow(mark_boundaries(smoothed,labels, color=(0, 1, 0), mode="thick"))
        # plt.title(f"σ = {sigma} | superpixel = {n_labels}")
        # plt.axis('off')

def segmentation_distance(adata: ad.AnnData, labels, coords):
    """ 
    The segmentation-based distance is calculated as the unique number of region 
    labels on a straight line between any two spatial spots. 
    """ 
    
    rows, cols = coords[:,1].astype(int), coords[:,0].astype(int)
    n = len(adata.obs.index)
    distance_matrix = np.zeros((n, n), dtype=np.int32)

    for i in range(n):    
        r1 = rows[i]
        c1 = cols[i]
        
        for j in range(i, n):        
            r2 = rows[j]
            c2 = cols[j]
            
            rr, cc = draw.line(r1, c1, r2, c2)
            labels_barcodes = labels[rr, cc]
    
            dist = len(np.unique(labels_barcodes)) - 1
            distance_matrix[i,j] = dist
            distance_matrix[j,i] = dist

    df = pd.DataFrame(distance_matrix, columns=adata.obs.index, index=adata.obs.index)
    segmentation_dist = df.values
    
    return segmentation_dist


def _get_segmentation_distance(adata: ad.AnnData, coords, sigma: int = 0, superpixel: Optional[Literal["slic", "seed"]] = None):
    """  
    The segmentation distance is computed via `segmentation_distance`, scaled and saved to adata. 
    Parameters
    ----------
    adata : AnnData
    library_id: str = None
        library identifier
    weights: Tuple[float, float] | None = None
        weights used for weighing the distances
    coords:
        coordinates of spatial barcodes on the tissue
    """ 
    scaler = MinMaxScaler()
    segmentation_key = f'segmentation_{superpixel}_sigma{sigma}'
    library_id = list(adata.uns['spatial'].keys())[0]
    labels = adata.uns['spatial'][library_id][segmentation_key]['segmentation_labels']
        
    # distance matrix for segmentation labeling
    if 'segmentation_dist' not in adata.obsp:
        segmentation_dist = segmentation_distance(adata, labels, coords)
        segmentation_dist_scaled = scaler.fit_transform(segmentation_dist)        
    
        # save distances to adata
        adata.obsp[f'segmentation_{superpixel}_sigma{sigma}_dist'] = segmentation_dist
        adata.obsp[f'segmentation_{superpixel}_sigma{sigma}_dist_scaled'] = segmentation_dist_scaled
        distance_matrix = segmentation_dist_scaled

    else:
        segmentation_dist_scaled = adata.obsp[f'segmentation_{superpixel}_sigma{sigma}_dist_scaled']
        distance_matrix = segmentation_dist_scaled

    return distance_matrix