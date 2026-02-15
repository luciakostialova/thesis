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

def _perform_segmentation(adata: ad.AnnData, image, coords, library_id: str = None, res: str = 'hires', sigma: int = 0, 
                          superpixel: Optional[Literal["slic", "seed"]] = None):   
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
    images = adata.uns['spatial'][library_id]['images']
    image_rgb = images[res]
    
    if 'segmentation' not in adata.uns['spatial'][library_id].keys():

        # Ensure uint8 for OpenCV compatibility
        if image_rgb.dtype != np.uint8:
            image_rgb = (image_rgb * 255).astype(np.uint8)      
        
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)  # Create BGR copy for OpenCV-based SEEDS
        
        # 2. Gaussian smoothing
        if sigma > 0:            
            image_rgb_float = img_as_float(image_rgb)  
            image_rgb_smooth = gaussian(image_rgb_float, sigma=sigma)  # For SLIC (float image)
                    
            ksize = int(6 * sigma + 1)
            if ksize % 2 == 0:
                ksize += 1
            image_bgr_smooth = cv2.GaussianBlur(image_bgr, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)  # For SEEDS (uint8 image)
        else:
            image_rgb_smooth = img_as_float(image_rgb)
            image_bgr_smooth = image_bgr.copy()

        n_segments = 400   # use SAME params for fair comparison
        num_iterations = 50
        
        if superpixel == 'slic':
            compactness = 20
        
            labels = slic(image_rgb_smooth, n_segments=n_segments, compactness=compactness, max_num_iter=num_iterations, 
                          sigma=sigma, spacing=None, convert2lab=True, start_label=0)
        
            print(f'Number of SLIC segments, sigma={sigma}: {len(np.unique(labels))}')        
            plt.figure(figsize=(8, 8))
            plt.imshow(mark_boundaries(image_rgb_smooth, labels))
            plt.title(f'SLIC | sigma={sigma}')
            plt.axis('off')
        
        
        elif superpixel == 'seed':
            num_levels = 4
            prior = 2
            num_histogram_bins = 10
        
            height, width, channels = image_bgr_smooth.shape
            seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels, n_segments, num_levels, prior, num_histogram_bins)
            seeds.iterate(image_bgr_smooth, num_iterations)
            labels = seeds.getLabels()
        
            print(f'Number of SEEDS segments, sigma={sigma}: {len(np.unique(labels))}')   
            image_rgb_vis = cv2.cvtColor(image_bgr_smooth, cv2.COLOR_BGR2RGB)        
            plt.figure(figsize=(8, 8))
            plt.imshow(mark_boundaries(image_rgb_vis, labels))
            plt.title(f'SEEDS | sigma={sigma}')
            plt.axis('off')

    
        row, col = coords[:,1].astype(int), coords[:,0].astype(int)
    
        adata.obs['segmentation_label'] = labels[row, col]
        adata.uns['spatial'][library_id]['segmentation'] = {}
        adata.uns['spatial'][library_id]['segmentation']['segmentation_labels'] = labels
        adata.uns['spatial'][library_id]['segmentation']['sigma'] = sigma

    else:  
        labels = adata.uns['spatial'][library_id]['segmentation']['segmentation_labels']
        sigma = adata.uns['spatial'][library_id]['segmentation']['sigma']
        filtered_img = gaussian(image, sigma=sigma)
        img_flt = img_as_float(filtered_img)
        print(f'Image segmentation has already been done.')
        print(f'Number of SLIC segments, sigma={sigma} : {len(np.unique(labels))}')
        plt.figure(figsize=(8,8))
        plt.imshow(mark_boundaries(img_flt, labels))
        plt.title('Smoothing with Gaussian, sigma=')



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


def _get_segmentation_distance(adata: ad.AnnData, coords, library_id: str = None, weights: Tuple[float, float] | None = None):
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
    labels = adata.uns['spatial'][library_id]['segmentation']['segmentation_labels']
        
    # distance matrix for segmentation labeling
    if 'segmentation_dist' not in adata.obsp:
        segmentation_dist = segmentation_distance(adata, labels, coords)
        segmentation_dist_scaled = scaler.fit_transform(segmentation_dist)        
    
        # save distances to adata
        adata.obsp['segmentation_dist'] = segmentation_dist
        adata.obsp['segmentation_dist_scaled'] = segmentation_dist_scaled
        distance_matrix = segmentation_dist_scaled

    else:
        segmentation_dist_scaled = adata.obsp['segmentation_dist_scaled']
        distance_matrix = segmentation_dist_scaled
        
    if weights is not None:
        distance_matrix *= weights[0]

    return distance_matrix