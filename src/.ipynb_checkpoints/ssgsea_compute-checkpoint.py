from scipy.sparse import issparse
import numpy as np
import json


def compute_hallmark_es(adata, json_path, layer=None, tau=0.25):  # add correct normalisation!!!
    """
    Compute normalized enrichment scores (NES) for all Hallmark gene sets.

    Parameters
    ----------
    adata : AnnData
    json_path : str
        Path to MSigDB hallmark JSON file
    layer : str or None

    Returns
    -------
    results : dict
        {gene_set_name: normalized ES (n_spots,)}
    """

    # Load gene sets
    with open(json_path, 'r') as file:
        data = json.load(file)

    results = {}
    n_spots = adata.n_obs

    for gs_name, gs_info in data.items():
        gene_set = gs_info['geneSymbols']

        try:
            # Compute RS + ES
            rs, es = compute_running_sum_statistic(
                adata, gene_set, layer=layer, tau=tau
            )

            # --- Min-max normalization ---
            # es_min = np.min(es)
            # es_max = np.max(es)

            # if es_max > es_min:
            #     nes = (es - es_min) / (es_max - es_min)
            # else:
            #     # constant vector → set to 0
            #     nes = np.zeros_like(es)

            results[gs_name] = es

        except Exception as e:
            print(f"Skipping {gs_name}: {e}")
            continue

    # for gs_name, nes in results.items():
    #     adata.obs[gs_name] = nes
        
    return results


def compute_gene_set_cdf(adata, gene_set, layer=None, tau=0.25):  # add weight param!!!
    """
    Compute empirical CDF of a gene set S over ranked genes for each spot.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (spots x genes), can be sparse.
    gene_set : list or set
        List of gene names (must match adata.var_names).
    layer : str or None
        If provided, use adata.layers[layer] instead of adata.X.

    Returns
    -------
    cdf : np.ndarray
        Array of shape (n_spots, n_genes) containing P_S^{(j)}(k)
    """

    # Select matrix
    X = adata.layers[layer] if layer is not None else adata.X

    n_spots, n_genes = adata.shape

    # Map gene set to indices
    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}
    S_idx = np.array([gene_to_idx[g] for g in gene_set if g in gene_to_idx])

    if len(S_idx) == 0:
        raise ValueError("None of the gene_set genes were found in adata.var_names.")

    # Boolean mask for gene set
    S_mask = np.zeros(n_genes, dtype=bool)
    S_mask[S_idx] = True

    N_S = S_mask.sum()

    # Output
    cdf = np.zeros((n_spots, n_genes), dtype=np.float32)

    for j in range(n_spots):
        # Extract row
        row = X[j]

        # Convert sparse row → dense vector
        if issparse(row):
            row = row.toarray().ravel()
        else:
            row = np.asarray(row).ravel()

        # Rank genes (descending expression)
        ranked_idx = np.argsort(-row)

        # # Indicator for whether ranked genes belong to S
        # hits = S_mask[ranked_idx].astype(np.int32)

        # # Cumulative sum → C_S^{(j)}(k)
        # cumsum_hits = np.cumsum(hits)

        # # Normalize → P_S^{(j)}(k)
        # cdf[j, :] = cumsum_hits / N_S

        # --- Define rank values ---
        # Highest expression gets largest rank
        ranks = np.arange(n_genes, 0, -1)  # p, p-1, ..., 1

        # Apply exponent tau
        rank_weights = ranks ** tau

        # Keep only genes in S
        weighted_hits = rank_weights * S_mask[ranked_idx]

        # Running sum
        cumsum_vals = np.cumsum(weighted_hits)

        # Normalize by total weight in S
        total_weight = np.sum(rank_weights[S_mask[ranked_idx]])

        if total_weight == 0:
            cdf[j, :] = 0
        else:
            cdf[j, :] = cumsum_vals / total_weight

        # weights = row[ranked_idx]

        # # keep only genes IN S
        # weighted_hits = (weights * S_mask[ranked_idx]) ** tau

        # cumsum_vals = np.cumsum(weighted_hits)

        # # normalize by total weight in S for this spot
        # total_weight = row[S_mask].sum()

        # if total_weight == 0:
        #     cdf[j, :] = 0  # avoid division by zero
        # else:
        #     cdf[j, :] = cumsum_vals / total_weight

    return cdf




# def compute_gene_set_cdf(adata, gene_set, layer=None, tau=1):
#     """
#     Rank-weighted empirical CDF (Barbie et al. style, using ranks).

#     Parameters
#     ----------
#     adata : AnnData
#     gene_set : list or set
#     layer : str or None
#     tau : float
#         Exponent applied to rank weights

#     Returns
#     -------
#     cdf : np.ndarray (n_spots, n_genes)
#     """

#     # Select matrix
#     X = adata.layers[layer] if layer is not None else adata.X
#     n_spots, n_genes = adata.shape

#     # Map gene set
#     gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}
#     S_idx = np.array([gene_to_idx[g] for g in gene_set if g in gene_to_idx])

#     if len(S_idx) == 0:
#         raise ValueError("None of the gene_set genes were found.")

#     # Mask
#     S_mask = np.zeros(n_genes, dtype=bool)
#     S_mask[S_idx] = True

#     # Output
#     cdf = np.zeros((n_spots, n_genes), dtype=np.float32)

#     for j in range(n_spots):
#         row = X[j]

#         # Sparse → dense
#         if issparse(row):
#             row = row.toarray().ravel()
#         else:
#             row = np.asarray(row).ravel()

#         # Rank genes (descending expression)
#         ranked_idx = np.argsort(-row)

#         # --- Define rank values ---
#         # Highest expression gets largest rank
#         ranks = np.arange(n_genes, 0, -1)  # p, p-1, ..., 1

#         # Apply exponent tau
#         rank_weights = ranks ** tau

#         # Keep only genes in S
#         weighted_hits = rank_weights * S_mask[ranked_idx]

#         # Running sum
#         cumsum_vals = np.cumsum(weighted_hits)

#         # Normalize by total weight in S
#         total_weight = np.sum(rank_weights[S_mask[ranked_idx]])

#         if total_weight == 0:
#             cdf[j, :] = 0
#         else:
#             cdf[j, :] = cumsum_vals / total_weight

#     return cdf



def compute_background_cdf(adata, gene_set, layer=None):
    """
    Compute empirical CDF over genes NOT in gene_set,
    using mean gene expression across all spots.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (spots x genes), can be sparse.
    gene_set : list or set
        List of gene names.
    layer : str or None
        Optional layer to use instead of adata.X.

    Returns
    -------
    cdf : np.ndarray
        background CDF curve
    """

    # Select matrix
    X = adata.layers[layer] if layer is not None else adata.X
    n_spots, n_genes = adata.shape

    # Map gene set
    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}
    S_idx = np.array([gene_to_idx[g] for g in gene_set if g in gene_to_idx])

    # Mask for gene set
    S_mask = np.zeros(n_genes, dtype=bool)
    S_mask[S_idx] = True

    # Complement mask (genes NOT in S)
    Sc_mask = ~S_mask
    N_Sc = Sc_mask.sum()

    if N_Sc == 0:
        raise ValueError("All genes are in the gene set; complement is empty.")

    # --- Compute mean expression ---
    if issparse(X):
        mean_expr = np.array(X.mean(axis=0)).ravel()
    else:
        mean_expr = np.asarray(X.mean(axis=0)).ravel()

    # --- Rank genes by mean expression ---
    ranked_idx = np.argsort(-mean_expr)

    # --- Indicator for genes NOT in S ---
    hits = Sc_mask[ranked_idx].astype(np.int32)

    # --- CDF ---
    cdf = np.cumsum(hits) / N_Sc

    # # --- Weighted cumulative sum ---
    # weights = mean_expr[ranked_idx]

    # # keep only genes outside S
    # weighted_hits = weights * Sc_mask[ranked_idx]

    # cumsum_vals = np.cumsum(weighted_hits)

    # # --- Normalization ---
    # total_weight = mean_expr[Sc_mask].sum()

    # cdf = cumsum_vals / total_weight

    return cdf



def compute_running_sum_statistic(adata, gene_set, layer=None, tau=0.25):
    """
    Compute running sum (GSEA-like enrichment) for all spots.

    Returns
    -------
    rs : np.ndarray (n_spots, k_max)
    es : np.ndarray (n_spots,)  # max deviation (enrichment score)
    """

    # Foreground (per spot)
    cdf_S = compute_gene_set_cdf(
        adata, gene_set, layer=layer, tau=tau
    )

    # Background (global)
    cdf_Sc = compute_background_cdf(
        adata, gene_set, layer=layer
    )

    # Broadcast background to all spots
    cdf_Sc = np.tile(cdf_Sc, (adata.n_obs, 1))

    # Running sum
    rs = cdf_S - cdf_Sc

    # Enrichment score (max deviation like GSEA)
    # es = np.max(rs, axis=1)  # change depending on where the GS should be enriched - up or down. GSEA KS_ statistic used
    es = np.sum(rs, axis=1)  # ssGSEA ES, needs to be normalised - minmax normalisation with all the ES scores

    return rs, es



