import numpy                  as     np

from   scipy.spatial          import KDTree, Voronoi
from   scipy.ndimage          import distance_transform_edt 
from   skimage.morphology     import skeletonize
from   skimage                import measure, feature, color, filters
from   skimage.transform      import resize
from   collections            import defaultdict
from   src.routines           import centroid_dist, boundary_dist

def compute_nucleus_shape_features(mask):
    props = measure.regionprops(mask)

    features = {
        'convexity'          : [],
        'roundness'          : [],
        'solidity'           : [],
        'eccentricity'       : [],
        'skeleton_length'    : [],
        'skeleton_complexity': [],
    }

    for prop in props:
        convex_area = prop.convex_area
        features['convexity'].append(prop.area / convex_area if convex_area > 0 else 0)
        features['roundness'].append((4 * np.pi * prop.area) / (prop.perimeter ** 2) if prop.perimeter > 0 else 0)
        features['solidity'].append(prop.solidity)
        features['eccentricity'].append(prop.eccentricity)

        region_mask = prop.image
        skeleton    = skeletonize(region_mask)
        skel_length = np.sum(skeleton)
        features['skeleton_length'].append(skel_length)
        features['skeleton_complexity'].append(skel_length / prop.area if prop.area > 0 else 0)

    return features

def compute_nucleus_appearance_features(image, mask):
    if len(image.shape) == 3:
        rgb       = image.astype(np.float32) / 255.0
        gray      = color.rgb2gray(image)
        has_color = True
    else:
        gray      = image
        has_color = False

    if gray.shape != mask.shape:
        mask = resize(
            mask, gray.shape,
            order=0, preserve_range=True, anti_aliasing=False
        ).astype(mask.dtype)

    props              = measure.regionprops(mask, intensity_image=gray)

    gabor_frequencies  = [0.1, 0.3, 0.5]
    gabor_orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    gabor_responses    = {}

    for freq in gabor_frequencies:
        for theta in gabor_orientations:
            real, imag                     = filters.gabor(gray, frequency=freq, theta=theta)
            gabor_responses[(freq, theta)] = np.sqrt(real**2 + imag**2)

    features = {
        'mean_intensity':    [],

        'glcm_contrast':     [],
        'glcm_homogeneity':  [],
        'glcm_energy':       [],
        'glcm_correlation':  [],

        'mean_r':            [],
        'mean_g':            [],
        'mean_b':            [],
    }

    for freq in gabor_frequencies:
        for theta in gabor_orientations:
            features[f"gabor_f{freq:.2f}_t{theta:.2f}_mean"] = []

    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        nucleus_region         = gray[minr:maxr, minc:maxc]
        nucleus_mask           = prop.image
        nucleus_pixels         = nucleus_region[nucleus_mask]

        if len(nucleus_pixels) == 0:
            continue

        features['mean_intensity'].append(np.mean(nucleus_pixels))

        if nucleus_region.shape[0] > 5 and nucleus_region.shape[1] > 5:
            try:
                nucleus_norm  = (nucleus_region - nucleus_region.min()) / \
                                (nucleus_region.max() - nucleus_region.min() + 1e-10)
                nucleus_uint8 = (nucleus_norm * 255).astype(np.uint8)

                glcm = feature.graycomatrix(
                    nucleus_uint8,
                    distances = [1],
                    angles    = [0, np.pi / 4, np.pi / 2, 3*np.pi / 4],
                    levels    = 256,
                    symmetric = True,
                    normed    = True
                )
                features['glcm_contrast'].append(np.mean(feature.graycoprops(glcm,    'contrast')))
                features['glcm_homogeneity'].append(np.mean(feature.graycoprops(glcm, 'homogeneity')))
                features['glcm_energy'].append(np.mean(feature.graycoprops(glcm,      'energy')))
                features['glcm_correlation'].append(np.mean(feature.graycoprops(glcm, 'correlation')))
            except Exception:
                for k in ('glcm_contrast', 'glcm_homogeneity', 'glcm_energy', 'glcm_correlation'):
                    features[k].append(np.nan)
        else:
            for k in ('glcm_contrast', 'glcm_homogeneity', 'glcm_energy', 'glcm_correlation'):
                features[k].append(np.nan)

        for freq in gabor_frequencies:
            for theta in gabor_orientations:
                response_roi = gabor_responses[(freq, theta)][minr:maxr, minc:maxc]
                masked_vals  = response_roi[nucleus_mask]

                if len(masked_vals) > 0:
                    features[f"gabor_f{freq:.2f}_t{theta:.2f}_mean"].append(np.mean(masked_vals))
                else:
                    features[f"gabor_f{freq:.2f}_t{theta:.2f}_mean"].append(np.nan)

        if has_color:
            rgb_region = rgb[minr:maxr, minc:maxc]

            r = rgb_region[:, :, 0][nucleus_mask]
            g = rgb_region[:, :, 1][nucleus_mask]
            b = rgb_region[:, :, 2][nucleus_mask]

            features['mean_r'].append(np.mean(r))
            features['mean_g'].append(np.mean(g))
            features['mean_b'].append(np.mean(b))

        else:
            for k in ('mean_r', 'mean_g', 'mean_b'):
                features[k].append(np.nan)

    return features

def compute_patch_appearance_features(image, switches):
    features = {}

    if len(image.shape) == 3:
        rgb       = image.astype(np.float32) / 255.0
        gray      = color.rgb2gray(image)
        has_color = True
    else:
        gray      = image
        has_color = False

    gray                       = (gray - gray.min()) / (gray.max() - gray.min() + 1e-10)
    gray_uint8                 = (gray * 255).astype(np.uint8)
    distances                  = [1, 2, 3]
    angles                     = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    if switches['mean_intensity'] == True:
        features['mean_intensity'] = np.mean(gray)

    if switches['glcm'] == True: 
        try:
            glcm = feature.graycomatrix(gray_uint8, 
                                        distances = distances, 
                                        angles    = angles,
                                        levels    = 256, 
                                        symmetric = True, 
                                        normed    = True
                                        )

            features['glcm_contrast']    = np.mean(feature.graycoprops(glcm, 'contrast'))
            features['glcm_homogeneity'] = np.mean(feature.graycoprops(glcm, 'homogeneity'))
            features['glcm_energy']      = np.mean(feature.graycoprops(glcm, 'energy'))
            features['glcm_correlation'] = np.mean(feature.graycoprops(glcm, 'correlation'))
        except Exception:
            features['glcm_contrast']    = np.nan
            features['glcm_homogeneity'] = np.nan
            features['glcm_energy']      = np.nan
            features['glcm_correlation'] = np.nan

    if switches['rgb'] == True:
        if has_color:
            r                            = rgb[:, :, 0]
            g                            = rgb[:, :, 1]
            b                            = rgb[:, :, 2]

            features['mean_r']           = np.mean(r)
            features['mean_g']           = np.mean(g)
            features['mean_b']           = np.mean(b)

        else:
            color_keys = [
                'mean_r', 
                'mean_g', 
                'mean_b'
            ]
            for k in color_keys:
                features[k] = np.nan

    return features

def compute_topological_features(mask, k=5, radius=50):
    props      = measure.regionprops(mask)
    centroids  = np.array([p.centroid for p in props])       
    labels     = np.array([p.label    for p in props])

    boundaries = []
    
    for prop in props:
        region_mask = (mask == prop.label).astype(np.uint8)
        edt         = distance_transform_edt(region_mask)
        boundary_px = np.argwhere(edt == 1)

        boundaries.append(boundary_px)

    N                 = len(props)
    tree              = KDTree(centroids)

    vor               = Voronoi(centroids)
    voronoi_neighbors = {i: set() for i in range(N)}

    for ridge in vor.ridge_points:
        i, j = ridge

        voronoi_neighbors[i].add(j)
        voronoi_neighbors[j].add(i)

    features = {key: [] for key in [
        'nn_centroid', 
        'nn_boundary',
        'knn_centroid_mean', 
        'knn_boundary_mean', 
        'radius_centroid_mean', 
        'radius_boundary_mean',
        'voronoi_centroid_mean', 
        'voronoi_boundary_mean'
    ]}

    for i in range(N):
        nn_dists, nn_idxs = tree.query(centroids[i], k=2)
        j_nn              = nn_idxs[1]

        features['nn_centroid'].append(centroid_dist(i, j_nn, centroids))
        features['nn_boundary'].append(boundary_dist(i, j_nn, boundaries))

        k_actual    = min(k + 1, N)
        _, knn_idxs = tree.query(centroids[i], k=k_actual)
        knn_idxs    = knn_idxs[1:]

        features['knn_centroid_mean'].append(
            np.mean([centroid_dist(i, j, centroids)     for j in knn_idxs]))
        features['knn_boundary_mean'].append(
            np.nanmean([boundary_dist(i, j, boundaries) for j in knn_idxs]))

        radius_idxs = tree.query_ball_point(centroids[i], r=radius)
        radius_idxs = [j for j in radius_idxs if j != i]

        if len(radius_idxs) > 0:
            features['radius_centroid_mean'].append(
                np.mean([centroid_dist(i, j, centroids)     for j in radius_idxs]))
            features['radius_boundary_mean'].append(
                np.nanmean([boundary_dist(i, j, boundaries) for j in radius_idxs]))
        else:
            features['radius_centroid_mean'].append(np.nan)
            features['radius_boundary_mean'].append(np.nan)

        vor_idxs = list(voronoi_neighbors[i])

        if len(vor_idxs) > 0:
            features['voronoi_centroid_mean'].append(
                np.mean([centroid_dist(i, j, centroids)     for j in vor_idxs]))
            features['voronoi_boundary_mean'].append(
                np.nanmean([boundary_dist(i, j, boundaries) for j in vor_idxs]))
        else:
            features['voronoi_centroid_mean'].append(np.nan)
            features['voronoi_boundary_mean'].append(np.nan)

    return features