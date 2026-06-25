##
## By Rucha Deshpande
## Code refactor by Elim Thompson (6/11/2025)
## 
## This script processes a pair of mammogram datasets to extract image
## features and gradients, and saves the results. The pair of datasets
## are a real-synthetic pair where the synthetic dataset was generated
## from the corresponding real dataset. Results include a histogram of
## angle gradients. The histogram binning is based on the range of
## angle gradients in the real dataset. 
##
## With the entire real dataset, this script also builds normalized
## histograms of angle gradients for all images with its range and
## binning defined only by the real dataset. 
##
## To run this script, use the command:
##     python process_datasets.py --first_n_files 10 --verbose
##                                --outpath 'path/to/save/data.p'
######################################################################

################################
## Import packages
################################
import os
import glob
import pickle
import argparse
import warnings
import numpy as np

import ImageProcessor

# Ignore all UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

################################
## Define constants
################################
## User must change the paths to the datasets below.
## Key = dataset name
## Value = dictionary with keys 'real' and 'synthetic' for the paths to the datasets
datasets = {'VinDr':{'real':'example_inputs/real/',
                     'synthetic':'example_inputs/synthetic/'}}

## List of datasets that contain only MLO images
MLO_datasets = ['VinDr', 'CSAW'] 

## Number of bins for normalized angle gradients histogram
n_distribution_bins = 16

################################
## Define functions
################################
def obtain_binning (real_angle_gradients, n_distribution_bins=n_distribution_bins):

    ''' Obtain bin edges for angle gradient distributions based on the real dataset.

        Inputs
        ------
        real_angle_gradients : list of np.ndarray
            Angle gradients from the real dataset.
        n_distribution_bins : int
            Number of bins for the histogram of angle gradients.
            Default is set to 16.

        Output
        ------
        bin_edges : np.ndarray  
            Array of bin edges for the histogram of angle gradients.
    '''

    ## Gather all angle gradients from the real dataset
    all_angle_gradients = np.stack (real_angle_gradients)

    ## Define the bin edges based on the 1st and 99th percentiles of the angle gradients
    p1, p99 = np.percentile(all_angle_gradients, [1, 99])
    #  nbins-2 intervals with nbins-1 bin edges
    inner_bins = np.linspace(p1, p99, n_distribution_bins-1) 

    ## First and last bins are set to -inf and +inf, respectively
    bin_edges = np.concatenate(([-np.inf], inner_bins, [np.inf])) 
    return bin_edges

def get_normalized_angle_gradient_distributions (data, n_distribution_bins=n_distribution_bins):

    ''' Get normalized angle gradient distributions for all images in the dataset.
        Distribution binning is based on the real dataset's angle gradients. 
        Normalization is done with respect to each image.

        Inputs
        ------
        data : dict
            Dictionary containing angle gradients from the real and synthetic datasets.
        n_distribution_bins : int
            Number of bins for the histogram of angle gradients.
            Default is set to 16.

        Output
        ------
        data : dict
            Updated dictionary with angle gradient distributions for each image type.
            Each distribution is normalized and stored in 'angle_gradient_distributions'.
            The bin edges are stored in 'angle_gradient_distribution_bin_edges'.
    '''

    ## Get bin edges based on the real dataset's angle gradients
    distribution_bin_edges = obtain_binning(data['real']['angle_gradients'], n_distribution_bins)

    ## Get the normalized angle gradient distributions for each image type
    for image_type in ['real', 'synthetic']:
        angle_gradient_distributions = [np.histogram (angle_gradient, bins=distribution_bin_edges)[0] / len(angle_gradient)
                                        for angle_gradient in data[image_type]['angle_gradients']]
        data[image_type]['angle_gradient_distributions'] = angle_gradient_distributions

    ## Store the bin edges in the data dictionary
    data['angle_gradient_distribution_bin_edges'] = distribution_bin_edges
    return data

def get_gradient_per_image (image_path, dataset_name, short_filename, 
                            isSynthetic=False, isMLO=False, isCSAW=False):

    ''' Get angle gradients for a single image.

        Inputs
        ------
        image_path : str
            Path to the image file.
        dataset_name : str
            Name of the dataset (e.g., 'VinDrReal', 'VinDrSynthetic').
        short_filename : str
            Short filename for the image, used for identification.
        isSynthetic : bool, optional
            Whether the image is from a synthetic dataset (default is False).
        isMLO : bool, optional
            Whether the image is an MLO image (default is False).
        isCSAW : bool, optional
            Whether the image is from the CSAW dataset (default is False).

        Output
        ------
        adict : dict
            Dictionary containing angle gradients, pixels along edge, and
            breast area for the image.
    '''

    image_type = 'synthetic' if isSynthetic else 'real'

    processor = ImageProcessor.ImageProcessor(image_path, dataset_name, image_type)
    processor.isMLO = isMLO
    processor.isCSAW = isCSAW
    processor.short_filename = short_filename
    processor.do_intermediate_plots = False
    processor.do_plots = False
    processor.build_angle_gradients ()

    adict = {'angle_gradient':processor.binned_angle_gradients,
             'pixels_along_edge':processor.pixels_along_edge,
             'breast_area':processor.breast_area}

    return adict

def get_gradient_per_dataset (dataset_folder, dataset_name, first_n_files=None, isSynthetic=False,
                              isMLO=False, isCSAW=False, verbose=False):

    """ Get angle gradients for all images in a dataset.

        Inputs
        ------
        dataset_folder : str
            Path to the folder containing datasets with PNG images.        
        dataset_name : str
            Name of the dataset (e.g., 'VinDrReal', 'VinDrSynthetic').
        first_n_files : int, optional
            If specified, only process the first N files in the dataset
            (default is None, meaning all files).
        isSynthetic : bool, optional
            Whether the dataset is synthetic (default is False).
        isMLO : bool, optional
            Whether the dataset is MLO images (default is False).
        isCSAW : bool, optional
            Whether the dataset is CSAW images (default is False).
        verbose : bool, optional
            Whether to print verbose output during processing (default is False).

        Output
        ------
        pandas.DataFrame
            DataFrame containing image paths, angle gradients, short filenames,
            breast area, and pixels along edge for each image in the dataset.
    """

    image_paths = glob.glob(dataset_folder + '/*.png')

    data_dict = {'image_path': [], 'angle_gradients': [], 
                 'short_filename': [], 'pixels_along_edge': [],
                 'breast_area': []}

    for idx, image_path in enumerate (image_paths[:first_n_files]):

        if verbose and idx%50==0: print (f'   -- {image_path}')

        short_filename = dataset_name + '_' + str(idx).zfill(10)
        adict = get_gradient_per_image (image_path, dataset_name, short_filename, 
                                        isSynthetic=isSynthetic, isMLO=isMLO, isCSAW=isCSAW)

        data_dict['image_path'].append(image_path)
        data_dict['short_filename'].append(short_filename)
        data_dict['breast_area'].append(adict['breast_area'])
        data_dict['angle_gradients'].append(adict['angle_gradient'])
        data_dict['pixels_along_edge'].append(adict['pixels_along_edge'])

    return data_dict

def get_gradients (first_n_files=None, verbose=False,
                   n_distribution_bins=n_distribution_bins):

    ''' Process all datasets to extract angle gradients and save results.

        Inputs
        ------
        first_n_files : int, optional
            If specified, only process the first N files in each dataset
            (default is None, meaning all files).
        verbose : bool, optional
            Whether to print verbose output during processing (default is False).
        n_distribution_bins : int, optional
            Number of bins for the histogram of angle gradients (default is 16).
    '''

    ## Initialize the data dictionary to store results
    data = {'real':{'image_path': [], 'angle_gradients': [],
                    'breast_area': [], 'short_filename': [],
                    'pixels_along_edge': [], 'dataset_name':[]},
            'synthetic':{'image_path': [], 'angle_gradients': [],
                         'breast_area': [], 'short_filename': [],
                         'pixels_along_edge': [], 'dataset_name':[]}}

    ## Process each dataset to output angle gradients and pixels along edge.
    for dataset_name, dataset_info in datasets.items():

        isMLO = dataset_name in MLO_datasets
        isCSAW = dataset_name == 'CSAW'

        for image_type in ['real', 'synthetic']:
            dataset_fullname = dataset_name + '_' + image_type
            dataset_folder = dataset_info[image_type]
            if verbose: print (f'Processing {image_type} dataset: {dataset_fullname}')
            subdata = get_gradient_per_dataset(dataset_folder, dataset_fullname, first_n_files=first_n_files,
                                               isSynthetic=False, isMLO=isMLO, isCSAW=isCSAW,
                                               verbose=verbose)
            for key in data[image_type]:
                if key == 'dataset_name': continue
                data[image_type][key].extend(subdata[key])
            data[image_type]['dataset_name'].extend([dataset_fullname] * len(subdata['image_path']))

    ## Once all the angle gradients are collected, get the normalized
    ## angle gradient distributions for each image. Binning is based on
    ## the real dataset's angle gradients.
    data = get_normalized_angle_gradient_distributions (data, n_distribution_bins=n_distribution_bins)

    return data

################################
## Script starts here
################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Process an integer argument.")
    parser.add_argument('--first_n_files', type=int, default=None, help='Only process the first N files')
    parser.add_argument('--verbose', action='store_true', default=False, help='Enable verbose output')
    parser.add_argument('--outpath', type=str, required=True, help='Output file path for the data dictionary')
    args = parser.parse_args()

    verbose = args.verbose
    outpath = args.outpath
    first_n_files = args.first_n_files

    ## Check if the output path exists, if not, create it
    if not os.path.exists(outpath):
        os.makedirs (outpath, exist_ok=True)

    ## Process all data with angle gradients and save results
    data = get_gradients (first_n_files=first_n_files, verbose=verbose)

    # Put them all into a python dictionary
    with open(outpath + 'data.p', 'wb') as f:
        pickle.dump(data, f)
    f.close ()

    if verbose: print ('Done processing all datasets!')
