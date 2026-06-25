##
## By Rucha Deshpande
## Code refactor by Elim Thompson (6/10/2025)
##
## This script contains the imageProcessor class that 
##    * smooths an image
##    * detects edges
##    * extracts and quantizes angular gradient features
##    * builds normalized distribution of the angular gradient 
##
## To use this class, create an instance of the ImageProcessor class
## with the image path, dataset name, and image type (real or synthetic).
##
## Example:
## 
##    image_path = '/path/to/image.png'
##    dataset_name = 'VinDrReal'
##    image_type = 'synthetic' if isSynthetic else 'real'
##    short_filename = dataset_name + '_test'
##
##    processor = ImageProcessor.ImageProcessor(image_path, dataset_name, image_type)
##    processor.isMLO = True
##    processor.isCSAW = False
##    processor.short_filename = short_filename
##    processor.do_intermediate_plots = False
##    processor.do_plots = False
##    processor.build_angle_gradients ()
##    adict = {'angle_gradient':processor.binned_angle_gradients,
##             'pixels_along_edge':processor.pixels_along_edge,
##             'angles_along_edge':processor.angles_along_edge}
####################################################################

##################################
## Import packages
##################################
import os
import warnings
import numpy as np
from math import atan2
from copy import deepcopy
import matplotlib.pyplot as plt

import scipy
import scipy.stats
from scipy import ndimage

from skimage import io
from skimage import morphology, measure
from skimage.morphology import (disk)
from skimage.filters import gaussian

from offsets import get_offsets

##################################
## Define constants
##################################
## +----------------------------------
## | For image processing
## +----------------------------------
default_is_CSAW = False
default_mask_threshold = 0
default_small_object_size = 0.1    # Size of the smallest object to be removed [0, 1]
default_smooth_window_size = 10
default_nSteps_stop_connect_skeleton = 3

## MLO edge removal
default_is_MLO = True
default_MLO_row_buffer_pixels = 5         # Buffer for removing top edge pixels
default_MLO_row_mode_count_threshold = 20 # Threshold for # row pixels with row mode
default_MLO_column_buffer_percentage = 10 # Buffer in % for removing left vertical edge pixels

## +----------------------------------
## | For defining angular gradient features
## +----------------------------------
default_edge_bin_counts = 64
default_distribution_bin_counts = 16

## +----------------------------------
## | For do flags
## +----------------------------------
default_do_plots = False
default_do_reprocess = True
default_do_intermediate_plots = False

##################################
## Define short functions
##################################
def get_angle (image_shape, current, new):
    '''Calculate the angle between two points in the image.

       Inputs
       ------
        image_shape: tuple
            Shape of the image (height, width).
        current: tuple
            Current pixel coordinates (row, column).
        new: tuple
            New pixel coordinates (row, column).

       Outputs
       -------
        angle: float
            Angle in radians between the two points.
    '''
    return atan2((image_shape[0] - new[0]) - (image_shape[0] - current[0]), new[1] - current[1])

##################################
## Define ImageProcessor class
##################################
class ImageProcessor (object):

    def __init__ (self, image_path, dataset_name, image_type, output_path=None):

        '''Initialize the image processor class.

           Inputs
           ------
            image_path: str
                Path to the input image file.
            dataset_name: str
                Name of the dataset (e.g., 'vindr', 'csaw').
            image_type: str 
                Type of the image: 'real' or 'synthetic'.
            output_path: str
                Path to save the plots.
        '''

        ## Inputs
        self._image_path = self._check_path (image_path, create=False)
        self._dataset_name = self._check_str ("dataset_name", dataset_name)
        self._image_type = self._check_image_type (image_type) 
        self._output_path = None if output_path is None else \
                            self._check_path (output_path, create=True)

        ## Outputs
        self._breast_area = None
        self._angles_along_edge = None
        self._pixels_along_edge = None
        self._binned_angle_gradients = None
        self._normalized_gradient_distribution = None

        ## Intermediate variables
        self._image = io.imread (self.image_path, as_gray=True)
        self._skeleton = None

        self._small_object_size = default_small_object_size
        self._MLO_row_buffer_pixels = default_MLO_row_buffer_pixels
        self._MLO_row_mode_count_threshold = default_MLO_row_mode_count_threshold
        self._MLO_column_buffer_percentage = default_MLO_column_buffer_percentage
        self._nSteps_stop_connect_skeleton = default_nSteps_stop_connect_skeleton
     
        ## Parameters that users can modify
        self._isMLO = default_is_MLO
        self._isCSAW = default_is_CSAW
        #  need to use output filename in case the image basename is too long
        self._short_filename = os.path.basename (image_path)[:100]

        self._edge_bin_counts = default_edge_bin_counts
        self._distribution_bin_counts = default_distribution_bin_counts
        self._mask_threshold = default_mask_threshold
        self._smooth_window_size = default_smooth_window_size

        self._do_reprocess = default_do_reprocess
        self._do_plots = default_do_plots
        self._do_intermediate_plots = default_do_intermediate_plots

        self._output_plot_path = None
        self._output_intermediate_plot_path = None 

    ## +---------------------------------
    ## | Properties
    ## +---------------------------------    
    @property
    def image_path (self):
        return self._image_path
    @property
    def dataset_name (self):
        return self._dataset_name        
    @property
    def image_type (self):
        return self._image_type
    @property
    def breast_area (self):
        return self._breast_area     
    @property
    def angles_along_edge (self):
        return self._angles_along_edge
    @property
    def pixels_along_edge (self):
        return self._pixels_along_edge        
    @property
    def binned_angle_gradients (self):
        return self._binned_angle_gradients
    @property
    def normalized_gradient_distribution (self):
        return self._normalized_gradient_distribution

    @property
    def image (self):
        return self._image
    @property
    def skeleton (self):
        return self._skeleton

    @property
    def isMLO (self):
        return self._isMLO
    @isMLO.setter
    def isMLO (self, value):
        self._isMLO = self._check_boolean ("isMLO", value)

    @property
    def isCSAW (self):
        return self._isCSAW
    @isCSAW.setter
    def isCSAW (self, value):
        self._isCSAW = self._check_boolean ("isCSAW", value)

    @property
    def output_path (self):
        return self._output_path
    @output_path.setter
    def output_path (self, value):
        self._output_path = self._check_path (value, create=True)

    @property
    def short_filename (self):
        return self._short_filename
    @short_filename.setter
    def short_filename (self, value):
        self._short_filename = self._check_str ("short_filename", value)

    @property
    def mask_threshold (self):
        return self._mask_threshold
    @mask_threshold.setter
    def mask_threshold (self, value):
        self._mask_threshold = self._check_positive_integer ("mask_threshold", value)

    @property
    def edge_bin_counts (self):
        return self._edge_bin_counts
    @edge_bin_counts.setter
    def edge_bin_counts (self, value):
        self._edge_bin_counts = self._check_positive_integer ("edge_bin_counts", value)

    @property
    def distribution_bin_counts (self):
        return self._distribution_bin_counts
    @distribution_bin_counts.setter
    def distribution_bin_counts (self, value):
        self._distribution_bin_counts = self._check_positive_integer ("distribution_bin_counts", value)

    @property
    def smooth_window_size (self):
        return self._smooth_window_size
    @smooth_window_size.setter
    def smooth_window_size (self, value):
        self._smooth_window_size = self._check_positive_integer ("smooth_window_size", value)

    @property
    def do_reprocess (self):
        return self._do_reprocess
    @do_reprocess.setter
    def do_reprocess (self, value):
        self._do_reprocess = self._check_boolean ("do_reprocess", value)

    @property
    def do_plots (self):
        return self._do_plots
    @do_plots.setter
    def do_plots (self, value):
        self._do_plots = self._check_boolean ("do_plots", value)
        ## If not do_plots, do not create the plot path.
        if not value: return            
        if self._output_path is None: 
            raise ValueError ("Please provide output to save plots via `processor.output_path = '/path/to/save/outputs/'`.")
        ## Create a sub-folder to store plots if it does not exist.
        self._output_plot_path = os.path.join(self.output_path, 'plots/')
        os.makedirs (self._output_plot_path, exist_ok=True)

    @property
    def do_intermediate_plots (self):   
        return self._do_intermediate_plots
    @do_intermediate_plots.setter
    def do_intermediate_plots (self, value):
        self._do_intermediate_plots = self._check_boolean ("do_intermediate_plots", value)
        ## If not do_intermediate_plots, do not create the intermediate plot path.
        if not value: return
        if self._output_path is None: 
            raise ValueError ("Please provide output to save intermediate plots via `processor.output_path = '/path/to/save/outputs/'`.")
        ## Create a sub-folder to store intermediate plots if it does not exist.
        self._output_intermediate_plot_path = os.path.join(self.output_path, 'intermediate_plots/'+self._short_filename+'/')
        os.makedirs (self._output_intermediate_plot_path, exist_ok=True)

    ## +-------------------------------------
    ## | Private functions to check inputs
    ## +-------------------------------------   
    def _check_image_type (self, image_type):

        ''' Check if the image type is valid.

            input
            ------
                image_type: str
                    Type of the image: 'real' or 'synthetic'.

            Outputs
            -------
                image_type: str
                    The input image type if it is valid.
        '''

        if image_type not in ['real', 'synthetic']:
            raise ValueError("Invalid image type. Must be 'real' or 'synthetic'.")
        return image_type

    def _check_str (self, variable, string_value):

        ''' Check if the image type is valid.

            input
            ------
                string_value: str
                    Input string value to be checked.
                variable: str
                    Name of the variable for error message.

            Outputs
            -------
                string_value: str
                    The input string value if it is a valid string.
        '''

        if not isinstance (string_value, str):
            raise ValueError("Invalid string type. {0} must be a string".format (variable))
        return string_value.lower()

    def _check_path (self, apath, create=False):

        ''' Check if the image type is valid.

            input
            ------
                apath: str
                    Path to the image file or directory.
                create: bool    
                    If True, create the directory if it does not exist.                    

            Outputs
            -------
                apath: str
                    The input path if it is a valid string and exists.
        '''

        if not isinstance(apath, str):
            raise ValueError("Path must be a string.")
        if not os.path.exists (apath):
            if create:
                os.makedirs (apath, exist_ok=True)
            else:
                raise ValueError("Path does not exist: {}".format (apath))
        return apath

    def _check_positive_integer (self, variable, value):

        ''' Check if the input value is a positive integer. =

            Inputs
            ------
                variable: str
                    Name of the variable for error message.
                value: int
                    Value to be checked.

            Outputs
            -------
                value: int
                    The input value if it is a positive integer.
        '''

        if not isinstance(value, int) or value <= 0:
            raise ValueError("{} must be a positive integer.".format (variable))
        return value

    def _check_boolean (self, variable, value):

        ''' Check if the input value is a boolean.

            Inputs
            ------
                variable: str
                    Name of the variable for error message.
                value: bool 
                    Value to be checked.

            Outputs
            -------
                value: bool
                    The input value if it is a boolean.
        '''

        if not isinstance(value, bool):
            raise ValueError("{} must be a boolean value.".format (variable))
        return value        

    ## +----------------------------------------------
    ## | Functions to detect edges
    ## +----------------------------------------------
    def _plot_2Darray (self, array2D, title, fullfilename, edge_pixels=None):
        
        ''' Plot a normalized 2D array and save it as an image file.

            Inputs
            ------
                array2D: np.ndarray
                    2D array to be plotted.
                title: str
                    Title of the plot.
                fullfilename: str
                    Full filename of the image plotted including path.
                edge: np.ndarray, optional
                    Coordinates of edge pixels. If provided, overlay on top of the image.
        '''

        ## If array2D is an array of boolean, convert it to int.
        if array2D.dtype == bool: array2D = array2D.astype(int)
        normalized_array2D = (array2D - np.min(array2D)) / (np.max(array2D) - np.min(array2D))

        plt.imshow (normalized_array2D, cmap='gray')

        ## Overlay edge pixels on top of the image
        if edge_pixels is not None:
            plt.scatter(edge_pixels[:, 1], edge_pixels[:, 0], color='red', s=1)

        plt.title (title)
        plt.savefig (fullfilename, bbox_inches='tight', dpi=600)
        plt.close()

    def _get_component (self, binary):

        ''' Get the largest connected component in the binary mask. The binary
            mask should have at least one connected component.

            Inputs
            ------  
            binary: np.ndarray
                Binary mask of the image.

            Outputs
            -------
            binary: np.ndarray
                Binary mask containing only the one and only connected component
        '''

        # Get the label ID for each pixel. Since there is only one component,
        # labels will be 1 for the component and 0 for the background.
        labels = measure.label(binary)

        # Asserting that there is at least one connected component in the binary mask.
        if np.max(labels) == 0:
            raise ValueError("No connected component found in the binary mask in\n"
                             "   {}".format(self.image_path))

        # Keep the largest component
        return labels == np.argmax(np.bincount(labels.flat)[1:]) + 1 

    def _handle_MLO (self, skeleton):

        ''' Handle the MLO image type by removing the left vertical and
            top horizontal borders. This is not needed for CC view. Other
            modalities may need it depending on the image itself. Customized
            edge removal should be done by users if needed.

            inputs
            ------
                skeleton: np.ndarray
                    Edge of breast with left vertical and top horizontal borders.
            
            outputs
            -------
                skeleton: np.ndarray
                    Edge of breast with only the curvature of the breast.   
        '''

        ## Only clean up the skeleton if it is an MLO image.
        if not self.isMLO: return skeleton

        # Get the total number of pixels in rows and columns
        nrows, ncols = skeleton.shape

        ## 1. Remove the left vertical edge
        #     Get the column indices of all edge pixels
        _, column_idx = np.where (skeleton)
        #     Get the mode of column coordinates. The vertical pixels
        #     will likely have the same / similar column index. 
        column_mode = np.uint16 (scipy.stats.mode(column_idx)[0])  
        #     Define the range of column indices to be removed. It is 
        #     set to be about 1/10-th of the image width. For example,
        #     given a 512 x 512, the width is 512. 10% is 51.2, i.e.
        #     roughly 25 pixels before and after the mode coordinate.
        one_sided_buffer = int (ncols * self._MLO_column_buffer_percentage / 100 // 2)
        #     Replace the pixels in the column index range with 0.
        skeleton[:, column_mode - one_sided_buffer:column_mode + one_sided_buffer] = 0

        ## 2. Remove the top horizontal edge
        #     Get the row indices of top quarter of edge pixels
        row_idx, _ = np.where (skeleton[:nrows // 4, :])
        #     Get the mode of row coordinates. The horizontal pixels
        #     will likely have the same / similar row index.
        row_mode, row_mode_counts = scipy.stats.mode(row_idx)
        #     Do not remove the top edge if the # pixels with the mode
        #     value is too low. Arbitarily set the threshold of 20 pixels.
        if row_mode_counts < self._MLO_row_mode_count_threshold: return skeleton
        #     For those that have enough pixels with the mode value,
        #     remove the top edge pixels by setting them to 0.
        row_mode = np.uint16 (row_mode)
        skeleton[:row_mode + self._MLO_row_buffer_pixels, :] = 0 

        return skeleton
 
    def detect_edges (self):

        ''' Detect edges in the image using morphological operations.
            This function performs the following steps:
                1. Read the image.
                2. Smooth the image using Gaussian filter.
                3. Apply a threshold to create a binary mask.
                4. Fill holes inside the binary mask.
                5. Dust up the mask by removing small objects.
                6. Get the largest connected component in the binary mask.
                7. Skeletonize the binary mask to get a 1-pixel-wide edge.
                8. If MLO, remove the left vertical and top horizontal edges.
                9. Remove any small holes that are less than X% of the total area of the breast.
                10. If multiple components, connect them.
            
            Breast area and skeleton (1-pixel-wide edge border) are stored 
            as internal variable.
        '''

        # 1. Read the image
        image = self._image
        if self._do_intermediate_plots: self._plot_2Darray (image, "Original", self._output_intermediate_plot_path+"01_original_image.png")
        #    If CSAW synthetic data, cut the top and last 4 rows to keep the
        #    image dimensions the same as the CSAW real data. The group that
        #    generated the CSAW synthetic data added this paddings.
        if self.isCSAW and self.image_type == 'synthetic':
            image = image[4:-4, :]
            if self._do_intermediate_plots:
                self._plot_2Darray (image, "CSAW synthetic cropped", self._output_intermediate_plot_path+"01.5_cropped_image.png")

        # 2. Smooth the image
        image = np.uint8 (gaussian(image, sigma=0.5) * 255)
        if self._do_intermediate_plots: self._plot_2Darray (image, "Gaussian smoothed", self._output_intermediate_plot_path+"02_smoothed_image.png")

        # 3. Apply threshold: Pixels with values above the threshold are considered as foreground.
        binary = image > self.mask_threshold
        if self._do_intermediate_plots: self._plot_2Darray (binary, "Thresholded", self._output_intermediate_plot_path+"03_thresholded_binary.png")

        # 4. Fill holes inside the binary mask
        #       0 0 0 0 0       0 0 0 0 0
        #       0 1 1 1 0       0 1 1 1 0
        #       0 1 0 1 0  -->  0 1 1 1 0
        #       0 1 1 1 0       0 1 1 1 0
        #       0 0 0 0 0       0 0 0 0 0
        binary = ndimage.binary_fill_holes (binary)
        if self._do_intermediate_plots: self._plot_2Darray (binary, "Holes filled", self._output_intermediate_plot_path+"04_filled_binary.png")

        # 5. Dust up binary mask by removing small objects using erosion then dilation
        #       https://en.wikipedia.org/wiki/Opening_(morphology)
        #    For pixels along image border, use the minimum value of the surrounding pixels.
        binary = morphology.opening (binary, footprint=disk(10), mode='min')
        if self._do_intermediate_plots: self._plot_2Darray (binary, "Dusted", self._output_intermediate_plot_path+"05_dusted_binary.png")

        # At this point, the binary mask should have at least one connected component.
        # 6. Get largest connected component in the binary mask
        binary = self._get_component (binary)
        if self._do_intermediate_plots: self._plot_2Darray (binary, "Largest", self._output_intermediate_plot_path+"06_largest_binary.png")
        #    Before skeletonize the edges, get the area of this largest component
        self._breast_area = np.count_nonzero(binary) / (binary.shape[0] * binary.shape[1])

        # 7. Skeletonize the binary mask to get the edges
        #    First, create a slightly small (eroded) mask. The difference between the two
        #    masks will be the border pixels.
        mask_border = binary - morphology.binary_erosion(binary, disk(2), mode='min').astype (int)
        #    Skeletonize the border mask to get the 1-pixel-width edges.
        skeleton = morphology.skeletonize (mask_border).astype (int)
        if self._do_intermediate_plots: self._plot_2Darray (skeleton, "Skeleton", self._output_intermediate_plot_path+"07_skeleton_binary.png")

        # 8. Handle MLO images by removing the left vertical and top horizontal edges
        if self.isMLO:
            skeleton = self._handle_MLO (skeleton)
            if self._do_intermediate_plots:
                self._plot_2Darray (skeleton, "MLO edges removed", self._output_intermediate_plot_path+"08_skeleton_MLO_edges_removed.png")

        # 9. Remove any small holes that are less than X% of the total area of the breast.
        #    In 2D morphology, connectivity = 1 refers to the direct 4-neighborhood,
        #                                     2 includes diagnoal pixels 8-neighborhood
        #    `min_size` here is the smallest allowable size. If set to be 0, nothing will be
        #    removed. Problem was seen in some images that the min_size of 50 is too large
        #    that it removes the entire skeleton. Also note that this function only works
        #    with numerical values and not boolean arrays.
        skeleton = morphology.remove_small_objects (skeleton, connectivity=1,
                                                    min_size=skeleton.shape[0] * self._small_object_size)
        if self._do_intermediate_plots: self._plot_2Darray (skeleton, "Skeleton holes removed", self._output_intermediate_plot_path+"09_skeleton_holes_removed.png")
        
        # Check how many connected components are in the skeleton.
        n_skeletons = measure.label(skeleton).max()
        self._skeleton = skeleton
        # When only 1 connected component remains in the skeleton, return it!
        if n_skeletons == 1: return
        # When none, raise an error.
        if n_skeletons == 0:
            raise ValueError("No connected component found in the skeleton in\n"
                             "   {}".format(self.image_path))

        # At this point, more than 1 connected components exist and will be connected.
        # 10. Before starting the while loop, pad the skeleton with zeros to respect
        #     boundaries that are on the image edges.
        skeleton_padded = np.zeros((skeleton.shape[0] + 2, skeleton.shape[1] + 2), dtype=np.uint8)
        skeleton_padded[1:-1, 1:-1] = skeleton
        #     Connect the components by closing with increasingly large disk sizes.
        n_steps = 0
        while n_skeletons > 1 and n_steps < self._nSteps_stop_connect_skeleton:
            skeleton_padded = morphology.binary_closing(skeleton_padded, disk(2**n_steps), mode="ignore")
            skeleton_padded = morphology.skeletonize(skeleton_padded).astype (int)
            n_skeletons = measure.label(skeleton_padded).max()
            n_steps += 1
        skeleton = skeleton_padded[1:-1, 1:-1]
        #      Raise warning if it still has more than 1 connected component after the loop.
        if n_skeletons > 1:
            #warnings.warn("More than one connected component found in the skeleton. Keeping the largest component.\n"
            #              "   {}\n".format(self.image_path))
            skeleton = self._get_component (skeleton)  

        if self._do_intermediate_plots: self._plot_2Darray (skeleton, "Skeleton connected", self._output_intermediate_plot_path+"10_skeleton_connected.png")
        self._skeleton = skeleton

    ## +----------------------------------------------
    ## | Functions to walk along edge
    ## +----------------------------------------------
    def _plot_arrows_along_edge (self, angles_along_edge, xvalues_along_edge,
                                 yvalues_along_edge, title, fullfilename, bin_edges=None):

        ''' Plot the angles along the edge as arrows.

            Inputs
            ------
                angles_along_edge: np.ndarray
                    Angles along the edge in degrees.
                xvalues_along_edge: np.ndarray
                    X coordinates of the edge pixels.
                yvalues_along_edge: np.ndarray
                    Y coordinates of the edge pixels.
                title: str
                    Title of the plot.
                fullfilename: str
                    Full filename of the image plotted including path.
        '''

        #     Define arrow origin points
        x_orig = xvalues_along_edge
        y_orig = yvalues_along_edge
        #     Arrow directions: angles are clockwise from the horizontal axis
        #     and rotated for plotting purposes
        angles_in_radians = np.deg2rad (angles_along_edge)
        dx = np.cos(2*np.pi + angles_in_radians)
        dy = np.sin(2*np.pi + angles_in_radians)
        dx_rot = -dy
        dy_rot = dx
        #     Create color values using a colormap
        cmap = plt.get_cmap('hsv')  
        colors = cmap(np.linspace(0, 1, len(angles_along_edge)))

        #     Plot
        plt.figure(figsize=(6, 6))
        #       Arrows with angles depending on the angle of the edge pixel
        plt.quiver (x_orig, y_orig, dx_rot, dy_rot, color=colors, angles='xy', scale_units='xy',
                    scale=0.9, width=0.001, headlength=2, headwidth=1)
        #       Binning of the edges
        if bin_edges is None:
            plt.grid(True)
        if bin_edges is not None:
            for x in bin_edges+x_orig[0]:
                plt.plot([x, x], [200, 350], color='black', linewidth=0.25)

        plt.axis('equal')
        plt.title(title)
        #     Save the plot
        plt.savefig (fullfilename, bbox_inches='tight', dpi=900)
        plt.close()

    def _plot_walk_along_edge (self, fullfilename):

        ''' Plot the edge pixels and angles along the edge.

            Inputs
            ------
                fullfilename: str
                    Full filename of the image plotted including path.
        '''
       
        ## Only do plot if asked
        if not self._do_plots: return

        ## If angles were not already calculated, raise an error.
        if self._angles_along_edge is None:
            raise ValueError("No angles found. Please run `walk_along_edge()` first.")

        ## 1. Plot the original image with the edge pixels overlayed.
        self._plot_2Darray (self._image, "Image with edge pixels",
                            self._output_plot_path + self.short_filename + "_image_with_edge_pixels.png",
                            edge_pixels=self._pixels_along_edge)

        ## 2. Plot angles as arrows
        xvalues = self._pixels_along_edge[:, 0]
        yvalues = self._pixels_along_edge[:, 1]
        angles = self._angles_along_edge
        title = "Angles along the edge for {}".format(self.short_filename)
        self._plot_arrows_along_edge (angles, xvalues, yvalues, title, fullfilename, bin_edges=None)

    def _get_neighbors (self, grid_size, current_pixel, current_skeleton):

        ''' Get the neighbors of the current pixel in the skeleton.

            Inputs
            ------
                grid_size: int
                    Size of the neighborhood grid (3 or 5).
                current_pixel: np.ndarray
                    Coordinates of the current edge pixel.
                current_skeleton: np.ndarray
                    1-pixel-wide binary mask of the image.
                    This skeleton is updated by turning off the current pixel.

            Outputs
            -------
                neighbor_indices: list
                    List of coordinates of the neighboring pixels.
                neighbor_bools: list
                    List of boolean values indicating whether the neighboring pixels are edge pixels.
        '''

        # Get the coordinates of the current pixel
        row, col = current_pixel
        image_shape = current_skeleton.shape

        # Define the neighborhood offsets 
        offsets = get_offsets (grid_size, image_shape, current_pixel)

        # Collect neighbors' pixels indices and binary values
        neighbor_indices, neighbor_bools = [], []
        for dr, dc in offsets:
            index = [row + dr, col + dc]
            neighbor_indices.append(index)
            neighbor_bools.append(current_skeleton[index[0], index[1]])      

        return np.array (neighbor_indices), np.array (neighbor_bools).astype (bool)

    def _get_next_edge_pixel_in_grid (self, grid_size, current_pixel, current_skeleton, current_angle):

        ''' Get the next edge pixel from the current pixel in a grid of size grid_size x grid_size.
            The next pixel is the one that is connected to the current pixel and has a non-zero value
            in the skeleton array.
            Inputs
            ------
                grid_size: int
                    Size of the neighborhood grid (3 or 5).
                current_pixel: np.ndarray
                    Coordinates of the current edge pixel.
                current_skeleton: np.ndarray
                    1-pixel-wide binary mask of the image.
                current_angle: float
                    Current angle in radians from the horizontal axis.

            Outputs
            -------
                angle: float
                    Angle from current to the next edge pixel in radians.
                new_pixel: np.ndarray
                    Coordinates of the next edge pixel.
        '''

        image_shape = current_skeleton.shape

        # Collect neighbors' pixels indices and binary values in 3 x 3 grid
        neighbor_indices, neighbor_bools = self._get_neighbors (grid_size, current_pixel, current_skeleton)

        # Count number of `true` neighbors i.e. edge pixels around the current pixel
        nEdgePixels = len (neighbor_bools[neighbor_bools])

        # Simply return the angle when exactly one edge pixel is found
        if nEdgePixels == 1:
            new_pixel = neighbor_indices[neighbor_bools][0]
            angle = get_angle (image_shape, current_pixel, new_pixel)
            return angle, new_pixel

        # If multiple edge pixels are found, return the one with the smallest angle.
        # Here, angle is measured with respect to the horizonal axis. Negative angle
        # is clockwise (3rd and 4th quadrants), positive angle is counterclockwise.
        # By "smallest", we mean the angle in the 3rd and 4th quadrants because of
        # the curvature of the breast in MLO and CC where edges only goes down. But
        # other modalities may include up-going edges (e.g. any round anatomy).
        if nEdgePixels > 1:
            neighbors_with_edge = neighbor_indices[neighbor_bools]
            angles = np.array ([get_angle(image_shape, current_pixel, np.array(image_idx))
                                for image_idx in neighbors_with_edge])
            min_angle_index = np.argmin(current_angle - angles)
            new_pixel = neighbor_indices[neighbor_bools][min_angle_index]
            angle = angles[min_angle_index]
            return angle, new_pixel

        # At this point, no edge pixel is found around the current pixel.
        return None, None

    def _get_next_edge_pixel (self, current_pixel, current_skeleton, current_angle):

        ''' Get the next edge pixel from the current pixel.
            The next pixel is the one that is connected to the
            current pixel and has a non-zero value in the skeleton array.

            Inputs
            ------
                current_pixel: np.ndarray
                    Coordinates of the current edge pixel.
                current_skeleton: np.ndarray
                    1-pixel-wide binary mask of the image.
                    This skeleton is updated by turning off the current pixel.
                current_angle: float
                    Current angle in radians from the horizontal axis.

            Outputs
            -------
                angle: float   
                    Angle from current to the next edge pixel in radians.
                new_pixel: np.ndarray
                    Coordinates of the next edge pixel.
                new_skeleton: np.ndarray
                    Updated skeleton with the current pixel turned off.
        '''

        image_shape = current_skeleton.shape

        # Update skeleton by turning off the current pixel..
        current_skeleton[current_pixel[0], current_pixel[1]] = 0

        # Search in a 3 x 3 grid around the current pixel. If non-None angle /
        # new pixel are found. there exist an edge pixel around the current pixel.
        angle, new_pixel = self._get_next_edge_pixel_in_grid (3, current_pixel, current_skeleton, current_angle)
        if angle is not None: return angle, new_pixel, current_skeleton

        # At this point, no edge pixel is found around the current pixel.
        # If it is at the end of the edge i.e. the number of remaining available edge
        # pixel in the skeleton is less than 1/10-th of the image size, return None
        # to signal stop tracking.
        if np.sum (current_skeleton) <= image_shape[0] // 10:
            return None, None, current_skeleton

        # Now, there is no edge pixel around the 3 x 3 neighbor *and* it is at the
        # beginning of the tracking. Expand the search area to a 5 x 5 grid. Again,
        # If non-None angle / new pixel are found. there exist an edge pixel around
        # the current pixel.
        angle, new_pixel = self._get_next_edge_pixel_in_grid (5, current_pixel, current_skeleton, current_angle)
        if angle is not None: return angle, new_pixel, current_skeleton
        
        # None of the neighbors in the 5 x 5 grid or 3 x 3 is an edge pixel.
        #raise ValueError("No edge pixel found around the current pixel {0} in the skeleton\n"
        #                    "   {1}".format(current_pixel, self.image_path))        
        return None, None, current_skeleton
            
    def walk_along_edge (self):

        ''' The key function of this code is to ...
                * find the next neighbor
                * calculate the angle of the edge pixel
            We cannot use typical functions to find next neighbor
            because the edge pixels are not necessarily connected
            in a straight line. This function is to order the edge
            pixels found in `skeleton`.

            angles_along_edge and pixels_along_edge are stored as
            internal variables.

            Outputs
            -------
                angles_along_edge: list
                    Array of angles in radians along the edge.
                pixels_along_edge: list
                    Array of coordinates of the edge pixels.
        '''

        ## Detect edges if not done already
        if self._skeleton is None: self.detect_edges ()
        total_n_edge_pixels = np.count_nonzero(self._skeleton)

        ## Initialize the lists to store angles and pixels along the edge.
        angles_along_edge = []
        pixels_along_edge = []

        ## Start with the first pixel
        this_angle = 0
        this_skeleton = deepcopy (self._skeleton)
        this_edge_pixel = np.column_stack(np.where(this_skeleton))[0]

        ## Loop until all edge pixels are processed
        ## Pixels_along_edge cannot have more than the total number of edge pixels.
        while len (pixels_along_edge) < total_n_edge_pixels:

            ## Get the next edge pixel
            next_angle, next_edge_pixel, next_skeleton = self._get_next_edge_pixel (this_edge_pixel, this_skeleton, this_angle)
            if next_edge_pixel is None: break

            ## Store the angle and pixel coordinates
            angles_along_edge.append(next_angle)
            pixels_along_edge.append(next_edge_pixel)

            ## Move to the next edge pixel
            this_edge_pixel = next_edge_pixel
            this_skeleton = next_skeleton
            this_angle = next_angle

        self._angles_along_edge = np.array(angles_along_edge) * 180 / np.pi  # Convert radians to degrees
        self._pixels_along_edge = np.array(pixels_along_edge)

        if self._do_plots:
            fullfilename = os.path.join(self._output_plot_path, self.short_filename + "_angles_along_edge.png")
            self._plot_walk_along_edge (fullfilename)

    ## +----------------------------------------------
    ## | Functions to build angular gradient features
    ## +----------------------------------------------
    def _smooth_angles_along_edge (self):

        ''' Smooth the angles along the edge using a moving average filter.
            This function is used to smooth the angles along the edge
            before calculating the gradient trajectory. 

            If it is an MLO image, vertical angles are identified as chest
            wall. Only the angles around the breast are returned.

            Outputs
            -------
                smoothed_angles: np.ndarray
                    Smoothed angles along the edge in degrees.
                smoothed_pixels: np.ndarray
                    Coordinates of edge pixels with smoothed angles
        '''

        ## If angles were not already calculated, do it now.
        if self._angles_along_edge is None: self.walk_along_edge ()

        ## If the number of edge pixels is less than self._edge_bin_counts, 
        ## raise a warning, and no gradient is calculated.
        if len(self._angles_along_edge) < self._edge_bin_counts:
            warnings.warn("Number of edge pixels ({}) is less than the number of edge bins ({}). No gradient calculated.".format(
                len(self._angles_along_edge), self._edge_bin_counts))
            return None
        
        ## Smooth the angles along the edge using a moving average filter.
        window = np.ones (self._smooth_window_size) / self._smooth_window_size
        smoothed_angles_along_edge = np.convolve (self._angles_along_edge % 360, window, mode='same')

        ## Typically, start and end points are the first and last points of the trajectory.
        if not self.isMLO:
            return smoothed_angles_along_edge, self._pixels_along_edge

        ## Special treatments if MLO
        #  The last 20 pixels are not useful for MLO images.
        smoothed_angles_along_edge = smoothed_angles_along_edge[:-20]

        #  Starting point
        #    Identify the breast attachment points from 1st quarter of the trajectory.
        first_quarter = smoothed_angles_along_edge[:len(smoothed_angles_along_edge) // 4]
        #    Start point is near verticals (270 degrees) with a 10 degree uncertainty.
        try:
            start_point = np.where(np.logical_and(first_quarter >= 260, first_quarter <= 280))[0][0]
        except:
            #    Elim (6/17): some images don't have vertical edge at the start.
            #                 In this case, use the first pixel.
            start_point = 0

        #  Ending point
        end_point = -1
        #    Identify the breast attachment points from last quarter of the trajectory.
        fourth_quarter = smoothed_angles_along_edge[3 * len(smoothed_angles_along_edge) // 4:]
        #    Check if the last 20 pixels are chest wall
        is_chest_wall = np.mean(fourth_quarter[-20:] >= 225)
        #    If it is chest wall, 
        if is_chest_wall:
            # For MLO, if +ve angle difference, the next pixel is more counterclockwise than the
            # current pixel i.e. curving out towards the 4th quadrant here, end point is defined
            # to be the first angle that curve out by 2 degree.
            end_points = np.where(np.diff(fourth_quarter) >= 2)[0]
            # If multiple end point candidates, choose the middle one.
            if len(end_points) >= 1: 
                end_point = end_points[len(end_points) // 2] + 3 * len(smoothed_angles_along_edge) // 4

        smoothed_angles = smoothed_angles_along_edge[start_point:end_point]
        smoothed_pixels = self._pixels_along_edge[start_point:end_point]
        return smoothed_angles, smoothed_pixels

    def _get_edge_bin_widths (self, n_angles):
    
        ''' Get the bin widths for the edge pixels based on the number of angles.
            The middle bins are of equal width, while the first and last bins may
            have different widths depending on whether the number of angles can be
            equally divided by the number of edge bins. 

            Inputs
            ------
                n_angles: int
                    Number of angles along the edge.

            Outputs
            -------
                bin_widths: list
                    List of bin widths for the edge pixels.
        '''

        ## Edge pixels are grouped into N bins (based on self._edge_bin_counts). The middle
        ## bins are of equal width, while the first and last bins may have different widths.
        n_middle_bins = self._edge_bin_counts - 2
        middle_bin_width = (n_angles - int(n_angles % self._edge_bin_counts)) // n_middle_bins

        ## The first and last bins may have different widths depending on whether the number
        ## of angles can be equally divided by the number of edge bins.
        first_bin_width = n_angles // self._edge_bin_counts
        last_bin_width = n_angles // self._edge_bin_counts        
        if n_angles % self._edge_bin_counts != 0:
            split = np.floor ((n_angles % self._edge_bin_counts) / 2)
            first_bin_width = int(n_angles % self._edge_bin_counts - split)
            last_bin_width = int(split)

        ## Create a list of bin widths
        return [first_bin_width] + [middle_bin_width]*n_middle_bins + [last_bin_width]

    def build_angle_gradients (self):

        ''' Build the angle gradients along the edge. Angle gradients are
            calculated by grouping the edge pixels into bins (default 64).
            Within each bin, the angle gradients are summed up, which is
            essentially the average rate of change in angle within the bin.
        '''

        ## If angles were not already calculated, do it now.
        if self._angles_along_edge is None: self.walk_along_edge ()

        ## Get the angle gradient from a smoothed angle profile.
        smoothed_angles, smoothed_pixels = self._smooth_angles_along_edge ()
        angle_gradients = np.gradient (smoothed_angles)

        ## Determine the bin widths and edge values for the grouped edge pixels.
        edge_bin_widths = self._get_edge_bin_widths (len(smoothed_angles))
        cumsum_bin_widths = [0] + list(np.cumsum(np.array(edge_bin_widths)))

        ## Within each bin, sum the angle gradients.
        binned_angle_gradients = [np.sum(angle_gradients[cumsum_bin_widths[idx]: cumsum_bin_widths[idx + 1]])
                                  for (idx, ii) in enumerate(cumsum_bin_widths[:-1])]
        binned_angle_gradients = np.array (binned_angle_gradients)
        binned_angle_gradients = binned_angle_gradients[~np.isnan (binned_angle_gradients)]
        self._binned_angle_gradients = binned_angle_gradients

        ## Plot if asked
        if self._do_plots:
            xvalues = smoothed_pixels[:, 0]
            yvalues = smoothed_pixels[:, 1]
            angles = smoothed_angles
            title = "Angles along the binned edge for {}".format(self.short_filename)
            fullfilename = os.path.join(self._output_plot_path, self.short_filename + "_angles_along_edge_with_bin_edges.png")
            self._plot_arrows_along_edge (angles, xvalues, yvalues, title, fullfilename, bin_edges=cumsum_bin_widths)
