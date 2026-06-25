## 
## This file contains utility functions for calculating offsets
## when looking for neighbors along the edge. 3 x 3 is a common
## 8-neighbor grid, and 5 x 5 is a 16-neighbor grid. Special
## cases are handled for pixels along the borders and at corners.
## There are probably smarter ways to do this, but the current
## implementation can be visualized via spacing between offset
## coordinates.
##################################################################

##################################
## Define constants
##################################

## +--------------------------------------
## | 3 x 3 grid of offsets 
## +--------------------------------------
## Typical middle: not next to border
middle_3x3offsets = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),           (0, 1),
                      (1, -1),  (1, 0),  (1, 1)]

## Along borders (not corners):
#    Along bottom border
bottom_middle_3x3offsets = [(-1, -1), (-1, 0), (-1, 1),
                             (0, -1),           (0, 1)]
#    Along top border 
top_middle_3x3offsets = [(0, -1),         (0, 1),
                         (1, -1), (1, 0), (1, 1)]
#    Along left border 
left_middle_3x3offsets = [(-1, 0), (-1, 1),
                                    (0, 1),
                           (1, 0),  (1, 1)] 
#    Along right border
right_middle_3x3offsets = [(-1, -1), (-1, 0),
                            (0, -1),  
                            (1, -1),  (1, 0)]

## Corners:
#    At top left
corner_top_left_3x3offsets = [        (0, 1),
                              (1, 0), (1, 1)]
#    At top right
corner_top_right_3x3offsets = [ (0, -1), 
                                (1, -1), (1, 0)]
#    At bottom left
corner_bottom_left_3x3offsets = [(-1, 0), (-1, 1),
                                           (0, 1)]
#    At bottom right
corner_bottom_right_3x3offsets = [(-1, -1), (-1, 0),
                                  (0, -1)          ]

## Full dictionary:
offsets3x3_dict = {'middle': middle_3x3offsets,
                   'bottom_middle': bottom_middle_3x3offsets,
                   'top_middle': top_middle_3x3offsets,
                   'left_middle': left_middle_3x3offsets,
                   'right_middle': right_middle_3x3offsets,
                   'corner_top_left': corner_top_left_3x3offsets,
                   'corner_top_right': corner_top_right_3x3offsets,
                   'corner_bottom_left': corner_bottom_left_3x3offsets,
                   'corner_bottom_right': corner_bottom_right_3x3offsets}

## +--------------------------------------
## | 5 x 5 grid of offsets 
## +--------------------------------------
## Typical middle: not next to border within 2 pixels
middle_5x5offsets = [(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
                     (-1, -2),                             (-1, 2),
                      (0, -2),                              (0, 2),
                     ( 1, -2),                              (1, 2),
                      (2, -2),  (2, -1),  (2, 0),  (2, 1),  (2, 2)] + middle_3x3offsets

## Along borders (not corners):
#    Along bottom border
bottom_middle_5x5offsets = [(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
                            (-1, -2),                             (-1, 2),
                             (0, -2),                              (0, 2)] + bottom_middle_3x3offsets
#    One pixel above bottom border
bottom_middle_above_5x5offsets = [(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
                                  (-1, -2),                             (-1, 2),
                                   (0, -2),                              (0, 2),
                                   (1, -2),                              (1, 2)] + middle_3x3offsets
#    Along top border
top_middle_5x5offsets = [ (0, -2),                          (0, 2),
                          (1, -2),                          (1, 2),
                          (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)] + top_middle_3x3offsets
#    One pixel below top border
top_middle_below_5x5offsets = [(-1, -2),                          (-1, 2),
                                (0, -2),                           (0, 2),
                                (1, -2),                           (1, 2),
                                (2, -2), (2, -1), (2, 0), (2, 1),  (2, 2)] + middle_3x3offsets
#    Along left border
left_middle_5x5offsets = [(-2, 0), (-2, 1), (-2, -2),
                                            (-1, -2),
                                             (0, -2),
                                             (1, -2),                                                
                           (2, 0),  (2, 1),  (2, -2)] + left_middle_3x3offsets
#    One pixel to the right of left border
left_middle_right_5x5offsets = [(-2, -1), (-2, 0), (-2, 1), (-2, -2),
                                                            (-1, -2),
                                                             (0, -2),
                                                             (1, -2),                                                
                                 (2, -1),  (2, 0),  (2, 1),  (2, -2)] + middle_3x3offsets
#    Along right border
right_middle_5x5offsets = [(-2, -2), (-2, -1), (-2, 0), 
                           (-1, -2),
                            (0, -2),
                            (1, -2),
                            (2, -2),  (2, -1),  (2, 0)] + right_middle_3x3offsets
#    One pixel to the left of right border
right_middle_left_5x5offsets = [(-2, -2), (-2, -1), (-2, 0), (-2, 1),
                                (-1, -2),                   
                                 (0, -2),                   
                                 (1, -2),                   
                                 (2, -2),  (2, -1),  (2, 0),  (2, 1)] + middle_3x3offsets

## Corners:
#    At top left
corner_top_left_5x5offsets = [                (0, 2),
                                              (0, 1),
                              (2, 0), (2, 1), (2, 2)] + corner_top_left_3x3offsets
#    One pixel below top left
corner_top_left_below_5x5offsets = [                 (-1, 2),
                                                      (0, 2),
                                                      (1, 2),
                                     (2, 0), (2, 1),  (2, 2)] + left_middle_3x3offsets
#    One pixel to the right of top left
corner_top_left_right_5x5offsets = [                          (0, 2),
                                                              (1, 2),
                                     (2, -1), (2, 0), (2, 1), (2, 2)] + top_middle_3x3offsets
#    Bottom-right of the top left
corner_top_left_bottom_right_5x5offsets = [                         (-1, 2),
                                                                     (0, 2),
                                                                     (1, 2),
                                           (2, -1), (2, 0), (2, 1),  (2, 2)] + middle_3x3offsets

#    At top right
corner_top_right_5x5offsets = [ (0, -2), 
                                (1, -2),
                                (2, -2), (2, -1), (2, 0)] + corner_top_right_3x3offsets
#    One pixel below top right
corner_top_right_below_5x5offsets = [(-1, -2),
                                      (0, -2),
                                      (1, -2),
                                      (2, -2), (2, -1), (2, 0)] + right_middle_3x3offsets
#    One pixel to the left of top right
corner_top_right_left_5x5offsets = [(0, -2),
                                    (1, -2),
                                    (2, -2), (2, -1), (2, 0), (2, 1)] + top_middle_3x3offsets
#    Bottom-left of the top right
corner_top_right_bottom_left_5x5offsets = [(-1, -2),
                                            (0, -2),
                                            (1, -2),
                                            (2, -2), (2, -1), (2, 0), (2, 1)] + middle_3x3offsets

#    At bottom left
corner_bottom_left_5x5offsets = [(-2, 0), (-2, 1), (-2, 2),
                                                   (-1, 1),
                                                    (0, 2)] + corner_bottom_left_3x3offsets
#    One pixel above bottom left
corner_bottom_left_above_5x5offsets = [(-2, 0), (-2, 1), (-2, 2),
                                                         (-1, 2),
                                                          (0, 2),
                                                          (1, 2)] + left_middle_3x3offsets
#    One pixel to the right of bottom left
corner_bottom_left_right_5x5offsets = [(-2, -1), (-2, 0), (-2, 1), (-2, 2),
                                                                   (-1, 2),
                                                                    (0, 2)] + bottom_middle_3x3offsets
#    Top-right of the bottom left
corner_bottom_left_top_right_5x5offsets = [(-2, -1), (-2, 0), (-2, 1), (-2, 2),
                                                                       (-1, 2),
                                                                        (0, 2),
                                                                        (1, 2)] + middle_3x3offsets

#    At bottom right
corner_bottom_right_5x5offsets = [(-2, -2), (-2, -1), (-2, 0),
                                  (-1, -2),
                                   (0, -2)                   ] + corner_bottom_right_3x3offsets
#    One pixel above bottom right
corner_bottom_right_above_5x5offsets = [(-2, -2), (-2, -1), (-2, 0),
                                        (-1, -2),
                                         (0, -2),
                                         (1, -2)                   ] + right_middle_3x3offsets
#    One pixel to the left of bottom right
corner_bottom_right_left_5x5offsets = [(-2, -2), (-2, -1), (-2, 0), (-2, 1),
                                       (-1, -2),
                                        (0, -2)                            ] + bottom_middle_3x3offsets
#    Top-left of the bottom right
corner_bottom_right_top_left_5x5offsets = [(-2, -2), (-2, -1), (-2, 0), (-2, 1),
                                           (-1, -2),
                                            (0, -2),
                                            (1, -2)                            ] + middle_3x3offsets

## Full dictionary:
offsets5x5_dict = {'middle': middle_5x5offsets,
                   'bottom_middle': bottom_middle_5x5offsets,
                     'bottom_middle_above': bottom_middle_above_5x5offsets,
                   'top_middle': top_middle_5x5offsets,
                     'top_middle_below': top_middle_below_5x5offsets,
                   'left_middle': left_middle_5x5offsets,
                     'left_middle_right': left_middle_right_5x5offsets,
                   'right_middle': right_middle_5x5offsets,
                     'right_middle_left': right_middle_left_5x5offsets,
                   'corner_top_left': corner_top_left_5x5offsets,
                     'corner_top_left_below': corner_top_left_below_5x5offsets,
                     'corner_top_left_right': corner_top_left_right_5x5offsets,
                     'corner_top_left_bottom_right': corner_top_left_bottom_right_5x5offsets,
                   'corner_top_right': corner_top_right_5x5offsets,
                     'corner_top_right_below': corner_top_right_below_5x5offsets,
                     'corner_top_right_left': corner_top_right_left_5x5offsets,
                     'corner_top_right_bottom_left': corner_top_right_bottom_left_5x5offsets,
                   'corner_bottom_left': corner_bottom_left_5x5offsets,
                     'corner_bottom_left_above': corner_bottom_left_above_5x5offsets,
                     'corner_bottom_left_right': corner_bottom_left_right_5x5offsets,
                     'corner_bottom_left_top_right': corner_bottom_left_top_right_5x5offsets,
                   'corner_bottom_right': corner_bottom_right_5x5offsets,
                     'corner_bottom_right_above': corner_bottom_right_above_5x5offsets,
                     'corner_bottom_right_left': corner_bottom_right_left_5x5offsets,
                     'corner_bottom_right_top_left': corner_bottom_right_top_left_5x5offsets}

##################################
## Define functions
##################################
def get_offsets (grid_size, image_shape, coordiante):

    ''' Get offsets of a coordinate for a given grid size and image shape.

        inputs
        ------
        grid_size : int
            Size of the grid (3 or 5).
        image_shape : tuple
            Shape of the image (height, width).
        coordiante : list
            Coordinate [x, y] for which to get the offsets.                        

        outputs
        -------
        offsets : list of list
            List of offsets for the given coordinate.
    '''

    ## Grid size must be either 3 x 3 or 5 x 5
    if grid_size not in [3, 5]:
        raise ValueError("Grid size must be either 3 or 5.")

    ## Get the offsets dictionary based on grid size
    offsets_dict = offsets3x3_dict if grid_size == 3 else offsets5x5_dict

    ## Get the last indices of the image shape
    last_row_idx = image_shape[0] - 1
    last_col_idx = image_shape[1] - 1

    ## Get the coordinate offsets based on the coordinate position
    #  Corners:
    if coordiante[0] == 0 and coordiante[1] == 0:
        return offsets_dict['corner_top_left']
    if coordiante[0] == 0 and coordiante[1] == last_col_idx:
        return offsets_dict['corner_top_right']
    if coordiante[0] == last_row_idx and coordiante[1] == 0:
        return offsets_dict['corner_bottom_left']
    if coordiante[0] == last_row_idx and coordiante[1] == last_col_idx:
        return offsets_dict['corner_bottom_right']
    #  If 5x5, the three pixel around corners:
    if grid_size == 5:
        if coordiante[0] == 1 and coordiante[1] == 0: # Top left corner, one pixel below
            return offsets_dict['corner_top_left_below']
        if coordiante[0] == 0 and coordiante[1] == last_col_idx-1: # Top right corner, one pixel to the left
            return offsets_dict['corner_top_right_left']
        if coordiante[0] == last_row_idx-1 and coordiante[1] == 0: # Bottom left corner, one pixel above
            return offsets_dict['corner_bottom_left_above']
        if coordiante[0] == last_row_idx-1 and coordiante[1] == last_col_idx-1: # Bottom right corner, one pixel to the left
            return offsets_dict['corner_bottom_right_left']

    #  Borders:
    if coordiante[0] == 0: # Top border
        return offsets_dict['top_middle']
    elif coordiante[0] == last_row_idx: # Bottom border
        return offsets_dict['bottom_middle']
    elif coordiante[1] == 0: # Left border
        return offsets_dict['left_middle']
    elif coordiante[1] == last_col_idx: # Right border
        return offsets_dict['right_middle']
    #  If 5x5, one pixel off the borders:
    if grid_size == 5:
        if coordiante[0] == 1: # One pixel below top border
            return offsets_dict['top_middle_below']
        elif coordiante[0] == last_row_idx - 1: # One pixel above bottom border
            return offsets_dict['bottom_middle_above']
        elif coordiante[1] == 1: # One pixel to the right of left border
            return offsets_dict['left_middle_right']
        elif coordiante[1] == last_col_idx - 1: # One pixel to the left of right border
            return offsets_dict['right_middle_left']

    # Middle of the image
    return offsets_dict['middle']