'''
Utils file containing all functions used in the project notebooks
'''

import cv2
import numpy as np
import os
import pandas as pd


#Define class labels for phases present in surgery
CLASS_LABELS = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]


TOOL_LIST = [
    "tool_Grasper",
    "tool_Bipolar",
    "tool_Hook", 
    "tool_Scissors",
    "tool_Clipper", 
    "tool_Irrigator",
    "tool_SpecimenBag"]


def extract_file_paths(dir, extension):
    """Gets file paths of all files in folder with selected extension

    Args
    ----------
    dir : str
        The folder path containing all files
    extension : str
        Extension for which we are looking for files    

    Returns
    -------
    list : list containing file paths of all selected files
    """
    #Get all  file paths from folder
    file_paths = []
    for root, _, files in os.walk(dir):
        for file in files:
            if file.endswith(extension):
                file_paths.append( os.path.join(root, file))

    return file_paths

def resize_image(img, ratio):
    """Resize image based on a certain percentage ratio

    Args
    ----------
    img : (np.array)
        current frame image in array form
    ratio : (int)
        Interage value of percentage scaling for original image
    Returns
    -------
    resized_img : (np.array)
        list containing file paths of all .mp4 files
    """
    #Percent image scale
    scale_percent = ratio 

    scaled_width = int(img.shape[1] * scale_percent / 100)
    scaled_height = int(img.shape[0] * scale_percent / 100)
    img_dim = (scaled_width, scaled_height)
    
    # resize image
    resized_img = cv2.resize(img, img_dim)

    return resized_img


def preprocess_center_image(image, blur_type='bilateral'):
    """Preprocess image frame prior to saving.
        We want to crop the image to preserve as much of the scen as possible whilst removing border.
        We also apply a selected blur component to add blur to images.

    Args
    ----------
    image : np.array
        Array representation of image frame.
    blur_type : string
        Type of blur to be applied to the image. Defaults to 'bilateral'  can be 'median' also. type 'none' for no blur.


    Returns
    -------
    cropped_img : np.array containing cropped and blurred image
    """
    #Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bin_image = cv2.threshold(gray_image, 15, 255, cv2.THRESH_BINARY)

    #Do blurring on greyscale image as its more efficient than 3 channel
    if blur_type == 'bilateral':
        bin_image = cv2.bilateralFilter(bin_image,5,75,75)
    elif blur_type =='median':
        bin_image = cv2.medianBlur(bin_image, 5)


    
    #Get shapes of image out of grayscale converted.
    x = bin_image.shape[0]
    y = bin_image.shape[1]


    #Get image sizes for x,y. Trim the bottom part of the y axis to remove blank spoace
    edges_x = []
    edges_y = []
    for i in range(x):
        for j in range(10,y-10):
            if bin_image.item(i, j) != 0:
                edges_x.append(i)
                edges_y.append(j)

    #If edges not found return original image
    if not edges_x:
        return image

    
    #Get borders of image
    left_border = min(edges_x)  
    right_border = max(edges_x)
    bottom_border = min(edges_y)
    top_border = max(edges_y)


    #Calculate width and height of iamge
    img_width = right_border - left_border  
    img_height = top_border - bottom_border  

    #Update image size with new border calculations
    cropped_img = image[left_border:left_border + img_width, bottom_border:bottom_border + img_height]  
    
    return cropped_img


def import_tool_annotation_file(video_num, file_list):
    """Import tool annotation file and create dataframe.
    Convert from 1 fps to native video 25fps annotation

    Args
    ----------
    video_num : str
        Video number for which to extract annotations 
    file_list : list
        List containing annotation file paths for each video  

    Returns
    -------
    tool_df : (pd.DataFrame)
        Dataframe containing per frame tool annotations
    """
    tool_file_df = pd.read_csv([v for v in file_list if video_num in v][0], delimiter = "\t")
    fps_multi = 25

    tool_df = []
    for row in tool_file_df.itertuples(index=False):
        tool_df.extend([list(row)] * fps_multi)

    tool_df.append(tool_df[-1])
    tool_df = np.array(tool_df)


    tool_df = pd.DataFrame(tool_df[:, 1:],
                    columns=TOOL_LIST)

    return tool_df


def import_phase_annotation_file(video_num, file_list):
    """Import phase annotation file and convert phase labels to integer.

    Args
    ----------
    video_num : str
        Video number for which to extract annotations 
    file_list : list
        List containing annotation file paths for each video   

    Returns
    -------
    phase_df : (pd.DataFrame)
        Dataframe containing per frame phase annotations
    """
    phase_df = pd.read_csv([v for v in file_list if video_num in v][0], delimiter = "\t")


    for j, p in enumerate(CLASS_LABELS):
        phase_df["Phase"] = phase_df.Phase.replace({p: j})
    
    return phase_df

