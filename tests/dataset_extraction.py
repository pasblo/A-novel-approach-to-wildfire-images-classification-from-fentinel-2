import hdf5plugin
import h5py
from h5py._hl.base import KeysViewHDF5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import zipfile
import imageio

# Paths to the datasets and the folder to extract zip files
DATASETS_FOLDER_PATHS = "G:\\0 - zip-data"
ZIP_FILES_FOLDER = "G:\\1 - raw-data"
JPEGS_FILES_FOLDER = "G:\\2 - processed-data"
DATASET_GROUP = "_60_"
OUTPUT_IMAGES_WIDTH = 500
OUTPUT_IMAGES_HEIGHT = 500
SHOW_INFO = False

########################################################
"""
ZIP TO H5
"""
########################################################

# Check if the data is already extracted
if len(os.listdir(ZIP_FILES_FOLDER)) > 0:
    print("Data is already extracted, skipping extracting process...")

else:

    # Ensure the ZIP_FILES_FOLDER exists
    if not os.path.exists(ZIP_FILES_FOLDER):
        os.makedirs(ZIP_FILES_FOLDER)

    # List all files in the datasets folder
    all_files = os.listdir(DATASETS_FOLDER_PATHS)

    # Get the names for all .zip files
    zip_files = [f for f in all_files if f.endswith('.zip')]

    # Extract all zip files into the ZIP_FILES_FOLDER
    print("Extracting zip files...")
    for zip_file in zip_files:
        zip_path = os.path.join(DATASETS_FOLDER_PATHS, zip_file)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(ZIP_FILES_FOLDER)
    print("Extraction complete.")

########################################################
"""
H5 broad analysis
"""
########################################################

if SHOW_INFO:

    # Check that all files in ZIP_FILES_FOLDER are .h5
    extracted_files = os.listdir(ZIP_FILES_FOLDER)
    h5_files = []
    print("Validating extracted files...")
    for f in extracted_files:
        if f.endswith('.h5') and f.find(DATASET_GROUP) != -1:
            h5_files.append(f)

    print(f"Validation complete, {len(h5_files)} h5 datasets found to process")

    def count_groups(name, obj):
        if isinstance(obj, h5py.Group):
            group_counter[0] += 1
        
        elif isinstance(obj, h5py.Dataset):
            obj: h5py.Dataset
            if not str(obj.shape) in group_size:
                group_size.append(str(obj.shape))

    # Initialize total group counter
    total_groups = 0

    # Count groups in all h5 files
    print("Counting groups in .h5 files...")
    for h5_file_name in h5_files:
        h5_file_path = os.path.join(ZIP_FILES_FOLDER, h5_file_name)
        with h5py.File(h5_file_path, 'r') as h5_file:
            group_counter = [0]  # Reset counter for each file
            group_size = []
            h5_file.visititems(count_groups)
            print(f"File: {h5_file_path}, total groups: {group_counter[0]}, sizes: {group_size}\n")
            total_groups += group_counter[0]

    print("Counting complete.")

    print(f"\nTotal number of groups in all {DATASET_GROUP} .h5 files: {total_groups}")

########################################################
"""
H5 to JPEG
"""
########################################################

# Define the path for the 'Input' and 'Output' folder and create them
input_folder_path = os.path.join(JPEGS_FILES_FOLDER, 'Input')
os.makedirs(input_folder_path, exist_ok=True)
output_folder_path = os.path.join(JPEGS_FILES_FOLDER, 'Output')
os.makedirs(output_folder_path, exist_ok=True)

# Initialize unique id counter
unique_id = 0

# Read file by file all files in the ZIP_FILES_FOLDER that contain in their name the keyword DATASET_GROUP and end in ".h5"
h5_files = [f for f in os.listdir(ZIP_FILES_FOLDER) if f.endswith('.h5') and DATASET_GROUP in f]

print(f"Processing {len(h5_files)} h5 files...")

# Function to scale images
def scale_image(img):
    img = img.astype(np.float32)
    img_min = img.min()
    img_max = img.max()
    if img_max - img_min != 0:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = img - img_min
    return img

# Function to convert float32 image to uint8
def img_float32_to_uint8(img):
    img = img * 255.0
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)

# Function to process label
def process_label(label):
    label = label.astype(np.uint8)
    if label.ndim > 2:
        label = np.squeeze(label)
    return label

# Function to process label for visualization
def visualize_label(label):
    label = label.astype(np.uint8)
    if label.ndim > 2:
        label = np.squeeze(label)
    
    # Scale label values to 0-255 for visualization
    label_max = label.max()
    if label_max > 0:
        label = (label / label_max) * 255
    label = label.astype(np.uint8)
    return label

for h5_file_name in h5_files:
    h5_file_path = os.path.join(ZIP_FILES_FOLDER, h5_file_name)
    print(f"Processing file: {h5_file_path}")
    with h5py.File(h5_file_path, 'r') as h5_file:
        # For each year group in the h5 file
        for year_group_name in h5_file.keys():
            year_group = h5_file[year_group_name]
            # For each event group in the year group
            for event_group_name in year_group.keys():
                group = year_group[event_group_name]
                print(f"Processing group: {year_group_name}/{event_group_name}")
                # Extract datasets 'sen2_60_post', 'sen2_60_pre', and 'label'
                if 'sen2_60_post' in group and 'sen2_60_pre' in group and 'label' in group:
                    sen2_60_post = group['sen2_60_post'][:]
                    sen2_60_pre = group['sen2_60_pre'][:]
                    label = group['label'][:]
                    label = np.squeeze(label)  # Ensure label is 2D

                    # Create a new label image of zeros with the same size as 'label'
                    new_label = np.zeros_like(label)

                    # Now we have four images: 'sen2_60_pre', 'sen2_60_post', 'label', 'new_label'

                    # Process images (convert to RGB images)
                    # Define the bands to use
                    bands = 'rgb'  # or 'nrg'

                    if bands == 'nrg':
                        sen2_plot_bands = [10, 3, 2]  # NIR, Red, Green bands
                    else:
                        sen2_plot_bands = [3, 2, 1]  # Red, Green, Blue bands

                    # For 'sen2_60_pre' and 'sen2_60_post', select the bands
                    num_bands = sen2_60_pre.shape[0]
                    if max(sen2_plot_bands) >= num_bands:
                        print(f"Dataset has only {num_bands} bands, cannot use bands {sen2_plot_bands}")
                        continue

                    # Extract the bands
                    sen2_60_pre_img = sen2_60_pre[sen2_plot_bands, ...]
                    sen2_60_post_img = sen2_60_post[sen2_plot_bands, ...]
                    label_img = label
                    new_label_img = new_label

                    # Scale the images
                    sen2_60_pre_img = scale_image(sen2_60_pre_img)
                    sen2_60_post_img = scale_image(sen2_60_post_img)

                    # Move the channel axis to the last dimension
                    sen2_60_pre_img = np.moveaxis(sen2_60_pre_img, 0, -1)
                    sen2_60_post_img = np.moveaxis(sen2_60_post_img, 0, -1)

                    # Get image dimensions
                    img_height, img_width = label_img.shape  # assuming label has same dimensions as the images

                    # Compute number of tiles
                    num_tiles_x = img_width // OUTPUT_IMAGES_WIDTH
                    num_tiles_y = img_height // OUTPUT_IMAGES_HEIGHT

                    # Loop over tiles
                    for y in range(num_tiles_y):
                        for x in range(num_tiles_x):
                            x_start = x * OUTPUT_IMAGES_WIDTH
                            y_start = y * OUTPUT_IMAGES_HEIGHT

                            x_end = x_start + OUTPUT_IMAGES_WIDTH
                            y_end = y_start + OUTPUT_IMAGES_HEIGHT

                            # Extract tile from each image
                            sen2_pre_tile = sen2_60_pre_img[y_start:y_end, x_start:x_end, :]
                            sen2_post_tile = sen2_60_post_img[y_start:y_end, x_start:x_end, :]
                            label_post_tile = label_img[y_start:y_end, x_start:x_end]
                            label_pre_tile = new_label_img[y_start:y_end, x_start:x_end]

                            # Skip tiles that are not the correct size
                            if sen2_pre_tile.shape[0] != OUTPUT_IMAGES_HEIGHT or sen2_pre_tile.shape[1] != OUTPUT_IMAGES_WIDTH:
                                continue

                            # Process pre image and label
                            # For pre image and label
                            has_nonzero_label = np.any(label_pre_tile != 0)
                            Y = int(has_nonzero_label)  # Y will be 0

                            X = unique_id
                            unique_id += 1
                            Z = f"{year_group_name}_{event_group_name}"

                            filename = f"{X}-{Y}-{Z}"

                            # Convert images to uint8
                            sen2_pre_tile_uint8 = img_float32_to_uint8(sen2_pre_tile)
                            label_pre_tile_uint8 = process_label(label_pre_tile)
                            label_pre_visual = visualize_label(label_pre_tile)

                            # Save pre image and label
                            pre_image_path = os.path.join(input_folder_path, f"{filename}.jpg")
                            imageio.imwrite(pre_image_path, sen2_pre_tile_uint8)

                            pre_label_path = os.path.join(output_folder_path, f"{filename}.jpg")
                            imageio.imwrite(pre_label_path, label_pre_visual) # label_pre_tile_uint8

                            # Process post image and label
                            has_nonzero_label = np.any(label_post_tile != 0)
                            Y = int(has_nonzero_label)

                            X = unique_id
                            unique_id += 1
                            Z = f"{year_group_name}_{event_group_name}"

                            filename = f"{X}-{Y}-{Z}"

                            # Convert images to uint8
                            sen2_post_tile_uint8 = img_float32_to_uint8(sen2_post_tile)
                            label_post_tile_uint8 = process_label(label_post_tile)
                            label_post_visual = visualize_label(label_post_tile)

                            # Save post image and label
                            post_image_path = os.path.join(input_folder_path, f"{filename}.jpg")
                            imageio.imwrite(post_image_path, sen2_post_tile_uint8)

                            post_label_path = os.path.join(output_folder_path, f"{filename}.jpg")
                            imageio.imwrite(post_label_path, label_post_visual) # label_post_tile_uint8

                else:
                    print(f"Group {year_group_name}/{event_group_name} does not contain the required datasets, skipping.")