import os
import numpy as np
import h5py
from PIL import Image
from sklearn.model_selection import train_test_split

GIT_FOLDER_ONLY = False # Set to true if working with the git folder database

# Configurations
DATASET_NAME = "512x512_Ignored2_60_rgb_png_7056"
TEST_SIZE = 0.2  # Set the percentage of data to be used for testing (e.g., 20%)

# Set the paths to the Input and Output folders
if GIT_FOLDER_ONLY:
    input_folder = os.path.abspath(f'../Final Project/repo/db/raw_images/{DATASET_NAME}/Input')
    output_folder = os.path.abspath(f'../Final Project/repo/db/raw_images/{DATASET_NAME}/Output')

else:
    input_folder = os.path.abspath(f'G:/2 - processed-data/{DATASET_NAME}/Input')
    output_folder = os.path.abspath(f'G:/2 - processed-data/{DATASET_NAME}/Output')

# Get the list of image filenames in the Input and Output folders
input_filenames = [f for f in os.listdir(input_folder) if f.endswith('.png')]
output_filenames = [f for f in os.listdir(output_folder) if f.endswith('.png')]

# Sort the filenames to ensure they are in the same order
input_filenames.sort()
output_filenames.sort()

# Check that the filenames match
for i in range(len(input_filenames)):
    if input_filenames[i] != output_filenames[i]:
        print(f"Mismatch between input and output filenames: {input_filenames[i]} vs {output_filenames[i]}")
        break

# Initialize lists to hold the data
X = []
Y = []

# Loop over the filenames and load the images
for fname in input_filenames:
    input_image_path = os.path.join(input_folder, fname)
    output_image_path = os.path.join(output_folder, fname)
    
    # Open and convert input image to numpy array
    input_img = Image.open(input_image_path)
    input_array = np.array(input_img) / 255.0  # Normalize to [0,1]
    
    # Open and convert output image to numpy array
    output_img = Image.open(output_image_path).convert('L')  # Convert to grayscale
    output_array = np.array(output_img)
    
    # Convert output array to binary mask (0 and 1)
    output_array = (output_array > 128).astype(np.uint8)
    
    # Check that input and output images have the same dimensions
    if input_array.shape[:2] != output_array.shape[:2]:
        print(f"Dimension mismatch between input and output images for file {fname}")
        continue
    
    # Append to lists
    X.append(input_array)
    Y.append(output_array)
    
# Convert lists to numpy arrays
X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.uint8)

# Split the data into training and testing sets
Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, test_size=TEST_SIZE, random_state=42)

# Save the datasets into an h5 file
if GIT_FOLDER_ONLY:
    h5_file = os.path.abspath(f'../Final Project/repo/db/formatted_db/{DATASET_NAME}.h5')

else:
    h5_file = os.path.abspath(f'G:/3 - datasets/{DATASET_NAME}.h5')

with h5py.File(h5_file, 'w') as hf:
    hf.create_dataset('Xtrain', data=Xtrain, dtype='float32')
    hf.create_dataset('Xtest', data=Xtest, dtype='float32')
    hf.create_dataset('ytrain', data=ytrain, dtype='uint8')
    hf.create_dataset('ytest', data=ytest, dtype='uint8')
