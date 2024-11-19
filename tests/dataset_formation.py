import os
import numpy as np
import h5py
from PIL import Image
from sklearn.model_selection import train_test_split
import albumentations as A

# Configurations
GIT_FOLDER_ONLY = False  # Set to true if working with the git folder database
DATASET_NAME = "512x512_Ignored2_60_rgb_png_7056_augmented"
TEST_SIZE = 0.2  # Percentage of data to be used for testing (e.g., 20%)
NUM_AUGMENTATIONS_PER_IMAGE = 1  # Number of augmented versions to create per image

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
    input_array = np.array(input_img)  # Keep as uint8
    
    # Open and convert output image to numpy array
    output_img = Image.open(output_image_path).convert('L')  # Convert to grayscale
    output_array = np.array(output_img)
    
    # Convert output array to binary mask (0 and 1)
    output_array = (output_array > 128).astype(np.uint8)
    
    # Check that input and output images have the same dimensions
    if input_array.shape[:2] != output_array.shape[:2]:
        print(f"Dimension mismatch between input and output images for file {fname}")
        continue
    
    # Append original images to lists
    X.append(input_array)
    Y.append(output_array)
    
# Convert lists to numpy arrays
X = np.array(X, dtype=np.uint8)
Y = np.array(Y, dtype=np.uint8)

# Split the data into training and testing sets
Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, test_size=TEST_SIZE, random_state=42)

# Define an augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(p=0.5),
    # Add more transformations as needed
])

# Apply augmentation to training data only
Xtrain_aug = []
ytrain_aug = []

for i in range(len(Xtrain)):
    input_array = Xtrain[i]
    output_array = ytrain[i]
    
    # Append original image
    Xtrain_aug.append(input_array)
    ytrain_aug.append(output_array)
    
    # Apply augmentation if the mask contains fire pixels
    if output_array.sum() > 0:
        for _ in range(NUM_AUGMENTATIONS_PER_IMAGE):
            # Apply augmentation
            augmented = transform(image=input_array, mask=output_array)
            aug_input_array = augmented['image']
            aug_output_array = augmented['mask']
            
            # Append augmented images to lists
            Xtrain_aug.append(aug_input_array)
            ytrain_aug.append(aug_output_array)

# Convert augmented training data to numpy arrays
Xtrain = np.array(Xtrain_aug, dtype=np.float32) / 255.0  # Normalize to [0,1]
ytrain = np.array(ytrain_aug, dtype=np.uint8)

# Convert test data to numpy arrays and normalize
Xtest = np.array(Xtest, dtype=np.float32) / 255.0  # Normalize to [0,1]
ytest = np.array(ytest, dtype=np.uint8)

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
