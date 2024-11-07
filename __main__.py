# Import libraries
import os
import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from src.model import WildfireSegmentation
from PIL import Image
from torch.cuda.amp import autocast, GradScaler

# Configuration
GIT_FOLDER_ONLY = True # Set to true if working with the git folder database
DATASET_NAME = "512x512_Ignored2_60_rgb_png_7056"
REAL_DATASET_HW = 512 # Source
GOAL_DATASET_HW = 256 # Goal
REAL_DATASET_SIZE = 7056
GOAL_DATASET_SIZE = 7056
N_BATCHES = 1008
VISUALIZE_DATA = False
LOAD_FROM_MODEL = True
TEST_MODEL_WITH_PICTURE = True
if GIT_FOLDER_ONLY:
    MODEL_PATH = os.path.abspath(f'../repo/model/{DATASET_NAME}_{GOAL_DATASET_HW}_{GOAL_DATASET_SIZE}.pth')
    DATASET_PATH = os.path.abspath(f'../repo/db/formatted_db/{DATASET_NAME}.h5')
    TEST_PICTURE = os.path.abspath(f'../repo/db/raw_images/{DATASET_NAME}/Input/323-1-2017_45.png')

else:
    MODEL_PATH = f'G:/4 - models/{DATASET_NAME}_{GOAL_DATASET_HW}_{GOAL_DATASET_SIZE}.pth'
    DATASET_PATH = f'G:/3 - datasets/{DATASET_NAME}.h5'
    TEST_PICTURE = f'G:/2 - processed-data/Input/407-1-2017_0.png'

# Check if cuda is available:
if torch.cuda.is_available():
    print('cuda is available')
    device = torch.device("cuda:0")
else:
    print('cuda is not available')
    device = torch.device("cpu")

# Empty cache
torch.cuda.empty_cache()

if LOAD_FROM_MODEL:
    
    # Instantiate your model class
    model = WildfireSegmentation()

    # Load the saved state_dict
    model.load_state_dict(torch.load(MODEL_PATH))

    # Move the model to the appropriate device
    model.to(device)

    # Set the model to evaluation mode
    model.eval()

else:
    #%% set up data sets:
    data = h5py.File(DATASET_PATH)
    Xtrain = np.array(data['Xtrain'])
    Xtest = np.array(data['Xtest'])
    ytrain = np.array(data['ytrain'])
    ytest = np.array(data['ytest'])

    # Get original dataset sizes
    original_train_size = Xtrain.shape[0]
    original_test_size = Xtest.shape[0]
    original_total_size = original_train_size + original_test_size

    # Calculate desired train and test sizes, maintaining the original ratio
    train_ratio = original_train_size / original_total_size
    test_ratio = original_test_size / original_total_size

    desired_train_size = int(GOAL_DATASET_SIZE * train_ratio)
    desired_test_size = GOAL_DATASET_SIZE - desired_train_size  # Ensure total adds up

    # Reduce training dataset
    if desired_train_size < original_train_size:
        train_indices = np.random.choice(original_train_size, desired_train_size, replace=False)
        Xtrain = Xtrain[train_indices]
        ytrain = ytrain[train_indices]
        print(f"Reduced training dataset size to {desired_train_size} samples.")
    else:
        print("Training dataset size remains unchanged.")

    # Reduce testing dataset
    if desired_test_size < original_test_size:
        test_indices = np.random.choice(original_test_size, desired_test_size, replace=False)
        Xtest = Xtest[test_indices]
        ytest = ytest[test_indices]
        print(f"Reduced testing dataset size to {desired_test_size} samples.")
    else:
        print("Testing dataset size remains unchanged.")

    # Convert numpy arrays to torch tensors
    Xtrain = torch.from_numpy(Xtrain).float()
    Xtest = torch.from_numpy(Xtest).float()
    ytrain = torch.from_numpy(ytrain).long()
    ytest = torch.from_numpy(ytest).long()

    # Permute dimensions from (N, H, W, C) to (N, C, H, W)
    Xtrain = Xtrain.permute(0, 3, 1, 2)
    Xtest = Xtest.permute(0, 3, 1, 2)

    # Resize dataset if needed
    if REAL_DATASET_HW != GOAL_DATASET_HW:

        def resize_labels(labels, size):
            # Ensure labels have a channel dimension
            if labels.dim() == 3:
                labels = labels.unsqueeze(1)
            # Convert to float for interpolation
            labels = labels.float()
            # Resize using nearest neighbor to preserve label values
            labels = F.interpolate(labels, size=size, mode='nearest')
            # Remove channel dimension and convert back to long
            labels = labels.squeeze(1).long()
            return labels

        # Resize images
        Xtrain = F.interpolate(Xtrain, size=(GOAL_DATASET_HW, GOAL_DATASET_HW), mode='bilinear', align_corners=False)
        Xtest = F.interpolate(Xtest, size=(GOAL_DATASET_HW, GOAL_DATASET_HW), mode='bilinear', align_corners=False)

        # Resize labels using the modified function
        ytrain = resize_labels(ytrain, (GOAL_DATASET_HW, GOAL_DATASET_HW))
        ytest = resize_labels(ytest, (GOAL_DATASET_HW, GOAL_DATASET_HW))

    #set up data sets:
    train_set = TensorDataset(Xtrain, ytrain)
    test_set = TensorDataset(Xtest, ytest)

    # Create datasets
    train_set = TensorDataset(Xtrain, ytrain)
    test_set = TensorDataset(Xtest, ytest)

    # Calculate batch size
    batch_size = max(1, len(train_set) // N_BATCHES)
    print(f"Batch size is: {batch_size}{', consider reducing it' if batch_size > 32 else ''}")

    # Create dataloaders
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(test_set, batch_size = 4)

    print("ytrain dtype after resizing:", ytrain.dtype)
    print("Unique values in ytrain:", torch.unique(ytrain))
    print("ytest dtype after resizing:", ytest.dtype)
    print("Unique values in ytest:", torch.unique(ytest))

    if VISUALIZE_DATA:
        # Select 5 random indices from the training set
        sample_indices = np.random.choice(len(train_set), 5, replace = False)

        # Create subplots: 2 rows (images and labels), 5 columns (samples)
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))

        # Visualize 5 random pictures
        for i, idx in enumerate(sample_indices):
            # Get the image and label
            image, label = train_set[idx]
            
            # Convert the tensors to NumPy arrays
            image = image.numpy()
            label = label.numpy()
            
            # Adjust image dimensions if necessary
            if image.ndim == 3 and image.shape[0] in [1, 3]:
                image = np.transpose(image, (1, 2, 0))

            elif image.ndim == 2 or (image.ndim == 3 and image.shape[0] == 1):
                image = image.squeeze()
            
            # Normalize image to [0,1] for plotting if necessary
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = np.clip(image, 0, 1)  # Ensure values are within [0,1]
            
            # Plot the image in the first row
            axes[0, i].imshow(image)
            axes[0, i].axis('off')
            
            # Plot the label (mask) in the second row
            axes[1, i].imshow(label, cmap='gray')
            axes[1, i].axis('off')

        # Adjust the layout to prevent overlap
        plt.tight_layout()
        plt.show()

    # Instantiate the model and move it to the appropriate device
    model = WildfireSegmentation().to(device)

    # Create a random input tensor with the shape [batch_size, channels, height, width]
    channels = 3 # Input RGB
    height = GOAL_DATASET_HW
    width = GOAL_DATASET_HW

    # Create the random input tensor
    input_tensor = torch.randn(batch_size, channels, height, width).to(device)

    # Forward pass through the model without calculating gradients
    with torch.no_grad():
        output = model(input_tensor)  # Get the output from the model
        print('Output Shape', output.shape)  # Print the output shape

    import time  # Make sure to import the time module

    def train_loop(model, dataloader, optimizer, device, criterion, epoch):
        model.train()  # Set the model to training mode
        running_loss = 0.0  # Initialize cumulative loss for the epoch
        total_batches = len(dataloader)  # Get the total number of batches in the dataloader
        start_time = time.time()  # Record the start time for epoch timing

        # Iterate over the batches of data
        for batch, (images, masks) in enumerate(dataloader):
            batch_start_time = time.time()  # Start time for the current batch

            # Move the images and masks to the specified device (CPU or GPU)
            images = images.to(device)
            masks = masks.to(device).unsqueeze(1).float()  # MOD

            # Forward pass: compute the model's output given the input images
            outputs = model(images)

            # Compute the loss between the predicted outputs and the true masks
            loss = criterion(outputs, masks)

            # Backward pass: compute gradients for optimization
            optimizer.zero_grad()  # Clear the gradients from the previous step
            loss.backward()  # Compute gradients of the loss w.r.t model parameters
            optimizer.step()  # Update model parameters based on gradients

            # Accumulate the loss scaled by the number of images in the batch
            running_loss += loss.item() * images.size(0)

            batch_end_time = time.time()  # End time for the current batch

            # Print progress for the current batch
            print(f'Epoch: {epoch}, Batch {batch + 1}/{total_batches}, Batch Loss: {loss.item():.4f}, Time per batch: {batch_end_time - batch_start_time:.3f} s')

        epoch_end_time = time.time()  # End time for the epoch
        # Compute the average loss for the epoch
        epoch_loss = running_loss / len(dataloader.dataset)

        # Print the total loss and time taken for the epoch
        print(f'Epoch: {epoch}, Total Train Loss: {epoch_loss:.4f}, Time per epoch: {epoch_end_time - start_time:.3f} seconds')

        return epoch_loss  # Return the average loss for the epoch

    def validate(model, dataloader, device, criterion, epoch):
        model.eval()  # Set the model to evaluation mode

        running_loss = 0.0  # Initialize cumulative loss for the validation

        # Disable gradient tracking during validation to save memory and computation
        with torch.no_grad():
            # Iterate over the validation batches
            for images, masks in dataloader:
                images = images.to(device)  # Move images to the specified device
                masks = masks.to(device).unsqueeze(1).float()  # Move masks to the specified device

                # Forward pass: compute the model's output
                outputs = model(images)

                # Compute the loss between the predicted outputs and the true masks
                loss = criterion(outputs, masks)

                # Accumulate the loss scaled by the number of images in the batch
                running_loss += loss.item() * images.size(0)

        # Compute the average loss for the validation set
        epoch_loss = running_loss / len(dataloader.dataset)

        # Print the total validation loss
        print(f'Epoch: {epoch}, Total Val Loss: {epoch_loss:.4f}')

        return epoch_loss  # Return the average loss for validation

    def visualize_predictions(model, dataloader, device):
        model.eval()
        images, masks = next(iter(dataloader))
        images = images.to(device)
        masks = masks.to(device).unsqueeze(1).float()

        with torch.no_grad():
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds = preds.cpu().numpy()
            images = images.cpu().numpy()
            masks = masks.cpu().numpy()

        # Plot images, ground truth masks, and predicted masks
        for i in range(min(3, images.shape[0])):  # Display up to 3 samples
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            # Original image
            axs[0].imshow(np.transpose(images[i], (1, 2, 0)))
            axs[0].set_title('Original Image')
            axs[0].axis('off')
            # Ground truth mask
            axs[1].imshow(masks[i, 0], cmap='gray')
            axs[1].set_title('Ground Truth Mask')
            axs[1].axis('off')
            # Predicted mask
            axs[2].imshow(preds[i, 0], cmap='gray')
            axs[2].set_title('Predicted Mask')
            axs[2].axis('off')
            plt.show()

    #criterion = nn.CrossEntropyLoss() # CrossEntropyLoss / BCEWithLogitsLoss
    foreground_weight = 10.0  # Adjust this value based the dataset
    criterion = nn.BCEWithLogitsLoss(pos_weight = torch.tensor([foreground_weight]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4) # MOD

    num_epochs = 10 # Define the number of epochs
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        train_loss = train_loop(model, train_dataloader, optimizer, device, criterion, epoch)
        val_loss = validate(model, val_dataloader, device, criterion, epoch)
        visualize_predictions(model, val_dataloader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.tight_layout()
    plt.show()

    # Save the model's state_dict
    torch.save(model.state_dict(), MODEL_PATH)

if TEST_MODEL_WITH_PICTURE:
    # Load the image using PIL
    input_image = Image.open(TEST_PICTURE).convert('RGB')

    # Resize the image to the model's expected input size
    input_image_resized = input_image.resize((GOAL_DATASET_HW, GOAL_DATASET_HW))

    # Convert the image to a NumPy array and normalize pixel values to [0, 1]
    input_array = np.array(input_image_resized) / 255.0

    # Convert the NumPy array to a PyTorch tensor
    input_tensor = torch.from_numpy(input_array).float()

    # Permute dimensions from (H, W, C) to (C, H, W)
    input_tensor = input_tensor.permute(2, 0, 1)

    # Add a batch dimension to the tensor
    input_tensor = input_tensor.unsqueeze(0)  # Shape: [1, C, H, W]

    # Move the tensor to the appropriate device
    input_tensor = input_tensor.to(device)

    # Disable gradient calculation
    with torch.no_grad():
        # Forward pass
        output = model(input_tensor)  # Output shape: [1, 2, H, W]

        # Get the predicted class for each pixel
        predicted_mask = torch.argmax(output, dim=1)  # Shape: [1, H, W]

        # Add a channel dimension
        predicted_mask = predicted_mask.unsqueeze(1)  # Shape: [1, 1, H, W]

    # Resize the predicted mask back to the original image size
    predicted_mask_resized = F.interpolate(
        predicted_mask.float(), 
        size=(REAL_DATASET_HW, REAL_DATASET_HW), 
        mode='nearest'
    )

    print("prediction dtype after resizing:", predicted_mask_resized.dtype)
    print("Unique values in prediction:", torch.unique(predicted_mask_resized))

    # Remove the batch and channel dimensions
    predicted_mask_resized = predicted_mask_resized.squeeze(0).squeeze(0)  # Shape: [H, W]

    # Convert the tensor to a NumPy array
    predicted_mask_np = predicted_mask_resized.cpu().numpy().astype(np.uint8)

    # Convert the input image to a NumPy array for visualization
    original_image_np = np.array(input_image)

    # Convert RGB to HSV
    hsv_image = cv2.cvtColor(predicted_mask_np, cv2.COLOR_RGB2HSV).astype(np.float32)

    # Increase saturation by 10x (1000%)
    hsv_image[:, :, 1] *= 10

    # Clip the saturation values to the maximum value of 255
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1], 0, 255)

    # Convert back to uint8
    hsv_image = hsv_image.astype(np.uint8)

    # Convert HSV back to RGB
    saturated_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Display the original image
    axs[0].imshow(original_image_np)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    # Display the predicted mask
    #axs[1].imshow(predicted_mask_np, cmap='gray')
    #axs[1].set_title('Predicted Mask')
    #axs[1].axis('off')

    axs[1].imshow(saturated_image)
    axs[1].set_title('Increased Saturation (1000%)')
    axs[1].axis('off')

    plt.show()