import os
import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from src.model import WildfireSegmentation, FocalLoss
from PIL import Image
import time
import json
import itertools
import copy

def generate_model_filename(config):
    """
    Generate a model filename based on hyperparameters.

    Args:
        config (dict): Configuration dictionary containing hyperparameters.

    Returns:
        str: Generated model filename.
    """
    # List of hyperparameters to include in the filename
    params_to_include = [
        'MODEL', 'LOSS_FUNCTION', 'OPTIMIZER', 'LEARNING_RATE',
        'NUM_EPOCHS', 'BATCH_SIZE', 'USE_SE', 'USE_CBAM',
        'FOCAL_LOSS_ALPHA', 'FOCAL_LOSS_GAMMA', 'FOREGROUND_WEIGHT'
    ]

    # Construct filename components
    filename_components = [config['DATASET_NAME']]
    for param in params_to_include:
        if param in config:
            value = config[param]
            # Simplify boolean values
            if isinstance(value, bool):
                value = 'T' if value else 'F'
            # Replace '.' in floats to avoid filesystem issues
            if isinstance(value, float):
                value = f"{value:.0e}" if value < 1e-2 else f"{value}"
                value = value.replace('.', 'p')
            filename_components.append(f"{param}_{value}")

    # Join components with underscores
    filename = '_'.join(filename_components) + '.pth'
    return filename

def run_experiment(CONFIG):

    # Create the model filename
    model_filename = generate_model_filename(CONFIG)

    if CONFIG['GIT_FOLDER_ONLY']:
        CONFIG['MODEL_PATH'] = os.path.abspath(f'../repo/model/{model_filename}')
    
    else:
        CONFIG['MODEL_PATH'] = f'G:/4 - models/{model_filename}'

    if base_CONFIG['LOAD_FROM_MODEL']:

        # Instantiate your model class
        model = WildfireSegmentation(use_se=CONFIG['USE_SE'], use_cbam=CONFIG['USE_CBAM'])

        # Load the saved state_dict
        model.load_state_dict(torch.load(CONFIG['MODEL_PATH']))

        # Move the model to the appropriate device
        model.to(device)

        # Set the model to evaluation mode
        model.eval()

    else:

        # Calculate batch size
        batch_size = CONFIG['BATCH_SIZE']
        print(f"Batch size is: {batch_size}{', consider reducing it' if batch_size > 32 else ''}")

        # Oversampling positive samples if required
        if CONFIG['OVERSAMPLE']:
            # Flatten masks to find positive samples
            ytrain_flat = ytrain.view(ytrain.size(0), -1).sum(dim=1)
            # Create labels: 0 if no positive pixels, 1 if any positive pixels
            ytrain_labels = (ytrain_flat > 0).long()

            class_sample_count = np.array([len(np.where(ytrain_labels.numpy() == t)[0]) for t in np.unique(ytrain_labels.numpy())])
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[t] for t in ytrain_labels.numpy()])
            samples_weight = torch.from_numpy(samples_weight).double()
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

            train_dataloader = DataLoader(train_set, batch_size=batch_size, sampler=sampler)
        else:
            train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        val_dataloader = DataLoader(test_set, batch_size=4)

        print("ytrain dtype after resizing:", ytrain.dtype)
        print("Unique values in ytrain:", torch.unique(ytrain))
        print("ytest dtype after resizing:", ytest.dtype)
        print("Unique values in ytest:", torch.unique(ytest))

        if CONFIG['VISUALIZE_DATA']:
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
        model = WildfireSegmentation(use_se=CONFIG['USE_SE'], use_cbam=CONFIG['USE_CBAM']).to(device)

        # Create a random input tensor with the shape [batch_size, channels, height, width]
        channels = 3  # Input RGB
        height = CONFIG['GOAL_DATASET_HW']
        width = CONFIG['GOAL_DATASET_HW']

        # Create the random input tensor
        input_tensor = torch.randn(batch_size, channels, height, width).to(device)

        # Forward pass through the model without calculating gradients
        with torch.no_grad():
            output = model(input_tensor)  # Get the output from the model
            print('Output Shape', output.shape)  # Print the output shape

        # Training and validation functions
        def train_loop(model, dataloader, optimizer, device, criterion, epoch):
            model.train()  # Set the model to training mode
            running_loss = 0.0  # Initialize cumulative loss for the epoch
            total_batches = len(dataloader)  # Get the total number of batches in the dataloader
            epoch_start_time = time.time()  # Record the start time for epoch timing

            # Iterate over the batches of data
            for batch, (images, masks) in enumerate(dataloader):
                batch_start_time = time.time()  # Start time for the current batch

                # Move the images and masks to the specified device (CPU or GPU)
                images = images.to(device)
                masks = masks.to(device).unsqueeze(1).float()

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

                # Calculate time estimates
                time_elapsed_so_far = time.time() - epoch_start_time
                estimated_total_time = (time_elapsed_so_far / (batch + 1)) * total_batches
                estimated_time_left = estimated_total_time - time_elapsed_so_far

                # Print progress for the current batch with time estimates
                print(f'Epoch: {epoch}, Batch {batch + 1}/{total_batches}, Batch Loss: {loss.item():.4f}, '
                      f'Time per batch: {batch_end_time - batch_start_time:.3f} s, '
                      f'Estimated time left for epoch: {estimated_time_left:.2f} s')

            epoch_end_time = time.time()  # End time for the epoch
            # Compute the average loss for the epoch
            epoch_loss = running_loss / len(dataloader.dataset)

            # Print the total loss and time taken for the epoch
            print(f'Epoch: {epoch}, Total Train Loss: {epoch_loss:.4f}, Time per epoch: {epoch_end_time - epoch_start_time:.3f} seconds')

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

        # Choose loss function
        if CONFIG['LOSS_FUNCTION'] == 'FocalLoss':
            criterion = FocalLoss(alpha=CONFIG['FOCAL_LOSS_ALPHA'], gamma=CONFIG['FOCAL_LOSS_GAMMA'], logits=True).to(device)

        elif CONFIG['LOSS_FUNCTION'] == 'CrossEntropy':
            criterion = nn.CrossEntropyLoss().to(device)

        elif CONFIG['LOSS_FUNCTION'] == 'BCEWithLogitsLoss':
            foreground_weight = CONFIG['FOREGROUND_WEIGHT']
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([foreground_weight]).to(device))

        else:
            criterion = nn.BCEWithLogitsLoss().to(device)

        # Choose optimizer
        if CONFIG['OPTIMIZER'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['LEARNING_RATE'])

        elif CONFIG['OPTIMIZER'] == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=CONFIG['LEARNING_RATE'])

        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['LEARNING_RATE'])

        num_epochs = CONFIG['NUM_EPOCHS']
        train_losses = []
        val_losses = []
        for epoch in range(num_epochs):
            train_loss = train_loop(model, train_dataloader, optimizer, device, criterion, epoch)
            val_loss = validate(model, val_dataloader, device, criterion, epoch)

            if CONFIG['VISUALIZE_AFTER_EPOCH']:
                visualize_predictions(model, val_dataloader, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

        if CONFIG['PLOT_CONFIG_LOSS']:
            plt.plot(train_losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.legend()
            plt.tight_layout()
            plt.show()

        # Save the model's state_dict
        torch.save(model.state_dict(), CONFIG['MODEL_PATH'])

        # Optionally, save the CONFIG dictionary as a JSON file alongside the model
        if CONFIG['SAVE_CONFIG_AS_JSON']:
            config_filename = CONFIG['MODEL_PATH'].replace('.pth', '_config.json')
            with open(config_filename, 'w') as f:
                json.dump(CONFIG, f, indent=4)

    if CONFIG['TEST_MODEL_WITH_PICTURE']:

        # Load the image using PIL
        input_image = Image.open(CONFIG['TEST_PICTURE']).convert('RGB')

        # Resize the image to the model's expected input size
        input_image_resized = input_image.resize((CONFIG['GOAL_DATASET_HW'], CONFIG['GOAL_DATASET_HW']))

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
            output = model(input_tensor)  # Output shape: [1, num_classes, H, W]

            # Apply sigmoid activation for binary classification
            predicted_mask = torch.sigmoid(output)
            predicted_mask = (predicted_mask > 0.5).float()

            # Remove batch dimension
            predicted_mask = predicted_mask.squeeze(0)  # Shape: [num_classes, H, W]

        # Resize the predicted mask back to the original image size
        predicted_mask_resized = F.interpolate(
            predicted_mask.unsqueeze(0),  # Add batch dimension
            size=(CONFIG['REAL_DATASET_HW'], CONFIG['REAL_DATASET_HW']),
            mode='nearest'
        ).squeeze(0)  # Remove batch dimension

        print("prediction dtype after resizing:", predicted_mask_resized.dtype)
        print("Unique values in prediction:", torch.unique(predicted_mask_resized))

        # Convert the tensor to a NumPy array
        predicted_mask_np = predicted_mask_resized.cpu().numpy().astype(np.uint8)

        # Convert the input image to a NumPy array for visualization
        original_image_np = np.array(input_image)

        # Resize original image to match the real dataset size
        original_image_resized = original_image_np

        if CONFIG['GOAL_DATASET_HW'] != CONFIG['REAL_DATASET_HW']:
            original_image_resized = cv2.resize(original_image_np, (CONFIG['REAL_DATASET_HW'], CONFIG['REAL_DATASET_HW']))

        # Overlay mask on the original image
        overlay = original_image_resized.copy()
        overlay[predicted_mask_np[0] == 1] = [255, 0, 0]  # Color the mask region red

        # Blend images
        blended = cv2.addWeighted(original_image_resized, 0.7, overlay, 0.3, 0)

        # Create a figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Display the original image
        axs[0].imshow(original_image_resized)
        axs[0].set_title('Original Image')
        axs[0].axis('off')

        # Display the blended image
        axs[1].imshow(blended)
        axs[1].set_title('Predicted Mask Overlay')
        axs[1].axis('off')

        plt.show()

    return train_losses, val_losses

# Base configuration
base_CONFIG = {
    'GIT_FOLDER_ONLY': False,
    'DATASET_NAME': '512x512_Ignored2_60_rgb_png_7056',
    'SAVE_CONFIG_AS_JSON': True,
    'REAL_DATASET_HW': 512,
    'GOAL_DATASET_HW': 256,
    'REAL_DATASET_SIZE': 7056,
    'GOAL_DATASET_SIZE': 7056,
    'N_BATCHES': 1008,
    'VISUALIZE_DATA': False,
    'VISUALIZE_AFTER_EPOCH': False,
    'PLOT_CONFIG_LOSS': False,
    'LOAD_FROM_MODEL': False,
    'TEST_MODEL_WITH_PICTURE': False,
    'MODEL_PATH': '',
    'DATASET_PATH': '',
    'TEST_PICTURE': '',
    'MODEL': 'UNet',
    'LOSS_FUNCTION': 'FocalLoss',
    'OPTIMIZER': 'Adam',
    'LEARNING_RATE': 1e-4,
    'BATCH_SIZE': 16,
    'USE_SE': False,
    'USE_CBAM': False,
    'OVERSAMPLE': True,
    'FOCAL_LOSS_ALPHA': 1,
    'FOCAL_LOSS_GAMMA': 2,
    'FOREGROUND_WEIGHT': 10.0,
    'USE_DATA_AUGMENTATION': False,
    # Add more hyperparameters as needed
}

# Hyperparameter grid
hyperparameters = {
    'MODEL': ['UNet', 'UNet_SE', 'UNet_CBAM'],
    'LOSS_FUNCTION': ['FocalLoss', 'CrossEntropy', 'BCEWithLogitsLoss'],
    'LEARNING_RATE': [1e-3, 1e-4, 1e-5],
    'BATCH_SIZE': [8, 16],
    'USE_SE': [False, True],
    'USE_CBAM': [False, True],
    'NUM_EPOCHS': [5, 10, 20]
    # Add more hyperparameters as needed
}

# Generate all combinations
keys, values = zip(*hyperparameters.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

if base_CONFIG['GIT_FOLDER_ONLY']:
    base_CONFIG['DATASET_PATH'] = os.path.abspath(f'../repo/db/formatted_db/{base_CONFIG["DATASET_NAME"]}.h5')
    base_CONFIG['TEST_PICTURE'] = os.path.abspath(f'../repo/db/raw_images/{base_CONFIG["DATASET_NAME"]}/Input/323-1-2017_45.png')
else:
    base_CONFIG['DATASET_PATH'] = f'G:/3 - datasets/{base_CONFIG["DATASET_NAME"]}.h5'
    base_CONFIG['TEST_PICTURE'] = f'G:/2 - processed-data/Input/407-1-2017_0.png'

# Check if cuda is available
if torch.cuda.is_available():
    print('cuda is available')
    device = torch.device("cuda:0")
else:
    print('cuda is not available')
    device = torch.device("cpu")

# Empty cache
torch.cuda.empty_cache()

if not base_CONFIG['LOAD_FROM_MODEL']:

    # Load data
    data = h5py.File(base_CONFIG['DATASET_PATH'], 'r')
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

    desired_train_size = int(base_CONFIG['GOAL_DATASET_SIZE'] * train_ratio)
    desired_test_size = base_CONFIG['GOAL_DATASET_SIZE'] - desired_train_size  # Ensure total adds up

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
    if base_CONFIG['REAL_DATASET_HW'] != base_CONFIG['GOAL_DATASET_HW']:
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
        Xtrain = F.interpolate(Xtrain, size=(base_CONFIG['GOAL_DATASET_HW'], base_CONFIG['GOAL_DATASET_HW']), mode='bilinear', align_corners=False)
        Xtest = F.interpolate(Xtest, size=(base_CONFIG['GOAL_DATASET_HW'], base_CONFIG['GOAL_DATASET_HW']), mode='bilinear', align_corners=False)

        # Resize labels
        ytrain = resize_labels(ytrain, (base_CONFIG['GOAL_DATASET_HW'], base_CONFIG['GOAL_DATASET_HW']))
        ytest = resize_labels(ytest, (base_CONFIG['GOAL_DATASET_HW'], base_CONFIG['GOAL_DATASET_HW']))

    # Create datasets
    train_set = TensorDataset(Xtrain, ytrain)
    test_set = TensorDataset(Xtest, ytest)

# Collect results
results = []
total_combinations = len(combinations)
overall_start_time = time.time()
total_epochs = sum(combo.get('NUM_EPOCHS', base_CONFIG.get('NUM_EPOCHS', 0)) for combo in combinations)
epochs_completed = 0
print(f"A total of {total_epochs} epochs will be trained")

for idx, combo in enumerate(combinations):
    combo_start_time = time.time()
    print(f"Running configuration {idx + 1}/{total_combinations}: {combo}")

    # Deep copy of the base CONFIG to avoid side effects
    config_run = copy.deepcopy(base_CONFIG)
    config_run.update(combo)

    # Generate the model filename and update paths
    model_filename = generate_model_filename(config_run)

    if config_run['GIT_FOLDER_ONLY']:
        config_run['MODEL_PATH'] = os.path.abspath(f'../repo/model/{model_filename}')
        config_run['DATASET_PATH'] = os.path.abspath(f'../repo/db/formatted_db/{config_run["DATASET_NAME"]}.h5')
        config_run['TEST_PICTURE'] = os.path.abspath(f'../repo/db/raw_images/{config_run["DATASET_NAME"]}/Input/323-1-2017_45.png')
    else:
        config_run['MODEL_PATH'] = f'G:/4 - models/{model_filename}'
        config_run['DATASET_PATH'] = f'G:/3 - datasets/{config_run["DATASET_NAME"]}.h5'
        config_run['TEST_PICTURE'] = f'G:/2 - processed-data/Input/407-1-2017_0.png'

    # Optionally, adjust other CONFIG settings based on hyperparameters
    if config_run['LOSS_FUNCTION'] == 'FocalLoss':
        config_run['FOCAL_LOSS_ALPHA'] = 0.5
        config_run['FOCAL_LOSS_GAMMA'] = 2

    # Run the experiment with the updated configuration
    train_losses, val_losses = run_experiment(config_run)

    # Update epochs completed
    num_epochs = config_run['NUM_EPOCHS']
    epochs_completed += num_epochs
    combo_end_time = time.time()
    combo_time = combo_end_time - combo_start_time
    total_time_so_far = time.time() - overall_start_time
    average_epoch_time = total_time_so_far / epochs_completed
    estimated_total_time = average_epoch_time * total_epochs
    estimated_time_left = estimated_total_time - total_time_so_far

    # Print estimated times
    print(f"Time for this combination: {combo_time:.2f} s")
    print(f"Total time elapsed: {total_time_so_far / 60:.2f} minutes")
    print(f"Estimated time left for all combinations: {estimated_time_left / 60:.2f} minutes")

    # Store results with hyperparameters
    results.append({
        'hyperparameters': combo,
        'train_losses': train_losses,
        'val_losses': val_losses,
    })

# After all experiments, save results to a file
with open('experiment_results.json', 'w') as f:
    json.dump(results, f, indent=4)