import hdf5plugin
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Configurations
GIT_FOLDER_ONLY = False # Set to true if working with the git folder database
DATASET_NAME = "FLOGA_dataset_2017_sen2_60_mod_500"
EVENT = "0"
GET_DATABASE_INFO = True
GET_SAMPLE_LAYERS = False
GET_PICTURES = True

# Generate paths
if GIT_FOLDER_ONLY:
    PATH_H5 = os.path.abspath(f'../repo/db/raw_db/{DATASET_NAME}.h5')
else:
    PATH_H5 = f"G:\\1 - raw-data\\{DATASET_NAME}.h5"

if GET_DATABASE_INFO:
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"D {name} {obj.shape} {dict(obj.attrs)}")

        elif isinstance(obj, h5py.Group):
            print(f"G {name} {dict(obj.attrs)}")

    with h5py.File(PATH_H5, 'r') as h5_file:
        h5_file.visititems(print_structure)

# Data format
"""
G 2018/0 {'post_image_date': '2018-09-01', 'post_modis_file': 'MOD09GA.A2018244.h19v05.006.2018246030653', 'post_sen2_file': 'S2A_MSIL2A_20180901T092031_N0208_R093_T34SEG_20180901T131611', 'pre_image_date': '2018-07-03', 'pre_modis_file': 'MOD09GA.A2018184.h19v05.006.2018186221221', 'pre_sen2_file': 'S2A_MSIL2A_20180703T092031_N0208_R093_T34SEG_20180703T121025'}

D 2018/0/clc_100_mask (1, 1626, 2040) {} X

D 2018/0/label (1, 1626, 2040) {} X

D 2018/0/mod_500_cloud_post (1, 1626, 2040) {}
D 2018/0/mod_500_cloud_pre (1, 1626, 2040) {}
D 2018/0/mod_500_post (7, 1626, 2040) {} X (Lower quality)
D 2018/0/mod_500_pre (7, 1626, 2040) {} X (Lower quality)

D 2018/0/sea_mask (1, 1626, 2040) {} X

D 2018/0/sen2_60_cloud_post (1, 1626, 2040) {}
D 2018/0/sen2_60_cloud_pre (1, 1626, 2040) {}
D 2018/0/sen2_60_post (11, 1626, 2040) {} X (Higher quality)
D 2018/0/sen2_60_pre (11, 1626, 2040) {} X (Higher quality)
"""

if GET_SAMPLE_LAYERS:
    # Open the file in read mode
    with h5py.File(PATH_H5, 'r') as h5_file:
        # Access the group '2018/0'
        group = h5_file['2017']['0']

        # Iterate over all datasets in the group
        for dataset_name in group:
            dataset = group[dataset_name]
            shape = dataset.shape
            num_layers = shape[0]
            print(f"Dataset: {dataset_name}, shape: {shape}")

            # Iterate over each layer in the dataset
            for i in range(num_layers):
                # Read the ith layer
                layer = dataset[i, :, :]

                # Display the layer as an image
                plt.imshow(layer, cmap='gray')
                plt.title(f"{dataset_name} - Layer {i}")
                plt.axis('off')  # Hide axis for better visualization
                plt.show(block=False)

                # Wait for the user to press Enter to proceed
                input("Press Enter to proceed to the next image...")
                plt.close()

if GET_PICTURES:
    # Define custom colormap for the labels
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', [(0, 0, 0, 10), (0.09019607843137255, 0.7450980392156863, 0.8117647058823529, 1.0), (0.8647058823529412, 0.30980392156862746, 0.45882352941176474, 1.0)], 3)
    cmap_sea = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap clouds', [(0, 0, 0, 1.0), (1.0, 1.0, 1.0, 1.0)], 2)
    cmap_clc = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap clc',
        [
            (0, 0, 0, 1.0),            # NODATA (Black)
            (0.902, 0.0, 0.302, 1.0),  # Continuous urban fabric (Dark Red)
            (1.0, 0.0, 0.0, 1.0),      # Discontinuous urban fabric (Red)
            (0.8, 0.302, 0.949, 1.0),  # Industrial or commercial units (Purple)
            (0.8, 0.0, 0.0, 1.0),      # Road and rail networks and associated land (Dark Red)
            (0.902, 0.8, 0.8, 1.0),    # Port areas (Light Pink)
            (0.901, 0.8, 0.902, 1.0),  # Airports (Lavender)
            (0.651, 0.0, 0.8, 1.0),    # Mineral extraction sites (Deep Purple)
            (0.651, 0.302, 1.0, 1.0),  # Dump sites (Violet)
            (1.0, 0.302, 1.0, 1.0),    # Construction sites (Magenta)
            (1.0, 0.651, 1.0, 1.0),    # Green urban areas (Light Magenta)
            (1.0, 0.902, 1.0, 1.0),    # Sport and leisure facilities (Pale Pink)
            (1.0, 1.0, 0.659, 1.0),    # Non-irrigated arable land (Light Yellow)
            (1.0, 1.0, 0.0, 1.0),      # Permanently irrigated land (Yellow)
            (0.902, 0.902, 0.0, 1.0),  # Rice fields (Golden Yellow)
            (0.902, 0.502, 0.0, 1.0),  # Vineyards (Orange)
            (0.949, 0.651, 0.302, 1.0),# Fruit trees and berry plantations (Light Orange)
            (0.902, 0.651, 0.0, 1.0),  # Olive groves (Mustard)
            (0.902, 0.902, 0.302, 1.0),# Pastures (Light Golden)
            (1.0, 0.902, 0.651, 1.0),  # Annual crops with permanent crops (Peach)
            (1.0, 0.902, 0.302, 1.0),  # Complex cultivation patterns (Light Orange)
            (0.902, 0.8, 0.302, 1.0),  # Agriculture with natural vegetation (Tan)
            (0.949, 0.8, 0.651, 1.0),  # Agro-forestry areas (Beige)
            (0.502, 1.0, 0.0, 1.0),    # Broad-leaved forest (Bright Green)
            (0.0, 0.651, 0.0, 1.0),    # Coniferous forest (Green)
            (0.302, 1.0, 0.0, 1.0),    # Mixed forest (Lime Green)
            (0.8, 0.949, 0.302, 1.0),  # Natural grasslands (Yellow-Green)
            (0.651, 1.0, 0.502, 1.0),  # Moors and heathland (Light Green)
            (0.651, 0.902, 0.302, 1.0),# Sclerophyllous vegetation (Olive Green)
            (0.651, 0.949, 0.0, 1.0),  # Transitional woodland-shrub (Chartreuse)
            (0.902, 0.902, 0.902, 1.0),# Beaches, dunes, sands (Light Gray)
            (0.8, 0.8, 0.8, 1.0),      # Bare rocks (Gray)
            (0.8, 1.0, 0.8, 1.0),      # Sparsely vegetated areas (Pale Green)
            (0.0, 0.0, 0.0, 1.0),      # Burnt areas (Black)
            (0.651, 0.902, 0.8, 1.0),  # Glaciers and perpetual snow (Pale Cyan)
            (0.651, 0.651, 1.0, 1.0),  # Inland marshes (Light Blue)
            (0.302, 0.302, 1.0, 1.0),  # Peat bogs (Blue)
            (0.8, 0.8, 1.0, 1.0),      # Salt marshes (Soft Blue)
            (0.902, 0.902, 1.0, 1.0),  # Salines (Very Light Blue)
            (0.651, 0.651, 0.902, 1.0),# Intertidal flats (Lavender Blue)
            (0.0, 0.8, 0.949, 1.0),    # Water courses (Cyan)
            (0.0, 1.0, 0.651, 1.0),    # Coastal lagoons (Aquamarine)
            (0.651, 1.0, 0.902, 1.0),  # Estuaries (Pale Turquoise)
            (0.902, 0.949, 1.0, 1.0),  # Sea and ocean (Very Light Cyan)
        ],
        N=44  # Number of colors
    )

    def scale_image(img):
        img = img.astype(np.float32)
        return img / img.max()

    event_id = '0'
    bands = 'nrg'  # 'nrg' for NIR-R-G composites, 'rgb' for R-G-B composites

    if bands == 'nrg':
        # Get band indices for R, G, B
        sen2_plot_bands = [3, 2, 1]
        mod_plot_bands = [0, 3, 2]
    else:
        # Get band indices for NIR, R, G
        sen2_plot_bands = [10, 3, 2]
        mod_plot_bands = [1, 0, 3]

    fig, ax = plt.subplots(2, 4, figsize=(15, 8))

    # Open the file in read mode
    with h5py.File(PATH_H5, 'r') as h5_file:
        # Access the group '2018/0'
        group = h5_file['2017'][EVENT]

        # MODIS pre-fire image
        img = group['mod_500_pre'][:][mod_plot_bands, ...]
        img = scale_image(img)
        img = np.moveaxis(img, 0, -1)
        ax[0, 0].imshow(img)
        ax[0, 0].set_title('MODIS pre-fire')

        # MODIS post-fire image
        img = group['mod_500_post'][:][mod_plot_bands, ...]
        img = scale_image(img)
        img = np.moveaxis(img, 0, -1)
        ax[0, 1].imshow(img)
        ax[0, 1].set_title('MODIS post-fire')

        # Sentinel-2 pre-fire image
        img = group['sen2_60_pre'][:][sen2_plot_bands, ...]
        img = scale_image(img)
        img = np.moveaxis(img, 0, -1)
        ax[0, 2].imshow(img * 7)
        ax[0, 2].set_title('Sentinel-2 pre-fire')

        # Sentinel-2 post-fire image
        img = group['sen2_60_post'][:][sen2_plot_bands, ...]
        img = scale_image(img)
        img = np.moveaxis(img, 0, -1)
        ax[0, 3].imshow(img * 7)
        ax[0, 3].set_title('Sentinel-2 post-fire')

        # CLC mask
        img = group['clc_100_mask'][:]
        img[(img == 48) | (img == 128)] = 0  # NODATA
        img = np.moveaxis(img, 0, -1)
        ax[1, 0].imshow(img, vmin=0, vmax=43, cmap=cmap_clc)
        ax[1, 0].set_title('CLC mask')

        # Sea mask
        img = group['sea_mask'][:]
        ax[1, 1].imshow(img.squeeze(), vmin=0, vmax=1, cmap=cmap_sea)
        ax[1, 1].set_title('Sea mask')

        # Label
        img = group['label'][:]
        ax[1, 2].imshow(img.squeeze(), vmin=0, vmax=2, cmap=cmap)
        ax[1, 2].set_title('Label')

        # Remove axes and ticks
        for i in range(2):
            for j in range(4):
                # Remove all axis labels
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])

                ax[i, j].spines['top'].set_visible(False)
                ax[i, j].spines['right'].set_visible(False)
                ax[i, j].spines['bottom'].set_visible(False)
                ax[i, j].spines['left'].set_visible(False)

        plt.subplots_adjust(wspace=0.05, hspace=0.01)

        plt.show(block=False)

        # Wait for the user to press Enter to proceed
        input("Press Enter to close")
        plt.close()