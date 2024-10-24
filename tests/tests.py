import h5py

PATH_H5 = "Dataset #3/FLOGA_dataset_2018_sen2_60_mod_500.h5"

with h5py.File(PATH_H5, 'r') as h5_file:
    group = h5_file['2018']['7']
    cloud_mask_dataset = group['sen2_60_cloud_pre']
    
    # Access the dataset creation property list
    dcpl = cloud_mask_dataset.id.get_create_plist()
    
    # Check the number of external files
    num_external = dcpl.get_external_count()
    print(f"Number of external files: {num_external}")
    
    if num_external > 0:
        for i in range(num_external):
            name, offset, size = dcpl.get_external(i)
            print(f"External file {i}: {name}, offset: {offset}, size: {size}")
    else:
        print("Dataset does not use external storage.")
"""
import h5py

with h5py.File(PATH_H5, 'r') as h5_file:
    group = h5_file['2018']['7']
    cloud_mask_dataset = group['sen2_60_cloud_pre']
    
    # Attempt to read the entire dataset
    try:
        cloud_mask = cloud_mask_dataset[()]
        print("Successfully read the entire dataset.")
    except Exception as e:
        print("Error reading the dataset:", e)
"""