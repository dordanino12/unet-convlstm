import numpy as np
import pandas as pd
import scipy.io as sio
import glob
import time
import matplotlib.pyplot as plt

import h5py, os, glob

# Define input directory and format
directory = '/wdata/yaelsc/AirMSPI_raw_data/raw_data/'
format_ = '*.hdf'
paths = sorted(glob.glob(directory + '/' + format_))

# Define and create an output directory for the saved images
output_dir = '/home/danino/PycharmProjects/pythonProject/airmspi/saved_airmspi_images'
os.makedirs(output_dir, exist_ok=True)

images = []
titles = []

for path in paths:
    f = h5py.File(path, 'r')
    channels_data = f['HDFEOS']['GRIDS']

    sun_azimuth = np.array(channels_data['660nm_band']['Data Fields']['Sun_azimuth'])
    sun_zenith = np.array(channels_data['660nm_band']['Data Fields']['Sun_zenith'])

    image = np.array(channels_data['660nm_band']['Data Fields']['I'])
    image = np.dstack((image, channels_data['555nm_band']['Data Fields']['I']))
    image = np.dstack((image, channels_data['445nm_band']['Data Fields']['I']))

    # Handle invalid pixels
    image[image == -999] = 0
    images.append(np.array(image))

    title = path.split('/')[-1].split('.')[0].split('ELLIPSOID_')[-1]
    titles.append(title)

    sun_azimuth = sun_azimuth[image[..., 0] > 0].ravel()
    sun_zenith = 180 - sun_zenith[image[..., 0] > 0].ravel()

    print(f'---------------{path}-------------------')
    print(f'sun azimuth {sun_azimuth}')
    print(f'sun zenith {sun_zenith}')
    print("------------------------------------")

for image, title in zip(images, titles):
    f, ax = plt.subplots(1, 1, figsize=(20, 20))
    image -= image.min()
    ax.imshow(image / image.max())

    ax.set_title(title)

    # Save the figure to the output directory instead of displaying it
    save_path = os.path.join(output_dir, f"{title}.png")
    plt.savefig(save_path, bbox_inches='tight')

    # Close the figure to free up memory
    plt.close(f)
    print(f"Successfully saved: {save_path}")