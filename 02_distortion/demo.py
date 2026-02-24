# /// script
# dependencies = [
#   "numpy",
#   "torch",
#   "opencv-contrib-python",
#   "imageio",
# ]
# ///

import numpy as np
import imageio.v3 as iio

from src.undistort import *

# Parameters for the camera used to capture the input images...
camera_matrix = np.array(
    [
        [5.3051028302789996e02, 0.0, 3.4179070548595206e02],
        [0.0, 5.3079385131369861e02, 2.3501869772968118e02],
        [0.0, 0.0, 1.0],
    ]
)
dist_coeffs = np.array(
    [
        2.6746117151856989e01,
        2.7832211569727447e00,
        8.9604039202972668e-04,
        -2.4471842150101253e-04,
        -1.8446151393836687e00,
        2.6918574199511411e01,
        1.1246374899558578e01,
        -3.6729116135011166e00,
    ]
)

input_filename = "assets/images/left03.jpg"

image = iio.imread(input_filename)
if image.ndim == 2:
    # Add color channel dimension
    image = np.stack([image] * 3, axis=-1)

# Perform undistortion
image_undist = undistort_image(image, dist_coeffs, camera_matrix)

if image_undist.ndim == 3 and image_undist.shape[-1] == 1:
    image_undist = image_undist[...,0]

iio.imwrite("demo_original.png", image)
iio.imwrite("demo_undistorted.png", image_undist.astype(np.uint8))
