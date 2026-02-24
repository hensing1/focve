from __future__ import annotations

import pathlib
import sys

import cv2
import imageio.v3 as iio
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
from undistort import (
    apply_camera_matrix,
    apply_inverse_camera_matrix,
    create_image_points,
    distort_points,
    remap,
    undistort_image,
)


def test_create_image():

    img = create_image_points(4, 4)
    img_ref = np.array(
        [
            [[0, 0], [1, 0], [2, 0], [3, 0]],
            [[0, 1], [1, 1], [2, 1], [3, 1]],
            [[0, 2], [1, 2], [2, 2], [3, 2]],
            [[0, 3], [1, 3], [2, 3], [3, 3]],
        ]
    )
    assert np.array_equal(img, img_ref)
    assert img.dtype == np.float32 or img.dtype == np.float64

    img = create_image_points(3, 5)
    img_ref = np.array(
        [
            [[0, 0], [1, 0], [2, 0]],
            [[0, 1], [1, 1], [2, 1]],
            [[0, 2], [1, 2], [2, 2]],
            [[0, 3], [1, 3], [2, 3]],
            [[0, 4], [1, 4], [2, 4]],
        ]
    )
    assert np.array_equal(img, img_ref)
    assert img.dtype == np.float32 or img.dtype == np.float64


def test_apply_inverse_camera_matrix():
    camera_matrix = np.array(
        [[22846.243, 0, 2044.271], [0, 22866.620, 1428.173], [0, 0, 1]]
    )
    image_coords = np.array(
        [
            [0.66832909, 0.30657991],
            [0.21363264, 0.87429444],
            [0.85240765, 0.66249069],
            [0.12145142, 0.03962581],
        ]
    )[None]
    camera_coords = np.array(
        [
            [-0.08945027, -0.06244327],
            [-0.08947018, -0.06241844],
            [-0.08944222, -0.0624277],
            [-0.08947421, -0.06245494],
        ]
    )[None]

    camera_coords_test = apply_inverse_camera_matrix(camera_matrix, image_coords)
    assert np.allclose(camera_coords, camera_coords_test, rtol=1e-3)
    image_coords_test = apply_camera_matrix(camera_matrix, camera_coords)
    assert np.allclose(image_coords, image_coords_test, rtol=1e-3)


def test_distort_points_radial_simple():

    image_coords = np.array(
        [
            [-0.0729448, -0.04908045],
            [0.57204673, -0.33461683],
            [0.56400536, -0.5652122],
            [0.83975392, -0.08028769],
            [-0.99125287, -0.01185885],
            [0.69663479, 0.43186507],
            [0.60125765, -0.3432707],
            [-0.23214581, 0.42506322],
        ]
    )[None]

    dist_coeffs = np.array([0.5, -0.1, 0, 0, 0, 0, 0, 0])
    dist_coords = np.array(
        [
            [-0.07322629, -0.04926985],
            [0.68663499, -0.40164485],
            [0.72087459, -0.72241709],
            [1.0960252, -0.10478943],
            [-1.38258663, -0.01654057],
            [0.89919613, 0.557439],
            [0.73154751, -0.41765593],
            [-0.25809574, 0.47257801],
        ]
    )[None]
    dist_coords_test = distort_points(image_coords, dist_coeffs)
    assert np.allclose(dist_coords_test, dist_coords, rtol=1e-3)

    dist_coeffs = np.array([-0.3, 0.2, 0, 0, 0, 0, 0, 0])
    dist_coords = np.array(
        [
            [-0.07277652, -0.04896722],
            [0.51874261, -0.30343676],
            [0.5019807, -0.50305482],
            [0.74552884, -0.07127896],
            [-0.89047416, -0.01065318],
            [0.61911523, 0.38380834],
            [0.54242505, -0.30968193],
            [-0.21836413, 0.39982872],
        ]
    )[None]
    dist_coords_test = distort_points(image_coords, dist_coeffs)
    assert np.allclose(dist_coords_test, dist_coords, rtol=1e-3)

    dist_coeffs = np.array([0.2, 0.0, 0, 0, 0, 0, 0, 0])
    dist_coords = np.array(
        [
            [-0.07305757, -0.04915633],
            [0.62229599, -0.36400997],
            [0.63592359, -0.63728432],
            [0.9592732, -0.09171476],
            [-1.18607825, -0.01418964],
            [0.79023568, 0.48989111],
            [0.65889968, -0.37617976],
            [-0.24303672, 0.44500467],
        ]
    )[None]
    dist_coords_test = distort_points(image_coords, dist_coeffs)
    assert np.allclose(dist_coords_test, dist_coords, rtol=1e-3)

    dist_coeffs = np.array([-0.2, 0.0, 0, 0, 0, 0, 0, 0])
    dist_coords = np.array(
        [
            [-0.07283203, -0.04900457],
            [0.52179747, -0.30522369],
            [0.49208713, -0.49314008],
            [0.72023464, -0.06886062],
            [-0.79642749, -0.00952806],
            [0.6030339, 0.37383903],
            [0.54361562, -0.31036164],
            [-0.2212549, 0.40512177],
        ]
    )[None]
    dist_coords_test = distort_points(image_coords, dist_coeffs)
    assert np.allclose(dist_coords_test, dist_coords, rtol=1e-3)

    dist_coeffs = np.array([0.0, 0.1, 0, 0, 0, 0, 0, 0])
    dist_coords = np.array(
        [
            [-0.07294524, -0.04908074],
            [0.58308161, -0.34107165],
            [0.5869317, -0.5881876],
            [0.88228084, -0.08435363],
            [-1.08698255, -0.01300411],
            [0.72807568, 0.45135623],
            [0.61507288, -0.3511581],
            [-0.23342315, 0.42740206],
        ]
    )[None]
    dist_coords_test = distort_points(image_coords, dist_coeffs)
    assert np.allclose(dist_coords_test, dist_coords, rtol=1e-3)

    dist_coeffs = np.array([0.0, -0.1, 0, 0, 0, 0, 0, 0])
    dist_coords = np.array(
        [
            [-0.07294436, -0.04908016],
            [0.56101185, -0.32816201],
            [0.54107902, -0.5422368],
            [0.797227, -0.07622175],
            [-0.89552319, -0.01071359],
            [0.6651939, 0.41237391],
            [0.58744242, -0.3353833],
            [-0.23086847, 0.42272438],
        ]
    )[None]
    dist_coords_test = distort_points(image_coords, dist_coeffs)
    assert np.allclose(dist_coords_test, dist_coords, rtol=1e-3)


def test_distort_points_radial_full():

    image_coords = np.array(
        [
            [-0.0729448, -0.04908045],
            [0.57204673, -0.33461683],
            [0.56400536, -0.5652122],
            [0.83975392, -0.08028769],
            [-0.99125287, -0.01185885],
            [0.69663479, 0.43186507],
            [0.60125765, -0.3432707],
            [-0.23214581, 0.42506322],
        ]
    )[None]

    dist_coeffs = np.array(
        [0.38927189, 0.84331867, 0, 0, 0.99947509, 0.86792649, 0.24668166, 0.04502397]
    )
    dist_coords = np.array(
        [
            [-0.07267933, -0.04890183],
            [0.56634821, -0.3312835],
            [0.62656446, -0.62790516],
            [0.98561499, -0.09423326],
            [-1.46118991, -0.01748094],
            [0.79333886, 0.49181486],
            [0.60645495, -0.34623795],
            [-0.21934835, 0.40163085],
        ]
    )[None]
    dist_coords_test = distort_points(image_coords, dist_coeffs)
    assert np.allclose(dist_coords_test, dist_coords, rtol=1e-3)

    dist_coeffs = np.array(
        [0.91619281, 0.31374706, 0, 0, 0.45106749, 0.27757134, 0.06483645, 0.25908771]
    )
    dist_coords = np.array(
        [
            [-0.0733052, -0.04932295],
            [0.74260021, -0.43438152],
            [0.81176403, -0.81350101],
            [1.25191482, -0.11969381],
            [-1.64957122, -0.01973464],
            [1.01934977, 0.63192589],
            [0.7977395, -0.45544634],
            [-0.26808772, 0.49087351],
        ]
    )[None]
    dist_coords_test = distort_points(image_coords, dist_coeffs)
    assert np.allclose(dist_coords_test, dist_coords, rtol=1e-3)

    dist_coeffs = np.array(
        [0.29101255, 0.356286, 0, 0, 0.83579121, 0.29873644, 0.40853195, 0.86962789]
    )
    dist_coords = np.array(
        [
            [-0.07294023, -0.04907737],
            [0.56476632, -0.33035818],
            [0.55155111, -0.5527313],
            [0.81835918, -0.07824217],
            [-0.95569562, -0.01143346],
            [0.68014443, 0.42164219],
            [0.59249089, -0.33826557],
            [-0.23106829, 0.42309026],
        ]
    )[None]
    dist_coords_test = distort_points(image_coords, dist_coeffs)
    assert np.allclose(dist_coords_test, dist_coords, rtol=1e-3)

    dist_coeffs = np.array(
        [0.41130837, 0.44598225, 0, 0, 0.21812795, 0.67338278, 0.49978374, 0.1316857]
    )
    dist_coords = np.array(
        [
            [-0.07279757, -0.04898139],
            [0.52388047, -0.30644214],
            [0.50764013, -0.50872636],
            [0.75359941, -0.07205058],
            [-0.89189728, -0.01067021],
            [0.62601903, 0.38808822],
            [0.54806597, -0.31290245],
            [-0.2197638, 0.40239153],
        ]
    )[None]
    dist_coords_test = distort_points(image_coords, dist_coeffs)
    assert np.allclose(dist_coords_test, dist_coords, rtol=1e-3)


def test_distort_points_tangential():

    image_coords = np.array(
        [
            [-0.0729448, -0.04908045],
            [0.57204673, -0.33461683],
            [0.56400536, -0.5652122],
            [0.83975392, -0.08028769],
            [-0.99125287, -0.01185885],
            [0.69663479, 0.43186507],
            [0.60125765, -0.3432707],
            [-0.23214581, 0.42506322],
        ]
    )[None]

    dist_coeffs = np.array([0, 0, 0.41306679, 0.28248623, 0, 0, 0, 0])
    dist_coords = np.array(
        [
            [-0.06479735, -0.04187475],
            [0.72286093, -0.16883962],
            [0.66047102, -0.21803732],
            [1.38349191, 0.180898],
            [-0.14880398, 0.40082884],
            [1.40913643, 1.03341971],
            [0.77040066, -0.16452872],
            [-0.21695567, 0.61547166],
        ]
    )[None]

    dist_coords_test = distort_points(image_coords, dist_coeffs)
    assert np.allclose(dist_coords_test, dist_coords, rtol=1e-3)

    dist_coeffs = np.array([0, 0, 0.57896195, 0.43097122, 0, 0, 0, 0])
    dist_coords = np.array(
        [
            [-0.06088156, -0.03872996],
            [0.82174598, -0.1156724],
            [0.74383787, -0.10094162],
            [1.67620802, 0.28107091],
            [0.29281329, 0.56739539],
            [1.75282916, 1.29609476],
            [0.88045455, -0.10720404],
            [-0.19886123, 0.68502921],
        ]
    )[None]
    dist_coords_test = distort_points(image_coords, dist_coeffs)
    assert np.allclose(dist_coords_test, dist_coords, rtol=1e-3)

    dist_coeffs = np.array([0, 0, 0.40248511, 0.34043624, 0, 0, 0, 0])
    dist_coords = np.array(
        [
            [-0.06380847, -0.04159259],
            [0.79029076, -0.19804196],
            [0.74103257, -0.26849172],
            [1.50788906, 0.16541711],
            [0.02177735, 0.39178942],
            [1.49794692, 1.05723251],
            [0.84444584, -0.19601586],
            [-0.19502794, 0.59772911],
        ]
    )[None]
    dist_coords_test = distort_points(image_coords, dist_coeffs)
    assert np.allclose(dist_coords_test, dist_coords, rtol=1e-3)

    dist_coeffs = np.array([0, 0, 0.51933497, 0.57233601, 0, 0, 0, 0])
    dist_coords = np.array(
        [
            [-0.05871139, -0.03846592],
            [0.99918111, -0.20933269],
            [0.96192034, -0.26718456],
            [1.98422529, 0.21880751],
            [0.70813893, 0.51210534],
            [1.94912993, 1.31885498],
            [1.07504023, -0.20819195],
            [-0.13869688, 0.62159702],
        ]
    )[None]
    dist_coords_test = distort_points(image_coords, dist_coeffs)
    assert np.allclose(dist_coords_test, dist_coords, rtol=1e-3)


def test_remap():
    width = 18
    height = 14
    image = np.arange(1, width * height + 1).astype(float).reshape(height, width, 1)

    points = np.array(
        [
            [-100, 4],
            [4, -100],
            [100, 4],
            [4, 100],
            [10.43060322, 10.31368403],
            [17.05036886, 12.58865645],
            [13.32739618, 4.79259189],
            [0.65746862, 13.30450043],
            [7.93875813, 0.07573221],
            [1.6563451, 0.15927703],
            [12.52195008, 12.11830958],
            [9.64733111, 7.84531167],
        ]
    ).reshape(-1, 1, 2)

    output_image = np.array(
        [
            [0.0],
            [0.0],
            [0.0],
            [0.0],
            [191.0],
            [252.0],
            [104.0],
            [236.0],
            [9.0],
            [3.0],
            [230.0],
            [155.0],
        ]
    ).reshape(-1, 1, 1)

    output_image_test = remap(image, points)

    assert np.array_equal(output_image, output_image_test)


def test_undistort_image():

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
    root = pathlib.Path(__file__).parents[1]
    image = iio.imread(root / "assets" / "images" / "left03.jpg")
    if image.ndim == 2:
        # Add color channel dimension
        image = np.stack([image] * 3, axis=-1)

    # Run undistortion...
    image_undist = undistort_image(image, dist_coeffs, camera_matrix)

    # Perform undistortion with OpenCV
    image_size = (image.shape[1], image.shape[0])
    rectify_map = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, np.eye(3), camera_matrix, image_size, cv2.CV_32FC1
    )
    image_undist_ref = cv2.remap(image, *rectify_map, cv2.INTER_NEAREST)

    diff = np.abs(
        np.mean(image_undist.astype(float), axis=-1)
        - np.mean(image_undist_ref.astype(float), axis=-1)
    )
    diff_count = np.sum(diff > 0)
    assert diff_count < 10
