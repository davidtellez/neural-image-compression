'''
Creates the images of the synthetic dataset described in https://doi.org/10.1109/TPAMI.2019.2936841

David Tellez
2019
'''


from os.path import exists, join, dirname, basename
import os
from glob import glob
import numpy as np
from PIL import Image, ImageDraw
import sys
import multiprocessing
import pandas as pd
import json


def create_mnist_dataset(mnist_data_dir, output_dir, samples_per_purpose=25000, n_processes=30, downsample_mnist=3,
                         mnist_tile_ratio_size=400, fill_ratio=0.002, shape_factor=50):
    '''
    Creates the images of the synthetic dataset described in https://doi.org/10.1109/TPAMI.2019.2936841

    :param mnist_data_dir: directory containing MNIST digits in numpy format: uint8 Nx28x28 xtrain.npy xtest.npy,
        and Nx1 ytrain.npy ytest.npy.
    :param output_dir: directory where the images will be created and stored.
    :param samples_per_purpose: number of images in each of the data partitions (training and test).
    :param n_processes: number of CPU cores used to parallelize the generation process.
    :param downsample_mnist: downscaling factor applied to 28x28 MNIST digits
    :param mnist_tile_ratio_size: this factor defines the total size of the generated image by multiplying the size of
        the downsampled MNIST digit.
    :param fill_ratio: defines the total number of digits inserted into the generated image.
    :param shape_factor: defines the smallest possible rectangle to be drawn.
    :return: nothing.
    '''

    # Output dir
    if not exists(join(output_dir, 'train')):
        os.makedirs(join(output_dir, 'train'))
    if not exists(join(output_dir, 'test')):
        os.makedirs(join(output_dir, 'test'))

    # Size of inserted MNIST digits
    mnist_size = 28 // downsample_mnist

    # Create multiprocessing job list for training samples
    jobs = []
    xtrain_path = join(mnist_data_dir, 'xtrain.npy')
    ytrain_path = join(mnist_data_dir, 'ytrain.npy')
    for i in range(samples_per_purpose):
        jobs.append((mnist_size * mnist_tile_ratio_size, mnist_size, fill_ratio, xtrain_path, ytrain_path,
                     join(output_dir, 'train'), i, downsample_mnist, shape_factor))

    # Create training samples
    pool = multiprocessing.Pool(processes=n_processes)
    pool.map(create_tile_process_wrapper, jobs)

    # Create multiprocessing job list for test samples
    jobs = []
    xtrain_path = join(mnist_data_dir, 'xtest.npy')
    ytrain_path = join(mnist_data_dir, 'ytest.npy')
    for i in range(samples_per_purpose):
        jobs.append((mnist_size * mnist_tile_ratio_size, mnist_size, fill_ratio, xtrain_path, ytrain_path,
                     join(output_dir, 'test'), i, downsample_mnist, shape_factor))

    # Create test samples
    pool = multiprocessing.Pool(processes=n_processes)
    pool.map(create_tile_process_wrapper, jobs)

    # Create CSV files listing all images
    create_index_csv(
        input_pattern=join(output_dir, 'train', '*_*_*_*_*_tile.png'),
        output_path=join(output_dir, 'train.csv')
    )
    create_index_csv(
        input_pattern=join(output_dir, 'test', '*_*_*_*_*_tile.png'),
        output_path=join(output_dir, 'test.csv')
    )

    # Create JSON file defining data partitions
    create_json(
        input_dir=output_dir,
        output_path=join(output_dir, 'mnist_folds_set.json'),
        sample_ratio=1
    )


def create_tile_process_wrapper(params):

    '''
    This function creates a single synthetic image with its corresponding ground truth mask.
    :param params: tuple consisting of: tile_size, patch_size, fill_ratio, xtrain_path, ytrain_path, output_dir,
        tile_id, downsample and shape_factor.
    :return: nothing.
    '''

    # Reseed (avoid multiprocessing issues)
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

    # Unpack params
    tile_size, patch_size, fill_ratio, xtrain_path, ytrain_path, output_dir, tile_id, downsample, shape_factor = params

    # Read MNIST digits from source location
    xtrain = np.load(xtrain_path)
    ytrain = np.load(ytrain_path)

    # Preprocess digits
    xtrain[xtrain > 63] = 255
    if downsample != 1:
        xtrain = xtrain[:, ::downsample, ::downsample]

    # Create tile
    tile, ref_tile, circle_digit, rectangle_digit, circle_size, rectangle_size = create_tile_mnist(
        tile_size=tile_size,
        fill_ratio=fill_ratio,
        xtrain=xtrain,
        ytrain=ytrain,
        patch_size=patch_size,
        shape_factor=shape_factor
    )

    # Store
    id = '{i}_{c}_{r}_{cs}_{rs}'.format(i=tile_id, c=circle_digit, r=rectangle_digit, cs=circle_size, rs=rectangle_size)
    Image.fromarray(tile).save(join(output_dir, '{id}_tile.png'.format(id=id)))
    Image.fromarray(ref_tile).save(join(output_dir, '{id}_ref.png'.format(id=id)))


def create_tile_mnist(tile_size, fill_ratio, xtrain, ytrain, patch_size, shape_factor):

    '''
    This function creates a single synthetic image with its corresponding ground truth mask.

    :param tile_size: total size of the generated image.
    :param fill_ratio: percentage of pixels to place an MNIST digit.
    :param xtrain: array with MNIST digits.
    :param ytrain: array with labels for MNIST digits.
    :param patch_size: size of MNIST digits.
    :param shape_factor: defines the smallest possible rectangle to be drawn.
    :return: tuple with elements: tile, ref_tile, circle_digit, rectangle_digit, circle_size, rectangle_size
    '''

    # Create reference tile
    ref_tile, circle_size, rectangle_size = create_tile_mask_rectangles(
        tile_size=tile_size,
        drawing_size=tile_size // 2,
        shape_factor=shape_factor
    )
    ref_tile = np.array(ref_tile, dtype='uint8')

    # Define label for entire image
    circle_digit = np.random.randint(0, 10)
    rectangle_digit = np.mod(circle_digit + np.random.randint(1, 9), 10)

    # Set digit mask
    ref_tile[ref_tile == 0] = 10  # flag for random digit
    ref_tile[ref_tile == 127] = circle_digit
    ref_tile[ref_tile == 255] = rectangle_digit

    # Create canvas
    tile_im = Image.new('L', (tile_size, tile_size))

    # Insert digits
    tile_im = insert_digits_with_mask(tile_im, ref_tile, xtrain, ytrain, fill_ratio, patch_size=patch_size)

    # Original mask
    ref_tile[ref_tile == circle_digit] = 127
    ref_tile[ref_tile == rectangle_digit] = 255
    ref_tile[ref_tile == 10] = 0

    return np.array(tile_im, dtype='uint8'), ref_tile, circle_digit, rectangle_digit, circle_size, rectangle_size


def create_tile_mask_rectangles(tile_size, drawing_size, shape_factor=4):

    '''
    Creates the ground truth mask for the synthetic images. Each mask consists of two rectangles representing
    lesions. The position and size of the rectangles is randomized. In the code of this function, "circle"
    corresponds to the 45-degree-tilted rectangle, and "rectangle" to the non-tilted rectangle (legacy reasons).

    :param tile_size: total size of the generated image.
    :param drawing_size: max size of the long side of the rectangles.
    :param shape_factor: defines the smallest possible rectangle to be drawn.
    :return: tuple consisting of tile_im, circle_size, and rectangle_size.
    '''

    # Create canvas
    tile_im = Image.new('L', (tile_size, tile_size))

    # Draw 45-deg rectangle alias "circle"
    circle_size = np.random.randint(drawing_size // shape_factor, drawing_size)
    tile_im_rectangle = Image.new('L', (circle_size, circle_size//2))
    tile_dr_rectangle = ImageDraw.Draw(tile_im_rectangle)
    circle_x0 = np.random.randint(0, tile_size // 2 - circle_size)
    circle_y0 = np.random.randint(0, tile_size - circle_size)
    tile_dr_rectangle.rectangle((0, 0, 0 + circle_size, 0 + circle_size//2), fill=127, outline=None)
    k = np.random.randint(0, 2)
    tile_im_rectangle = tile_im_rectangle.rotate(int(k * 90 + 45), expand=1)
    tile_im.paste(tile_im_rectangle, (circle_x0, circle_y0))

    # Draw rectangle
    tile_dr = ImageDraw.Draw(tile_im)
    rectangle_size = np.random.randint(drawing_size // shape_factor, drawing_size)
    rectangle_x0 = np.random.randint(tile_size // 2, tile_size - rectangle_size)
    rectangle_y0 = np.random.randint(0, tile_size - rectangle_size)
    tile_dr.rectangle((rectangle_x0, rectangle_y0, rectangle_x0 + rectangle_size, rectangle_y0 + rectangle_size//2), fill=255, outline=None)

    # Rotate image
    k = np.random.randint(0, 4)
    tile_im = tile_im.rotate(int(k*90))

    # Flip image
    flips = [None, Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]
    flip = np.random.randint(0, 3)
    if flips[flip] is not None:
        tile_im = tile_im.transpose(flips[flip])

    return tile_im, circle_size, rectangle_size


def insert_digits_with_mask(tile, ref_tile, digits, digits_labels, fill_ratio, patch_size):

    '''
    Creates image using a given reference mask. It inserts the MNIST digits in the appropriated locations defined by
    the mask.

    :param tile: PIL Image empty canvas to be filled with MNIST digits.
    :param ref_tile: corresponding reference mask to map the MNIST digits.
    :param digits: array with MNIST digits.
    :param digits_labels: array with MNIST labels.
    :param fill_ratio: perecentage of pixels to insert a digit.
    :param patch_size: size of the MNIST digits.
    :return: synthetic image corresponding to the input mask (PIL Image format).
    '''

    # Total number of digits
    tile_size = tile.height
    n_fill = int(tile_size * tile_size * fill_ratio)

    # Sample positions
    x_positions = []
    y_positions = []
    offset = patch_size
    for x in range(0, tile_size, offset):
        for y in range(0, tile_size, offset):
            x_positions.append(x)
            y_positions.append(y)

    # Add noise to positions
    x_locs = np.clip(x_positions + np.random.uniform(-offset//2, offset//2, len(x_positions)), 0, tile_size).astype('int')
    y_locs = np.clip(y_positions + np.random.uniform(-offset//2, offset//2, len(y_positions)), 0, tile_size).astype('int')

    # Sample locations
    idxs = np.random.choice(len(x_locs), n_fill, replace=False)
    x_locs = x_locs[idxs]
    y_locs = y_locs[idxs]
    digits_idxs = np.arange(digits.shape[0])

    # Draw
    for i in range(n_fill):

        # Find label
        x = x_locs[i]
        y = y_locs[i]
        digit_label = ref_tile[x, y]

        # Select digit
        if digit_label == 10:  # flag for random digit
            idx = np.random.choice(digits_idxs)
        else:
            idx = np.random.choice(digits_idxs[digits_labels == digit_label])
        digit = digits[idx, :, :]
        digit_size = digit.shape[-1]

        # Paste
        digit_im = Image.fromarray(digit)
        tile.paste(digit_im, (y - digit_size//2, x - digit_size//2))

    return tile


def create_index_csv(input_pattern, output_path):

    '''
    Creates a CSV listing image-mask filenames.

    :param input_pattern: glob pattern indicating the filenames of the images.
    :param output_path: full path to output file.
    :return: nothing.
    '''

    # List tile files
    paths = glob(input_pattern)

    # List filenames
    filenames = []
    for path in paths:
        ref_path = path[:-8] + 'ref.png'
        if exists(ref_path):
            filenames.append((basename(path), basename(ref_path)))

    # Store as CSV
    df = pd.DataFrame(filenames, columns=['tile', 'ref']).sort_values('tile')

    # Store CSV
    df.to_csv(output_path)


def create_json(input_dir, output_path, sample_ratio=1.0):

    '''
    Creates a JSON file with the data partition used in following experiments. First, a set of training images are
    selected to be used with the encoders. Second, the remaining images are divided into 4 folds for cross-validation.
    Finally, test images are stored.

    :param input_dir: directory where to find the CSV files listing image-mask filenames.
    :param output_path: full path to output file.
    :param sample_ratio: samples less images for debugging purposes.
    :return: nothing.
    '''

    # Test images
    data = {}
    df_test = pd.read_csv(join(input_dir, 'test.csv'), header=0, index_col=0)
    df_test = df_test.sample(int(len(df_test)*sample_ratio), replace=False)
    df_test['tile_id'] = df_test['tile'].apply(lambda x: 'test-' + x[:-4])
    data['test'] = list(df_test['tile_id'])

    # Train images
    df_train = pd.read_csv(join(input_dir, 'train.csv'), header=0, index_col=0)
    df_train = df_train.sample(int(len(df_train)*sample_ratio), replace=False)
    df_train['tile_id'] = df_train['tile'].apply(lambda x: 'train-' + x[:-4])

    # Encoder selection
    n_samples_encoding = int(len(df_train) * 0.1)
    df_encoder = df_train.iloc[:n_samples_encoding, :]
    n_samples_encoding_training = int(len(df_encoder) * 0.7)
    df_encoder_train = df_encoder.iloc[:n_samples_encoding_training, :]
    df_encoder_validation = df_encoder.iloc[n_samples_encoding_training:, :]
    data['encoder'] = {}
    data['encoder']['training'] = list(df_encoder_train['tile_id'])
    data['encoder']['validation'] = list(df_encoder_validation['tile_id'])

    # Tile classification
    df_train = df_train.iloc[n_samples_encoding:, :]
    n_folds = 4
    n_samples_fold = len(df_train) // n_folds
    data['wsi_classifier'] = {}
    for i in range(n_folds):
        data['wsi_classifier']['fold_{i}'.format(i=i)] = list(df_train.iloc[:n_samples_fold, :]['tile_id'])
        df_train = df_train.iloc[n_samples_fold:, :]

    # Write json
    with open(output_path, 'w') as f:
        json.dump(data, f, sort_keys=True, indent=4)


if __name__ == '__main__':

    # Directory containing MNIST digits in numpy format
    mnist_data_dir = sys.argv[1]

    # Directory where the images will be created
    output_dir = sys.argv[2]

    # Create tiles
    create_mnist_dataset(
        mnist_data_dir=mnist_data_dir,
        output_dir=output_dir,
        samples_per_purpose=25000,
        n_processes=30,  # make sure you have enough CPU cores
        downsample_mnist=3,
        mnist_tile_ratio_size=400,
        fill_ratio=0.002,
        shape_factor=50
    )
