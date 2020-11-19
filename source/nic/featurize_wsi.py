"""
This module runs an encoder over a vectorized whole-slide image to obtain features from it (compress it).
"""

import matplotlib as mpl
mpl.use('Agg')  # plot figures when no screen available

from matplotlib import pyplot as plt
from os.path import basename, dirname, join, exists, splitext
import os
import numpy as np
import random
from scipy.ndimage.morphology import distance_transform_edt
import keras
from nic.vectorize_wsi import vectorize_wsi
import sys


def encode_wsi_npy_simple(encoder, wsi_pattern, batch_size, output_path, output_preview_pattern=None,
                          output_distance_map=True):
    """
    Featurizes a vectorized whole-slide image using a pretrained encoder.

    Args:
        encoder: model transforming a patch to a vector code.
        wsi_pattern (str): path pattern pointing to vectorized WSI.
        batch_size (int): number of patches to encode simultaneously by the GPU.
        output_path (str): path pattern to output files.
            For example: /path/normal_001_features.npy'.
        output_preview_pattern (str or None): optional path pattern to preview files.
            For example: /path/normal_001_{f_min}_{f_max}_features.png'.
        output_distance_map (bool): True to write distance map useful to extract image crops.

    """

    # # Check if encoder accepts 128x128 patches
    # if encoder.layers[0].input_shape[1] == 64:
    #     encoder = add_downsample_to_encoder(encoder)
    # elif encoder.layers[0].input_shape[1] == 128:
    #     pass
    # else:
    #     raise Exception('Model input size not supported.')

    # Read wsi
    wsi_sequence = WsiNpySequence(wsi_pattern=wsi_pattern, batch_size=batch_size)

    # Config
    xs = wsi_sequence.xs
    ys = wsi_sequence.ys
    image_shape = wsi_sequence.image_shape

    # Predict
    patch_features = encoder.predict_generator(generator=wsi_sequence, steps=len(wsi_sequence), verbose=1)
    features = np.ones((patch_features.shape[1], image_shape[1], image_shape[0])) * np.nan

    # Store each patch feature in the right spatial position
    for patch_feature, x, y in zip(patch_features, xs, ys):
        features[:, y, x] = patch_feature

    # Populate NaNs
    features[np.isnan(features)] = 0

    # Save to disk float16
    np.save(output_path, features.astype('float16'))

    # Plot
    if output_preview_pattern:
        plot_feature_map(np.copy(features), output_preview_pattern)

    # Distance map
    if output_distance_map:
        try:
            filename = splitext(basename(output_path))[0]
            output_dm_path = join(dirname(output_path), filename + '_distance_map.npy')
            distance_map = compute_single_distance_map(features.astype('float32'))
            np.save(output_dm_path, distance_map)
        except Exception as e:
            print('Failed to compute distance map for {f}. Exception: {e}.'.format(f=output_path, e=e), flush=True)


def encode_wsi_npy_advanced(encoder, wsi_sequence, output_pattern, rot_deg, flip, overwrite,
                            output_preview_pattern=None, output_distance_map=True):
    """
    Featurizes a vectorized whole-slide image taking augmentations into account. Augments indexes and patches properly.
    Whole-slide image augmentations require special attention since a given patch and its rotated version may produce
    different embedding vectors.

    Args:
        encoder: model transforming a patch to a vector code.
        wsi_sequence (WsiNpySequence): vectorized WSI in Sequence format.
        output_pattern (str): path pattern to output files. For example: /path/normal_001_{rot_deg}_{flip}_features.npy'.
        rot_deg (int): rotation degree (0, 90, 180 or 270).
        flip (str): flipping augmentation ('none', 'horizontal', 'vertical' or 'both'.
        output_preview_pattern (str or None): optional path pattern to preview files. For example: /path/normal_001_{rot_deg}_{flip}_{f_min}_{f_max}_features.png'.
        overwrite (bool): True to overwrite existing files.

    """

    # Overwrite
    output_npy_path = output_pattern.format(rot_deg=rot_deg, flip=flip)
    if output_preview_pattern:
        output_png_path = output_preview_pattern.format(rot_deg=rot_deg, flip=flip, f_min='{f_min:.3f}', f_max='{f_max:.3f}')
    else:
        output_png_path = None
    if not exists(output_npy_path) or overwrite:

        # Read data
        print('Featurizing {path}'.format(path=output_npy_path), flush=True)
        try:
            try:
                code_size = int(encoder.output.shape[-1])
            except:
                code_size = int(encoder.layers[-2].output.shape[-1])
        except:
            code_size = encoder.code_size
        xs = wsi_sequence.xs
        ys = wsi_sequence.ys
        image_shape = wsi_sequence.image_shape

        # Prepare
        features = np.ones((code_size, image_shape[1], image_shape[0])) * np.nan
        idxs = np.arange(features.shape[1] * features.shape[2]).reshape((features.shape[1], features.shape[2]))

        # Augment
        idxs_rot = rot_flip_array(idxs, axes=(0, 1), rot_deg=rot_deg, flip=flip)
        features = rot_flip_array(features, axes=(1, 2), rot_deg=rot_deg, flip=flip)
        wsi_sequence.set_rot_flip(rot_deg, flip)

        # Predict
        patch_features = encoder.predict_generator(generator=wsi_sequence, steps=len(wsi_sequence))

        # Store each patch feature in the right position
        for patch_feature, x, y in zip(patch_features, xs, ys):
            idx = idxs[y, x]
            x_rot, y_rot = [ele for ele in zip(*np.where(idxs_rot == idx))][0]
            features[:, x_rot, y_rot] = patch_feature

        # Populate NaNs
        features[np.isnan(features)] = 0

        # Save to disk float16
        np.save(output_npy_path, features.astype('float16'))

        # Plot
        if output_preview_pattern:
            plot_feature_map(np.copy(features), output_png_path)  # without copy() it modifies features!!

        # Distance map
        if output_distance_map:
            try:
                filename = splitext(basename(output_npy_path))[0]
                output_dm_path = join(dirname(output_npy_path), filename + '_distance_map.npy')
                distance_map = compute_single_distance_map(features.astype('float32'))
                np.save(output_dm_path, distance_map)
            except Exception as e:
                print('Failed to compute distance map for {f}. Exception: {e}.'.format(f=output_npy_path, e=e), flush=True)


def encode_augment_wsi(wsi_pattern, encoder, output_dir, batch_size, aug_modes, overwrite):

    """
    Featurizes a vectorized whole-slide image given a set of augmentations (convenient wrapper
    for encode_wsi_npy_advanced() function).

    Args:
        wsi_pattern (str): path pattern pointing to location of vectorized WSI. For
        example: "/path/normal_060_{item}.npy".
        encoder: Keras model transforming a patch to a vector code.
        output_dir (str): output directory to store results.
        batch_size (int): batch size.
        aug_modes (list): list of pairs rotation-flipping values.
        overwrite (bool): True to overwrite existing files.
    """

    # Prepare paths
    if not exists(output_dir):
        os.makedirs(output_dir)
    filename = splitext(basename(wsi_pattern))[0]
    output_pattern = join(output_dir, filename.format(item='{rot_deg}_{flip}_features.npy'))
    output_preview_pattern = join(output_dir, filename.format(item='{rot_deg}_{flip}_{f_min}_{f_max}_features.png'))

    # Precheck
    process = False
    for flip, rot_deg in aug_modes:
        output_npy_path = output_pattern.format(rot_deg=rot_deg, flip=flip)
        if not exists(output_npy_path):
            process = True

    # Lock
    if process or overwrite:
        print('Featurizing image {image} ...'.format(image=wsi_pattern), flush=True)

        # Read wsi
        wsi_sequence = WsiNpySequence(wsi_pattern=wsi_pattern, batch_size=batch_size)

        # Iterate through augmentations
        for flip, rot_deg in aug_modes:

            # Encode
            try:
                encode_wsi_npy_advanced(
                    encoder=encoder,
                    wsi_sequence=wsi_sequence,
                    output_pattern=output_pattern,
                    output_preview_pattern=output_preview_pattern,
                    rot_deg=rot_deg,
                    flip=flip,
                    overwrite=overwrite
                )
            except Exception as e:
                print('Failed to encode {p} with rotation {r} and flip {f}. Exception: {e}'.format(p=output_pattern, r=rot_deg, f=flip, e=e), flush=True)

    else:
        print('Ignoring image {image} ...'.format(image=wsi_pattern), flush=True)


def rot_flip_array(array, axes, rot_deg, flip):
    """
    Batch augmentation function supporting 90 degree rotations and flipping.

    Args:
        array: batch in [b, x, y, c] format.
        axes: axes to apply the transformation.
        rot_deg (int): rotation degree (0, 90, 180 or 270).
        flip (str): flipping augmentation ('none', 'horizontal', 'vertical' or 'both'.

    Returns: batch array.

    """

    # Rot
    array = aug_rot(array, degrees=rot_deg, axes=axes)

    # Flip
    if flip == 'vertical':
        array = np.flip(array, axis=axes[0])
    elif flip == 'horizontal':
        array = np.flip(array, axis=axes[1])
    elif flip == 'both':
        array = np.flip(array, axis=axes[0])
        array = np.flip(array, axis=axes[1])
    elif flip == 'none':
        pass

    return array


def aug_rot(array, degrees, axes):
    """
    90 degree rotation.

    Args:
        array: batch in [b, x, y, c] format.
        degrees (int): rotation degree (0, 90, 180 or 270).
        axes: axes to apply the transformation.

    Returns: batch array.

    """

    if degrees == 0:
        pass
    elif degrees == 90:
        array = np.rot90(array, k=1, axes=axes)
    elif degrees == 180:
        array = np.rot90(array, k=2, axes=axes)
    elif degrees == 270:
        array = np.rot90(array, k=3, axes=axes)

    return array


class WsiNpySequence(keras.utils.Sequence):

    """
    This class is a Keras sequence used to make predictions on vectorized whole-slide images.
    """

    def __init__(self, wsi_pattern, batch_size, rot_deg=0, flip='none'):
        """
        This class is a Keras sequence used to make predictions on vectorized WSIs.

        Args:
            wsi_pattern (str): path pattern pointing to location of vectorized WSI.
                For example: "/path/normal_060_{item}.npy".
            batch_size (int): batch size to process the patches.
            rot_deg (int): rotation degree (0, 90, 180 or 270).
            flip (str): flipping augmentation ('none', 'horizontal', 'vertical' or 'both'.
        """

        # Params
        self.batch_size = batch_size
        self.wsi_pattern = wsi_pattern
        self.rot_flip_fn = None
        self.rot_deg = None
        self.flip = None

        # Read data
        self.image_tiles = np.load(wsi_pattern.format(item='patches'))
        self.xs = np.load(wsi_pattern.format(item='x_idx'))
        self.ys = np.load(wsi_pattern.format(item='y_idx'))
        self.image_shape = np.load(wsi_pattern.format(item='im_shape'))
        self.n_samples = self.image_tiles.shape[0]
        self.n_batches = int(np.ceil(self.n_samples / self.batch_size))

        # Set rot flip
        self.set_rot_flip(rot_deg, flip)

    def __len__(self):
        """
        Provide length in number of batches
        Returns (int): number of batches available in the entire dataset.
        """
        return self.n_batches

    def get_batch(self, idx):
        """
        Gets batches based on index. The last batch might have smaller length than batch size.
        Args:
            idx: index in batches..

        Returns: batch of image patches in [-1, +1] [b, x, y, ch] format.

        """

        # Get samples
        idx_batch = idx * self.batch_size
        if idx_batch + self.batch_size >= self.n_samples:
            idxs = np.arange(idx_batch, self.n_samples)
        else:
            idxs = np.arange(idx_batch, idx_batch + self.batch_size)

        # Build batch
        image_tiles = self.image_tiles[idxs, ...]

        # Format
        image_tiles = (image_tiles / 255.0 * 2) - 1

        return image_tiles

    def __getitem__(self, idx):
        batch = self.get_batch(idx)
        batch = self.rot_flip_fn(batch)
        batch = self.transform(batch)
        return batch

    def set_rot_flip(self, rot_deg, flip):
        """
        Sets the augmentation function applied to the entire batch.

        Args:
            rot_deg (int): rotation degree (0, 90, 180 or 270).
            flip (str): flipping augmentation ('none', 'horizontal', 'vertical' or 'both'.
        """
        self.rot_deg = rot_deg
        self.flip = flip
        self.rot_flip_fn = lambda batch: rot_flip_array(batch, axes=(1, 2), rot_deg=rot_deg, flip=flip)

    def transform(self, batch):
        return batch


def plot_feature_map(features, output_pattern):
    """
    Preview of the featurized WSI. Draws a grid where each small image is a feature map. Normalizes the set of feature
    maps using the 3rd and 97th percentiles of the entire feature volume. Includes these values in the filename.

    Args:
        features: numpy array with format [c, x, y].
        output_pattern (str): path pattern of the form '/path/tumor_001_90_none_{f_min:.3f}_{f_max:.3f}_features.png'

    """

    # Downsample to avoid memory error
    if features.shape[1] >= 800 or features.shape[2] >= 800:
        features = features[:, ::3, ::3]
    else:
        features = features[:, ::2, ::2]

    # Get range for normalization
    f_min = np.percentile(features[features != 0], 3)
    f_max = np.percentile(features[features != 0], 97)

    # Detect background (estimate)
    features[features == 0] = np.nan

    # Normalize and clip values
    features = (features - f_min) / (f_max - f_min + 1e-6)
    features = np.clip(features, 0, 1)

    # Add background
    features[features == np.nan] = 0.5

    # Make batch
    data = features[:, np.newaxis, :, :].transpose(0, 2, 3, 1)

    # Make grid
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0), (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=0.0)
    padding = ((0, 0), (5, 5), (5, 5)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=0.5)

    # Tile the individual thumbnails into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    # Map the normalized data to colors RGBA
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=0, vmax=1)
    image = cmap(norm(data[:, :, 0]))

    # Save the image
    plt.imsave(output_pattern.format(f_min=f_min, f_max=f_max), image)


def compute_single_distance_map(features):
    """
    Computes distance to tissue map. It is useful to detect where the tissue areas are located and take crops from them.

    :param features: featurized whole-slide image.
    :return: distance map array
    """

    # Binarize
    features = features.std(axis=0)
    features[features != 0] = 1

    # Distance transform
    distance_map = distance_transform_edt(features)
    distance_map = distance_map / np.max(distance_map)
    distance_map = np.square(distance_map)
    distance_map = distance_map / np.sum(distance_map)

    return distance_map


def add_downsample_to_encoder(model):

    """
    Adds downsampling layer to input (useful for BiGAN encoder trained with 64x64 patches).
    """

    input_layer = keras.layers.Input((128, 128, 3))
    x = keras.layers.AveragePooling2D()(input_layer)
    x = model(x)

    encoder = keras.models.Model(inputs=input_layer, outputs=x)

    return encoder


if __name__ == '__main__':

    # Paths
    image_path = sys.argv[1]
    mask_path = sys.argv[2]
    output_dir = sys.argv[3]
    image_level = int(sys.argv[4])
    mask_level = int(sys.argv[5])
    patch_size = int(sys.argv[6])
    stride = int(sys.argv[7])
    downsample = int(sys.argv[8])
    model_path = sys.argv[9]
    batch_size = sys.argv[10]
    filename = splitext(basename(image_path))[0]
    output_pattern = join(output_dir, filename + '_{item}.npy')
    output_path = join(output_dir, filename + '_features.npy')
    output_preview_pattern = join(output_dir, filename + '_{f_min}_{f_max}_features.png')

    # Vectorize slide
    vectorize_wsi(
        image_path=image_path,
        mask_path=image_path,
        output_pattern=output_pattern,
        image_level=image_level,
        mask_level=mask_level,
        patch_size=patch_size,
        stride=stride,
        downsample=downsample
    )

    # Load encoder model
    encoder = keras.models.load_model(
        filepath=model_path
    )

    # Featurize (encode) image
    encode_wsi_npy_simple(
        encoder=encoder,
        wsi_pattern=output_pattern,
        batch_size=batch_size,
        output_path=output_path,
        output_preview_pattern=output_preview_pattern,
        output_distance_map=True
    )
