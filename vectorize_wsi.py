"""
This module extracts all valid patches from a mr-image and builds a numpy array with them for later fast processing.
"""

import multiresolutionimageinterface as mri  # see https://github.com/computationalpathologygroup/ASAP
from os.path import basename, dirname, join, exists, splitext
import sys
from tqdm import tqdm
import numpy as np
from skimage.transform import downscale_local_mean
from matplotlib.image import imsave
import time


def vectorize_wsi(image_path, mask_path, output_pattern, image_level, mask_level, patch_size, stride, downsample=1):
    """
    Converts a whole-slide image into a numpy array with valid tissue patches for fast processing. It writes the
    following files for a given output pattern of '/path/normal_001_{item}.npy':
        * Patches: '/path/normal_001_patches.npy'
        * Indexes x: '/path/normal_001_x_idx.npy'
        * Indexes y: '/path/normal_001_y_idx.npy'
        * Image shape: '/path/normal_001_im_shape.npy'
        * Sanity check: '/path/normal_001_{item}.png'

    :param image_path: full path to whole-slide image file.
    :param mask_path: full path to tissue-background mask file corresponding to the image
        (see https://doi.org/10.1109/ISBI.2017.7950590).
    :param output_pattern: full path to output files using the tag {item}. For example: '/path/normal_001_{item}.npy'.
    :param image_level: magnification level to read the image.
    :param mask_level: magnification level to read the mask.
    :param patch_size: size of the stored patches in pixels.
    :param stride: size of the stride used among patches in pixels (same as patch_size for no overlapping).
    :param downsample: integer indicating downsampling ratio for patches.
    :return: nothing.
    """

    # Read slide
    si = SlideIterator(
        image_path=image_path,
        mask_path=mask_path,
        image_level=image_level,
        mask_level=mask_level,
        threshold_mask=0.6,
        load_data=True
    )

    # Process it
    si.save_array(
        patch_size=patch_size,
        stride=stride,
        output_pattern=output_pattern,
        downsample=downsample
    )


class SlideIterator(object):
    """
    Loads a pair of mr-image and mask at a given level and yields valid tissue patches.
    """

    def __init__(self, image_path, mask_path, image_level, mask_level, threshold_mask=0.6, load_data=True):
        """
        Loads a pair of mr-image and mask at a given level and yields valid tissue patches.

        Args:
            image_path (str): path to mr-image.
            mask_path (str): path to mask.
            image_level (int): image level to extract patches from.
            mask_level (int): mask level to get valid patches from.
            threshold_mask (float): ratio of tissue that must be present in the mask patch to qualify as valid.
            load_data (bool): True to immediately load the mr-image and mask.
        """

        self.image_path = image_path
        self.mask_path = mask_path
        self.image_level = image_level
        self.mask_level = mask_level
        self.image_shape = None
        self.threshold_mask = threshold_mask

        if load_data:
            self.load_data()

    def load_data(self):
        """
        Create readers for the image and mask files.
        """

        # Check image
        if not exists(self.image_path):
            raise Exception('WSI file does not exist in: %s' % str(self.image_path))

        # Load image
        image_reader = mri.MultiResolutionImageReader()
        self.image = image_reader.open(self.image_path)
        self.image_shape = self.image.getLevelDimensions(self.image_level)
        self.image_level_multiplier = self.image.getLevelDimensions(0)[0] // self.image.getLevelDimensions(1)[0]

        # Load mask
        mask_reader = mri.MultiResolutionImageReader()
        self.mask = mask_reader.open(self.mask_path)
        self.mask_shape = self.mask.getLevelDimensions(self.mask_level)
        self.mask_level_multiplier = self.mask.getLevelDimensions(0)[0] // self.mask.getLevelDimensions(1)[0]

        # Check dimensions
        if self.image_shape != self.mask_shape:
            self.image_mask_ratio = int(self.image_shape[0] / self.mask_shape[0])
        else:
            self.image_mask_ratio = 1

    def get_image_shape(self, stride):
        """
        Returns the image shape divided by the specified stride.

        Args:
            stride (int): pixels to ignore between patches.

        Returns: tuple or None.
        """

        if self.image_shape is not None:
            return (self.image_shape[0] // stride, self.image_shape[1] // stride)
        else:
            return None

    def iterate_patches(self, patch_size, stride, downsample=1):
        """
        Creates an iterator across valid patches in the mr-image (only non-empty mask patches qualify). It yields
        a tuple with:
            * Image patch: in [0, 255] uint8 [x, y, c] format.
            * Location of the patch in the image: index x divided by the stride.
            * Location of the patch in the image: index y divided by the stride.

        Args:
            patch_size (int): size of the extract patch.
            stride (int): pixels to ignore between patches.
            downsample (int): downsample patches and indexes (useful for half-level images).
        """

        # Iterate through all image patches
        self.feature_shape = self.get_image_shape(stride)
        for index_y in tqdm(range(0, self.image_shape[1], stride)):
            for index_x in range(0, self.image_shape[0], stride):

                # Avoid numerical issues by using the feature size
                if (index_x // stride >= self.feature_shape[0]) or (index_y // stride >= self.feature_shape[1]):
                    continue

                # Retrieve mask patch
                mask_tile = self.mask.getUCharPatch(
                    int((index_x * (self.mask_level_multiplier ** self.mask_level)) / self.image_mask_ratio),
                    int((index_y * (self.mask_level_multiplier ** self.mask_level)) / self.image_mask_ratio),
                    patch_size,
                    patch_size,
                    self.mask_level
                )[:, :, 0].flatten()

                # Remove 255 values (filling).
                mask_tile[mask_tile == 255] = 0

                # Continue only if it is full of tissue.
                if mask_tile.mean() > self.threshold_mask:

                    # Retrieve image patch
                    image_tile = self.image.getUCharPatch(
                        int(index_x * (self.image_level_multiplier ** self.image_level)),
                        int(index_y * (self.image_level_multiplier ** self.image_level)),
                        patch_size,
                        patch_size,
                        self.image_level
                    ).astype('uint8')

                    # Downsample
                    if downsample != 1:
                        image_tile = downscale_local_mean(image_tile, (downsample, downsample, 1)).astype('uint8')
                        # image_tile = image_tile[::downsample, ::downsample, :]  # faster

                    # Yield
                    yield (image_tile, index_x // stride, index_y // stride)

    def save_array(self, patch_size, stride, output_pattern, downsample=1):
        """
        Iterates over valid patches and save them (and the indexes) as a uint8 numpy array to disk. This function
        writes the following files given an output pattern of '/path/normal_001_{item}.npy':
            * Lock: '/path/normal_001_{item}.lock'
            * Patches: '/path/normal_001_patches.npy'
            * Indexes x: '/path/normal_001_x_idx.npy'
            * Indexes y: '/path/normal_001_y_idx.npy'
            * Image shape: '/path/normal_001_im_shape.npy'
            * Sanity check: '/path/normal_001_{item}.png'

        Args:
            patch_size (int): size of the extract patch.
            stride (int): pixels to ignore between patches.
            output_pattern (str): path to write output files.
            downsample (int): downsample patches and indexes (useful for half-level images).
        """

        # Paths
        filename = splitext(basename(output_pattern))[0]
        safety_path = join(dirname(output_pattern), filename + '.png')

        # Save image shape
        image_shape = self.get_image_shape(stride)

        # Iterate through patches
        image_tiles = []
        xs = []
        ys = []
        for image_tile, x, y in self.iterate_patches(patch_size, stride, downsample=downsample):
            image_tiles.append(image_tile)
            xs.append(x)
            ys.append(y)

        # Concat
        image_tiles = np.stack(image_tiles, axis=0).astype('uint8')
        xs = np.array(xs)
        ys = np.array(ys)

        # Store
        np.save(output_pattern.format(item='patches'), image_tiles)
        np.save(output_pattern.format(item='x_idx'), xs)
        np.save(output_pattern.format(item='y_idx'), ys)
        np.save(output_pattern.format(item='im_shape'), image_shape)

        # Safety check
        check_image = np.zeros(image_shape[::-1])
        for x, y in zip(xs, ys):
            check_image[y, x] = 1
        imsave(safety_path, check_image)


if __name__ == '__main__':

    # Vectorize slide
    vectorize_wsi(
        image_path=sys.argv[1],
        mask_path=sys.argv[2],
        output_pattern=sys.argv[3],
        image_level=sys.argv[4],
        mask_level=sys.argv[5],
        patch_size=int(sys.argv[6]),
        stride=int(sys.argv[7]),
        downsample=int(sys.argv[8])
    )

