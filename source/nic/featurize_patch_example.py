from PIL import Image
import numpy as np
import keras
import matplotlib
import matplotlib.pyplot as plt


from nic.featurize_wsi import add_downsample_to_encoder
matplotlib.use('TkAgg')

class ImageNormalizerMinusOneToOne(object):
    def __call__(self, patch):
        return (((patch / 255.0) * 2)-1).astype(np.float32)

    def reverse(self, patch):
        patch = np.clip(((patch+1)*255/2.0)+0.5,0, 255).astype(np.uint8)
        return patch

def load_encoder(path, downsampling=True, print_summary=False):
    """ The encoder works best at 0.5 spacing (20 magnification) with patch size 128x128
    (internally downsampled to 64, which corresponds to 64x64@1.0 spacing)"""
    encoder = keras.models.load_model(filepath=path)
    if print_summary:
        encoder.summary()
    if downsampling:
        encoder = add_downsample_to_encoder(encoder)
    return encoder

if __name__ == '__main__':
    patch_path = '../example_128.jpg'
    # model_path = '../../models/encoders_patches_pathology/encoder_bigan.h5'
    model_path = '../../models/encoders_patches_pathology/encoder_multitask-4.h5'

    patch = np.asarray(Image.open(patch_path))
    patch = ImageNormalizerMinusOneToOne()(patch)
    # plt.imshow(patch)
    # plt.show()

    encoder = load_encoder(model_path, print_summary=True, downsampling=True)

    patch = np.expand_dims(patch, axis=0)
    print(patch.dtype)
    features = encoder.predict(patch)
    print(f'compressed {patch.shape}->{features.shape}')






