from metadata import *
from config import Constant
from utils import Utility
import numpy as np
import tensorflow as tf

from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras import mixed_precision
import keras.backend as K

from utils import *
# logger = Utility.logger()

U = Utility()



C = Constant()
patches_metadata = Metadata().metadata


class ClassWeights(metaclass=Singleton):
    def __init__(self):
        self._class_weights = None
        
    @property
    def class_weights(self):
        if self._class_weights is None:
            self._class_weights = self._calculate_class_weights()
        return self._class_weights
        
    def _calculate_class_weights(self, mode='uniform'):
        """
        Return the list of weights for each class, with the constraint that sum(class_weights) == 1.0
        """
        if mode == 'uniform':
            return [1 / C.NUM_CLASSES] * C.NUM_CLASSES
        
        class_weights = [0] * 20
        for path in dataFrame['Semantic_segmentation_path']:
            mask = U.upload_npy_image(path)[0]
            for c in range(C.NUM_CLASSES):
                class_weights[c] += np.divide(np.count_nonzero(mask == c), mask.shape[0] ** 2)
        class_weights = [w / len(patches_metadata['Semantic_segmentation_path']) for w in class_weights]
        
        class_weights = [1 / w for w in class_weights]
        class_weights = class_weights / sum(class_weights) * 100
        return class_weights

def normalize_patch_spectra(patch):
        """Utility function totf.py_function normalize the Sentinel-2 patch spectra.
       The patch must consist of 10 spectra and the shape n*n*10.
       """
        norms = Metadata().norm_metadata['Fold_1']
        return (patch - norms['mean']) / norms['std']

def path_to_timeseries_input(tensor) -> tuple:
    path = tf.get_static_value(tensor).decode("utf-8")
    input_patches = U.upload_npy_image(path)
    patches = input_patches.swapaxes(1,3).swapaxes(1,2)
    normalized_patches = np.array(list(map(normalize_patch_spectra, patches)))
    pad_patches = U.pad_time_series_by_zeros(patches, C.TIME_SERIES_LENGTH)

    segmentation_mask_path = patches_metadata.loc[path]['Semantic_segmentation_path']
    segmentation_mask = U.upload_npy_image(segmentation_mask_path)[0]
    one_hot_segmentation_mask = U.one_hot(segmentation_mask)
    
    class_weights = tf.constant(ClassWeights().class_weights, tf.float32)
    sample_weights = tf.gather(class_weights, indices=tf.cast(segmentation_mask, tf.int32), name='cast_sample_weights')
    
    return pad_patches, one_hot_segmentation_mask, sample_weights

def get_dataset(paths_to_patches):
    # print("PA", paths_to_patches)
    files = tf.data.Dataset.list_files(paths_to_patches)
    dataset = files.map(lambda x: tf.py_function(path_to_timeseries_input, [x], [tf.float32, tf.float32, tf.float32]), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

