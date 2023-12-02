
import math
import numpy as np
from config import Constant
import logging
import os


C = Constant()


class Utility:
    

    def upload_npy_image(self, path: str):
        
        return np.load(path)

    def reshape_patch_spectra(self, patch):
        """Utility function to reshape patch shape from k*128*128 to 128*128*k.
        """
        reshaped_image = patch.swapaxes(0,2).swapaxes(0,1)
        return reshaped_image
    
    def get_rgb(self, time_series, t_show=-1):
        """Utility function to get a displayable rgb image 
        from a Sentinel-2 time series.
        """
        image = time_series[t_show, [2,1,0]]

        # Normalize image
        max_value = image.max(axis=(1,2))
        min_value = image.min(axis=(1,2))
        image_normalized = (image - min_value[:,None,None])/(max_value - min_value)[:,None,None]

        rgb_image = self.reshape_patch_spectra(image_normalized)
        return rgb_image

    def one_hot(self, a):
        return np.squeeze(np.eye(C.NUM_CLASSES)[a])
    
    def pad_time_series(self, time_series, size):
        """Pad the input time series with repeated values without violating the temporal feature of the series.
        The output time series will be of a given length and will contain some elements repeated k times and the rest k+1.
        Example: pad_time_series([1, 2, 3, 4], 10) => [1 1 1 2 2 2 3 3 4 4].
        """
        input_size = time_series.shape[0]
        diff = size - input_size
        if diff < 0:
            raise ValueError("Time series length exceeds expected result length")
        elif diff == 0:
            return time_series
        
        duplicate_times = math.ceil(size / input_size)
        repeat_times = [duplicate_times] * (size - input_size * (duplicate_times - 1)) + [duplicate_times - 1] * (input_size * duplicate_times - size)
        repeat_times[:(size - sum(repeat_times))] = [v+1 for v in repeat_times[:(size - sum(repeat_times))]]
        
        pad_result = np.repeat(time_series, repeat_times, axis=0)
        return pad_result

    def pad_time_series_by_zeros(self, time_series, size):
        diff = size - time_series.shape[0]
        if diff < 0:
            raise ValueError("Time series length exceeds expected result length")
        elif diff == 0:
            return time_series
        
        pads = np.zeros(shape=(diff,) + time_series.shape[1:])
        pad_result = np.concatenate([pads, time_series], axis=0)
        return pad_result

    def logger(self):

        logging.basicConfig(level=logging.INFO, filename = os.path.join(os.getcwd(), "unet_lstm.log"), filemode="a", format = "%(asctime)s %(levelname)s %(message)s")

        return logging.getLogger()

    def get_image_and_display_dataset_object_info(self, df_list):

        tensor = iter(df_list).get_next()
        path = tf.get_static_value(tensor).decode("utf-8")
        print(f'Path to the first dataset object: {path}')
        image = self.upload_npy_image(path)
        print(f'The shape of the array: {image.shape}')
        print(f'Array value range: [{np.amin(image)}, {np.amax(image)}]')
        return image, path
