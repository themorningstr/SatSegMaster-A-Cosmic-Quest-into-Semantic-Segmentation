import tensorflow as tf

from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras import mixed_precision
import keras.backend as K
from config import Constant
# from utils import *


# logger = Utility.logger()

C = Constant()


class UNetConvLSTMModel:
    def __init__(self, num_classes):
        self._model = self._build_model(num_classes)

    @property
    def model(self) -> tf.keras.Model:
        return self._model
    
    def _build_model(self, num_classes, input_shape=(C.TIME_SERIES_LENGTH, 128, 128, 10)) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # apply Encoder for each time series image
        encoder = self._encoder()
        encoder_outputs = [encoder(inputs[:, i]) for i in range(61)]
        encoder_outputs = list(zip(*encoder_outputs))
        encoder_output = tf.stack(encoder_outputs[-1], axis=1)
        # logger.debug(f'Encoder output tensor: {encoder_output}')
        
        # preparing skip connections to decoder
        skip_outputs = [tf.keras.layers.Concatenate()(encoder_output) for encoder_output in encoder_outputs[:-1]]
        filters_number = [encoder_output[0].shape[-1] for encoder_output in encoder_outputs[:-1]]
        skip_connections = []
        for filters, skip_output in zip(filters_number, skip_outputs):
            skip_connection = self._conv_blocks(filters=filters, size=1, apply_instance_norm=True)(skip_output)
            skip_connections.append(skip_connection)
        # logger.debug(f'Skip connections tensors: {skip_connections}')
        
        # ConvLSTM layer
        clstm_outputs = tf.keras.layers.ConvLSTM2D(filters=64, kernel_size=3,
                                                   strides=(1, 1), padding='same',
                                                   data_format='channels_last',
                                                   return_state=True)(encoder_output)
        clstm_output = clstm_outputs[-1]
        # logger.debug(f'ConvLSTM output tensor: {clstm_output}')
        
        # apply Decoder
        decoder_output = self._decoder(skip_connections, clstm_output)
        # logger.debug(f'Decoder output tensor: {decoder_output}')

        # This is the last layers of the model
        output = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='softmax')(decoder_output)
        # logger.debug(f'Model output tensor: {output}')

        return tf.keras.Model(inputs=inputs, outputs=output)
    
    def _encoder(self, input_shape=(128, 128, 10)):
        inputs = tf.keras.layers.Input(shape=input_shape)
        outputs = []

        model = tf.keras.Sequential()
        x = model(inputs)

        for filters in [16, 32, 32, 64]:
            x = self._conv_blocks(filters=filters, size=3, apply_instance_norm=True)(x)
            x = self._conv_blocks(filters=filters, size=3, apply_instance_norm=True)(x)
            outputs.append(x)
            x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

        output = self._conv_blocks(filters=98, size=3, apply_batch_norm=True, apply_dropout=True)(x)
        outputs.append(output)

        # Create the feature extraction model
        encoder = tf.keras.Model(inputs=inputs, outputs=outputs, name="encoder")
        encoder.trainable = True
        return encoder
    
    def _decoder(self, skip_connections, output):     
        x = output
        for filters, skip, apply_dropout in zip([64, 32, 32, 20], skip_connections[::-1], [True, True, False, False]):
            x = self._upsample_block(filters, 3)(x)
            x = tf.keras.layers.Concatenate()([x, skip])
            x = self._conv_blocks(filters, size=3, apply_batch_norm=True, apply_dropout=apply_dropout)(x)
            x = self._conv_blocks(filters, size=3, apply_batch_norm=True)(x)
        return x
    
    def _conv_blocks(self, filters, size, apply_batch_norm=False, apply_instance_norm=False, apply_dropout=False):
        """Downsamples an input. Conv2D => Batchnorm => Dropout => LeakyReLU
            :param:
                filters: number of filters
                size: filter size
                apply_dropout: If True, adds the dropout layer
            :return: Downsample Sequential Model
        """
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
          tf.keras.layers.Conv2D(filters, size, strides=1,
                                 padding='same', use_bias=False,
                                 kernel_initializer=initializer,))
        if apply_batch_norm:
            result.add(tf.keras.layers.BatchNormalization())
        if apply_instance_norm:
            result.add(tfa.layers.InstanceNormalization())
        result.add(tf.keras.layers.Activation(tfa.activations.mish))
        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.55))
        return result
    
    def _upsample_block(self, filters, size, apply_dropout=False):
        """Upsamples an input. Conv2DTranspose => Batchnorm => Dropout => LeakyReLU
            :param:
                filters: number of filters
                size: filter size
                apply_dropout: If True, adds the dropout layer
            :return: Upsample Sequential Model
        """
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
          tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                          padding='same',
                                          kernel_initializer=initializer,
                                          use_bias=False))
        result.add(tf.keras.layers.BatchNormalization())
        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.1))
        result.add(tf.keras.layers.Activation(tfa.activations.mish))
        return result