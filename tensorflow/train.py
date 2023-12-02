
from dataset import *
from model import *
from metric import *
import numpy as np
import tensorflow as tf

from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras import mixed_precision
import keras.backend as K

from config import Constant
from metadata import *
# from utils import *


# logger = Utility.logger()

C = Constant()


def get_compiled_model():
    optimizer = tfa.optimizers.RectifiedAdam(
        learning_rate=0.005,
        total_steps=C.EPOCHS_NUMBER * C.STEPS_PER_EPOCH,
        warmup_proportion=0.3,
        min_lr=0.0001,
    )
    optimizer = tfa.optimizers.Lookahead(optimizer)

    loss = tf.keras.losses.CategoricalCrossentropy()
    mIoU = IoU(num_classes=C.NUM_CLASSES, target_class_ids=list(range(0, C.NUM_CLASSES)), sparse_y_true=False, sparse_y_pred=False, name='mean-IoU')
    model = UNetConvLSTMModel(C.NUM_CLASSES).model
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=[mIoU,],
                 )
    return model






def main():

    patches_metadata = Metadata().metadata
    patches_list = tf.data.Dataset.list_files(f'{C.DATASET_DIR}DATA_S2/*', shuffle=False)


    train_dataset = get_dataset(patches_metadata[patches_metadata['Fold'].isin([1,2,3])].index)
    validation_dataset = get_dataset(patches_metadata[patches_metadata['Fold'] == 4].index)
    # test_dataset = get_dataset(patches_metadata[patches_metadata['Fold'] == 5].index)

    train_dataset = train_dataset.repeat().batch(C.TRAIN_BATCH_SIZE)
    validation_dataset = validation_dataset.batch(C.VALIDATION_BATCH_SIZE)
    # test_dataset = test_dataset.batch(TEST_BATCH_SIZE)




    model = get_compiled_model()

    trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
    print(f'Trainable params: {trainable_params}')

    checkpoint_filepath = C.CHECKPOINT_PATH
    save_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_mean-IoU',
        mode='max',
        save_best_only=True
    )


    # try:
    model_history = model.fit(train_dataset,
                          epochs=C.EPOCHS_NUMBER,
                          steps_per_epoch=C.STEPS_PER_EPOCH,
                          validation_data=validation_dataset,
                          callbacks=[save_callback],
                          verbose=True)
    # except Exception as error:
    #     print(f'Model output: {error}')



    model.load_weights(checkpoint_filepath)

    train_loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    train_mIoU = model_history.history['mean-IoU']
    val_mIoU = model_history.history['val_mean-IoU']


    visualize_model_loss_iou(model_epoch = model_history.epoch, 
        train_loss = train_loss, 
        val_loss = val_loss, 
        train_iou = train_mIoU, 
        val_iou = val_mIoU)




if __name__ == "__main__":

    main()
