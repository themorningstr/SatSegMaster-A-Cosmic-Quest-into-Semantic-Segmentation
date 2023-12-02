
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




def evaluation():

	patches_metadata = Metadata().metadata

	test_dataset = get_dataset(patches_metadata[patches_metadata['Fold'] == 5].index)


	test_dataset = test_dataset.batch(TEST_BATCH_SIZE)

	model = get_compiled_model()

	model.load_weights(C.CHECKPOINT_PATH)


	loss , iou = model.evaluate(test_dataset)

	TEST_LENGTH = len(patches_metadata[patches_metadata['Fold'] == 5].index)


	mIoU = IoU(num_classes=C.NUM_CLASSES, target_class_ids=list(range(0, C.NUM_CLASSES)), sparse_y_true=False, sparse_y_pred=False, name='mean-IoU')


	for patches, true_mask, _ in test_dataset.take(TEST_LENGTH):
	    pred_mask = model.predict(patches)
	    mIoU.update_state(true_mask, pred_mask)

		mIoU_classes = [IoU(num_classes=NUM_CLASSES, target_class_ids=[c], sparse_y_true=False, sparse_y_pred=False) for c in range(NUM_CLASSES)]
	    for c in range(NUM_CLASSES):
	        mIoU_classes[c].update_state(true_mask, pred_mask)


	print(mIoU.result())

	IoU_results_classes = [mIoU_class.result() for mIoU_class in mIoU_classes]

	lables = list(range(C.NUM_CLASSES))









