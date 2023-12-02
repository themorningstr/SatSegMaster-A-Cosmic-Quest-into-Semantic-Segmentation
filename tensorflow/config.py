
import os

class Constant:

	DATASET_DIR = '/home/oem/vishwas/kaleidEO/Dataset/PASTIS/'

	NUM_CLASSES = 20
	VOID_LABEL = 19
	TIME_SERIES_LENGTH = 61
	TRAIN_BATCH_SIZE = 4
	VALIDATION_BATCH_SIZE = 1
	TEST_BATCH_SIZE = 1

	# Train Model constants
	TRAIN_LENGTH = 1455
	EPOCHS_NUMBER = 50
	STEPS_PER_EPOCH = TRAIN_LENGTH // TRAIN_BATCH_SIZE
	CHECKPOINT_PATH = "/home/oem/vishwas/kaleidEO/model_checkpoint"
	SAVE_VISUILATION_PATH = "/home/oem/vishwas/kaleidEO/model_viz"

	os.makedirs(SAVE_VISUILATION_PATH, exist_ok=True)




