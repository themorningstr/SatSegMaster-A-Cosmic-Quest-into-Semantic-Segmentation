import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import ListedColormap
import matplotlib
import cv2
from metadata import *
from config import *
import tensorflow as tf
from utils import *

U = Utility()

C = Constant()

patches_metadata = Metadata().metadata

cm = matplotlib.cm.get_cmap('tab20')
def_colors = cm.colors
cus_colors = ['k'] + [def_colors[i] for i in range(1,20)]+['w']
semantic_cmap = ListedColormap(colors = cus_colors, name='agri',N=21)


def visualize_single_sample(dataframe , N = 5):
	"""
	This function require a dataframe created from metadata geojson.
	"""

	patches_list = tf.data.Dataset.list_files(f'{C.DATASET_DIR}DATA_S2/*', shuffle=True)
	
	matrix, path = get_image_and_display_dataset_object_info(patches_list)
	figName = path.split("/")[-1].split(".")[0]


	f,axs = plt.subplots(1, 3, figsize=(10, 10))
	axs[0].imshow(U.get_rgb(matrix, t_show=1))
	axs[0].set_title(f"{metadata[f'{1}_date']}")
	axs[0].savefig(f"{metadata[f'{1}_date']}.png")
	axs[1].imshow(U.get_rgb(matrix, t_show=3))
	axs[1].set_title(f"{metadata[f'{3}_date']}")
	axs[1].savefig(f"{metadata[f'{3}_date']}.png")
	axs[2].imshow(U.get_rgb(matrix, t_show=4))
	axs[2].set_title(f"{metadata[f'{4}_date']}")
	axs[2].savefig(f"{metadata[f'{4}_date']}.png")

	
	f,axs = plt.subplots(10, N, figsize=(15, 25))
	for day in range(N):
	    for spectral in range(10):
	        ax = axs[spectral % 10, day]
	        ax.imshow(matrix[day][spectral])

	columns = [f"{metadata[f'{i}_date']}" for i in range(N)]
	for ax, col in zip(axs[0], columns):
	    ax.set_title(col)

	f.savefig(os.path.join(C.SAVE_VISUILATION_PATH, f"{figName}.png"))






def visualize_image_annotation_sample():

	parcel_annotations = tf.data.Dataset.list_files(f'{DATASET_DIR}ANNOTATIONS/ParcelIDs_*', shuffle=False)
	print(f'There are {len(parcel_annotations)} parcel ids annotations.')


	parcel_annotation, _ = U.get_image_and_display_dataset_object_info(parcel_annotations)


	target_annotations = tf.data.Dataset.list_files(f'{DATASET_DIR}ANNOTATIONS/TARGET_*', shuffle=False)
	print(f'There are {len(target_annotations)} target annotations.')

	target_annotation,path = U.get_image_and_display_dataset_object_info(target_annotations)

	semantic_target = target_annotation[0].astype(int)
	print(f'Semantic target value range: [{np.amin(semantic_target)}, {np.amax(semantic_target)}]')


	figName = path.split("/")[-1].split(".")[0]

	f,axs = plt.subplots(1, 2, figsize=(15, 25))
	axs[0].imshow(U.get_rgb(time_series))
	axs[0].set_title('Satellite RGB images')
	axs[0].savefig("Satellite RGB images.png")


	axs[1].imshow(semantic_target, cmap=semantic_cmap, vmin=0, vmax=19)
	axs[1].set_title('Semantic labels')
	axs[1].savefig("Semantic labels.png")


	f,axs = plt.subplots(1, 3, figsize=(15, 25))
	axs[0].imshow(parcel_annotation)
	axs[0].set_title('Parcel annotation')
	axs[0].savefig("Parcel annotation.png")
	axs[1].imshow(target_annotation[1])
	axs[1].set_title('Target annotation second dimension')
	axs[0].savefig("Target annotation second dimension.png")
	axs[2].imshow(target_annotation[2])
	axs[2].set_title('Target annotation third dimension')
	axs[0].savefig("Target annotation third dimension.png")


	plt.show()



def visualize_dataset(dataset, N = 10):

	f,ax = plt.subplots(N, 2, figsize=(10, 4 * N))
	i = 0
	for patches, mask_batch, _ in dataset.take(N):
	    patches = tf.get_static_value(patches)[0].swapaxes(1,2).swapaxes(1,3)
	    ax[i, 0].imshow(get_rgb(patches))
	    ax[i, 0].set_title('image')
	    
	    mask = np.argmax(mask_batch[0], axis=2)
	    ax[i, 1].imshow(mask, cmap=semantic_cmap, vmin=0, vmax=19)
	    ax[i, 1].set_title('mask')
	    i += 1

	plt.show()



def visualize_model_loss_iou(model_epoch, train_loss, val_loss, train_iou, val_iou):

	plt.figure()
	plt.plot(model_epoch, train_loss, 'r', label='Training loss')
	plt.plot(model_epoch, val_loss, 'c', label='Validation loss')

	plt.title('Training and Validation Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss Value')
	plt.legend()
	plt.savefig(os.path.join(C.SAVE_VISUILATION_PATH, "Model.png"))




	plt.figure()
	plt.plot(model_epoch, train_iou, 'm', label='Training mean IoU')
	plt.plot(model_epoch, val_iou, 'y', label='Validation mean IoU')

	plt.title('Training and Validation Metrics')
	plt.xlabel('Epoch')
	plt.ylabel('Metric Value')
	plt.legend()
	plt.savefig(os.path.join(C.SAVE_VISUILATION_PATH, "Mean-IOU.png"))



def visualize_prediction_result(N = 5):

	f,ax = plt.subplots(N, 3, figsize=(10, 4 * N))
	i = 0
	for patches, mask_batch, _ in test_dataset.take(N):
	    rgb_patches = tf.get_static_value(patches)[0].swapaxes(1,2).swapaxes(1,3)
	    ax[i, 0].imshow(get_rgb(rgb_patches))
	    ax[i, 0].set_title('image')
	    
	    mask = np.argmax(mask_batch[0], axis=2)
	    ax[i, 1].imshow(mask, cmap=semantic_cmap, vmin=0, vmax=19)
	    ax[i, 1].set_title('true mask')

	    pred_mask = model.predict(patches)
	    pred_mask = np.argmax(pred_mask[0], axis=2)
	    ax[i, 2].imshow(pred_mask, cmap=semantic_cmap, vmin=0, vmax=19)
	    ax[i, 2].set_title('predicted mask')
	    i += 1

	f.savefig(os.path.join(C.SAVE_VISUILATION_PATH, "model_prediction.png"))

















