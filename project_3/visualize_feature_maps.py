""" Function to visualize conv layer feature maps """

import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf


from tensorflow.keras import  models # my import

from tensorflow.keras.preprocessing.image import load_img, img_to_array


data_dir = 'data/CCSN_v2/'
model_dir = 'models/cnn/'
figure_dir = 'figures/cnn/'
unique_dir = 'cloudnet128_100epochs_augmentFalse_optimizerSGD_lr0.001mom0.9/'

# load pre-trained model
model = models.load_model(model_dir+unique_dir+'model.keras')

def get_image(path):
    """ func for loading images and simple preprocessing """
    image = load_img(path, grayscale=False, color_mode='rgb', target_size=(128, 128))
    image=img_to_array(image)
    image=image/255.0
    image = np.expand_dims(image, axis=0)
    return image

image_paths = ['data/CCSN_v2/Cs/Cs-N099.jpg', 'data/CCSN_v2/Ct/Ct-N007.jpg', 'data/CCSN_v2/Ns/Ns-N107.jpg', 'data/CCSN_v2/Cb/Cb-N050.jpg']
all_images = [get_image(path) for path in image_paths]

# layers_to_visualize = {'conv2d': 12, 'conv2d_1': 100, 'conv2d_2': 300}#, 'conv2d_3': 230}
layers_to_visualize = {'conv2d': [12, 90], 'conv2d_1': [100]}

# getting layers and layer output instances
selected_layers = [layer for layer in model.layers if layer.name in layers_to_visualize.keys()]
selected_layer_outputs = [layer.output for layer in selected_layers]

# way of getting activation responses from pre-trained model
activation_model = tf.keras.models.Model(inputs=model.input, outputs=selected_layer_outputs)

def image_and_activations_subplot(image, activations, row,  layers_to_visualize):
    """ function for generating the subplot figures """
    # plot the original image
    plt.subplot(num_rows, num_columns, row * num_columns + 1)
    plt.imshow(np.squeeze(image))
    plt.axis('off')

    # plot feature maps
    subplot_index = 1
    for i, layer_activations in enumerate(activations):
        layer_name = selected_layers[i].name
        for channel in layers_to_visualize[layer_name]:
            feature_map = layer_activations[0, :, :, channel]

            plt.subplot(num_rows, num_columns, row * num_columns + subplot_index + 1)
            plt.imshow(feature_map, cmap='jet')
            plt.axis('off')
            subplot_index += 1

# Setup the plot
num_rows = len(all_images)
num_layers = len(layers_to_visualize)
num_columns = sum(len(channels) for channels in layers_to_visualize.values()) + 1  # +1 for the original image
plt.figure(figsize=(13, 13))
# plt.subplots_adjust(wspace=0.1, hspace=0.1)
titles = ['Original image', 'Conv1-Ch12', 'Conv1-Ch90', 'Conv2-Ch100']
row_labels = ['Cs', 'Ct', 'Ns', 'Cb']


# Iterate over the images and the layer channel activations
for idx, image in enumerate(all_images):
    activations = activation_model.predict(image)
    image_and_activations_subplot(image, activations, idx, layers_to_visualize)
    # for titles on first row
    if idx == 0:
        for i, title in enumerate(titles):
            plt.subplot(num_rows, num_columns, i + 1)
            plt.title(title, fontsize=24)
    ax = plt.subplot(num_rows, num_columns, idx * num_columns + 1)
    ax.set_ylabel(row_labels[idx], rotation=90, fontsize=12)

plt.tight_layout()
plt.savefig('CNN_layer_feature_maps.pdf', bbox_inches='tight')