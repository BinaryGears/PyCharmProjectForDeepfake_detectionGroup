import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from IPython.display import Image, display
import matplotlib as mpl

# Model parameters
input_shape = (160, 160, 3)
epochs = 1
batch_size = 64
num_classes = 2

# Training data
df1 = pd.read_csv("images/train/image_labels.csv")
datagen1 = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = datagen1.flow_from_dataframe(
    dataframe=df1,
    directory="images/train/",
    x_col="filename",
    y_col="class",
    class_mode="categorical",
    target_size=(160, 160),
    batch_size=batch_size
)

# Validation data
df2 = pd.read_csv("images/val/image_labels.csv")
datagen2 = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
validation_generator = datagen2.flow_from_dataframe(
    dataframe=df2,
    directory="images/val/",
    x_col="filename",
    y_col="class",
    class_mode="categorical",
    target_size=(160, 160),
    batch_size=batch_size
)

# The custom CNN model without flattening and with compatible output shape
custom_input = keras.layers.Input(shape=input_shape)
x = keras.layers.Conv2D(8, (3, 3), activation='leaky_relu', padding='same')(custom_input)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Conv2D(16, (3, 3), activation='leaky_relu', padding='same')(x)
x = keras.layers.Conv2D(16, (3, 3), activation='leaky_relu', padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.AveragePooling2D((2, 2))(x)
x = keras.layers.Conv2D(32, (3, 3), activation='leaky_relu', padding='same')(x)
x = keras.layers.Conv2D(32, (3, 3), activation='leaky_relu', padding='same')(x)
x = keras.layers.Conv2D(32, (3, 3), activation='leaky_relu', padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.AveragePooling2D((2, 2))(x)
x = keras.layers.Conv2D(64, (3, 3), activation='leaky_relu', padding='same')(x)
x = keras.layers.Conv2D(64, (3, 3), activation='leaky_relu', padding='same')(x)
x = keras.layers.Conv2D(64, (3, 3), activation='leaky_relu', padding='same')(x)
x = keras.layers.Conv2D(64, (3, 3), activation='leaky_relu', padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.AveragePooling2D((2, 2))(x)
x = keras.layers.Conv2D(128, (3, 3), activation='leaky_relu', padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
x = keras.layers.Conv2D(256, (3, 3), activation='leaky_relu', padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPooling2D((2, 2))(x)

# Upsample to match the input shape of Xception
x = keras.layers.Conv2DTranspose(3, (3, 3), strides=(32, 32), padding='same', activation='relu')(x)

# Load Xception model
xception_base = keras.applications.Xception(
    weights='imagenet', include_top=False, input_shape=(160, 160, 3)
)
xception_base.trainable = False

# Apply Xception to the output of the custom CNN (ensure the shapes are compatible)
xception_output = xception_base(x)
x = keras.layers.GlobalAveragePooling2D()(xception_output)
x = keras.layers.Dense(256, activation='leaky_relu')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(128, activation='leaky_relu')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(0.5)(x)
output = keras.layers.Dense(num_classes, activation='softmax')(x)

# Final model combining custom CNN and Xception
combined_model = keras.Model(inputs=custom_input, outputs=output)

# Compile the combined model
combined_model.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=['accuracy']
)

# Train the combined model
combined_model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# Display the image
img_path = "C:/Users/lalsa/Downloads/deepfake-detection-main/mycode/sample_image.jpg"
display(Image(img_path))


def get_img_array(img_path, size):
    img = keras.utils.load_img(img_path, target_size=size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Find the correct layer name for Grad-CAM
for layer in combined_model.layers:
    if isinstance(layer, keras.layers.Conv2D):
        print(layer.name)
img_array = keras.applications.xception.preprocess_input(get_img_array(img_path, size=(160, 160))) # Prepare the image array

# Use the correct layer name (change 'conv2d_x' to the appropriate layer name found in the print output)
heatmap = make_gradcam_heatmap(img_array, combined_model, 'conv2d_11')  # Adjust the name here based on the print output

# Visualize the heatmap
plt.matshow(heatmap)
plt.show()

def save_and_display_gradcam(img_path, heatmap, cam_path="save_cam_image.jpg", alpha=0.4):
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)
    heatmap = np.uint8(255 * heatmap)
    jet = mpl.cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)
    display(Image(cam_path))

# Save and display Grad-CAM result
save_and_display_gradcam(img_path, heatmap)
