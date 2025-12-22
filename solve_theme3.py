import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Flatten
from keras.models import Model
from keras import regularizers
from PIL import Image
import matplotlib.pyplot as plt
import os

# --- Helper Functions (copied/adapted) ---
def load_labels(filename):
    with open(filename,'r') as file:
       li = file.readlines()
    label_count = len(li)
    labels = np.empty((label_count,1), dtype='int')
    i = 0
    with open(filename) as f:
        for line in f:
            labels[i] = int(line.replace("\n", ""))
            i = i + 1
    return labels

def load_images(folder, image_count, image_size):
    array_shape = (image_count, image_size[0], image_size[1], image_size[2])
    imageset = np.empty(array_shape, dtype='float')
    for i in range(0,image_count):
        fname = os.path.join(folder, 'image_' + "{:04d}".format(i) + '.png')
        if not os.path.exists(fname):
             print(f"Warning: File {fname} not found.")
             continue
        image = Image.open(fname)
        imageset[i] = np.asarray(image)
    return imageset

def normalize_dataset(sampled_images):
    sampled_images = (sampled_images.astype('float32')-128) / 128
    return sampled_images

def split_test_val(data, splitpoint):
    return data[splitpoint:], data[:splitpoint]

def create_model(input_shape, dense_size, classes, l2_reg=None):
    x = Input(shape=(input_shape))
    y = Flatten()(x)
    
    if l2_reg:
        y = Dense(classes, activation='softmax', name='dense_layer', kernel_regularizer=l2_reg)(y)
    else:
        y = Dense(classes, activation='softmax', name='dense_layer')(y)
        
    model = Model(inputs=x, outputs=y)
    return model

def create_model_2_layers(input_shape, dense_size, classes):
    x = Input(shape=(input_shape))
    y = Flatten()(x)
    # Task 7: 2nd dense layer
    y = Dense(dense_size, activation='relu')(y)
    y = Dense(classes, activation='softmax', name='dense_layer_out')(y)
    model = Model(inputs=x, outputs=y)
    return model

def visualize_weights(model, layer_name, title, filename):
    try:
        weights, biases = model.get_layer(layer_name).get_weights()
    except ValueError:
        print(f"Layer {layer_name} not found or has no weights")
        return

    # Weights shape: (InputDim, Units) -> (3072, 10)
    input_shape = (32, 32, 3)
    n_units = weights.shape[1]
    
    # We want to show the weights for each class (unit)
    rows = 2
    cols = 5 
    
    fig, axes = plt.subplots(rows, cols, figsize=(10, 5))
    
    for i, ax in enumerate(axes.flat):
        if i >= n_units:
            break

        weight_vector = weights[:, i]
        # Reshape: (3072,) -> (32, 32, 3)
        weight_image = weight_vector.reshape(input_shape)

        # Normalize to 0-1 for display
        min_val = weight_image.min()
        max_val = weight_image.max()
        weight_image = (weight_image - min_val) / (max_val - min_val + 1e-5)

        ax.imshow(weight_image)
        ax.set_title(f'Class {i}')
        ax.axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved visualization to {filename}")

# --- Main Execution ---
print("--- Loading Data ---")
base_res = 'resources' # Assumed relative to CWD
y_train = load_labels(os.path.join(base_res, 'training/labels.csv'))
x_train_raw = load_images(os.path.join(base_res, 'training'), len(y_train), (32,32,3))
Y_test = load_labels(os.path.join(base_res, 'testing/labels.csv'))
X_test_raw = load_images(os.path.join(base_res, 'testing'), len(Y_test), (32,32,3))

x_train = normalize_dataset(x_train_raw)
X_test = normalize_dataset(X_test_raw)

# Note: The original code uses 'splitpoint' on X_test, which is oddly large?
# Let's check sizes.
# In original code: 
# X_test = load_images('testing', len(Y_test)...)
# splitpoint = 2000
# x_test, x_val = split_test_val(X_test, splitpoint)
# data[splitpoint:] is the FIRST part if split_test_val returns data[splitpoint:], data[:splitpoint]?
# data[splitpoint:] -> from 2000 to end
# data[:splitpoint] -> from 0 to 2000.
# So x_test is the TAIL, x_val is the HEAD.
# Let's stick to the original code's logic.

splitpoint = 2000
x_test, x_val = split_test_val(X_test, splitpoint)
y_test, y_val = split_test_val(Y_test, splitpoint)

class_count = len(np.unique(y_train))
dims = (32, 32, 3)
dense_sz = 100

print(f"Training set: {x_train.shape}")
print(f"Validation set: {x_val.shape}")
print(f"Test set: {x_test.shape}")
print(f"Classes: {class_count}")

# --- Task 3 & Routine ---
print("\n--- Task 3 (Standard Model 5 epochs) ---")
model = create_model(dims, dense_sz, class_count)
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5), metrics=['accuracy'])
model.summary()

l1 = model.get_layer('dense_layer')
w, b = l1.get_weights()
print(f"Linear Layer Weights shape: {w.shape}")

model.fit(x_train, y_train, epochs=5, validation_data=(x_val, y_val), verbose=2)
score = model.evaluate(x_test, y_test, verbose=0)
print(f'Test accuracy (5 epochs): {score[1]}')

# --- Task 4 ---
print("\n--- Task 4 (Standard Model 100 epochs) ---")
model_t4 = create_model(dims, dense_sz, class_count)
model_t4.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5), metrics=['accuracy'])
history = model_t4.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val), verbose=0)

plt.figure()
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Task 4: Accuracy vs Epochs')
plt.legend()
plt.savefig('task4_accuracy.png')
print("Saved task4_accuracy.png")
print(f"Final Train Acc: {history.history['accuracy'][-1]:.4f}")
print(f"Final Val Acc: {history.history['val_accuracy'][-1]:.4f}")

# --- Task 5 ---
print("\n--- Task 5 (Visualization) ---")
visualize_weights(model_t4, 'dense_layer', 'Task 5: Weights', 'task5_weights.png')

# --- Task 6 ---
print("\n--- Task 6 (L2 Regularization) ---")
model_l2 = create_model(dims, dense_sz, class_count, l2_reg=regularizers.L2(0.03))
model_l2.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5), metrics=['accuracy'])
history_l2 = model_l2.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val), verbose=0)
visualize_weights(model_l2, 'dense_layer', 'Task 6: L2 Weights', 'task6_weights.png')

plt.figure()
plt.plot(history_l2.history['accuracy'], label='train_accuracy')
plt.plot(history_l2.history['val_accuracy'], label='val_accuracy')
plt.title('Task 6: L2 Accuracy')
plt.legend()
plt.savefig('task6_accuracy.png')

score_l2 = model_l2.evaluate(x_test, y_test, verbose=0)
print(f'Test accuracy (L2, 100 epochs): {score_l2[1]}')

# --- Task 7 ---
print("\n--- Task 7 (2 Layers) ---")
model_2l = create_model_2_layers(dims, dense_sz, class_count)
model_2l.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5), metrics=['accuracy'])
history_2l = model_2l.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val), verbose=0)
score_2l = model_2l.evaluate(x_test, y_test, verbose=0)
print(f'Test accuracy (2 Layers, 100 epochs): {score_2l[1]}')

# --- Task 8 (Activations) ---
print("\n--- Task 8 (Activations Demo) ---")
input_mat = np.array([[-1, 0, 1, 2], [-2, 0, 1, 2], [10, 15, 20, 30], [-20, -10, -5, -1]]).astype(np.float32)

def demo_activation(name, func):
    out = func(input_mat)
    print(f"Activation: {name}")
    print(out.numpy())

demo_activation('ReLU', keras.activations.relu)
demo_activation('Sigmoid', keras.activations.sigmoid)
l_relu = keras.layers.LeakyReLU(negative_slope=0.1)
out_lrelu = l_relu(input_mat)
print("Activation: LeakyReLU (0.1)")
print(out_lrelu.numpy())
