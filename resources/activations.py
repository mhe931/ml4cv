import numpy as np
from tensorflow import keras

input = np.array([[-1, 0, 1, 2], [-2, 0, 1, 2], [10, 15, 20, 30], [-20, -10, -5, -1]]).astype(np.float32)

output = keras.activations.relu(input)

print(output.numpy())