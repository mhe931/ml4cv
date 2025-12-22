from matplotlib import pyplot as plt
weights, biases = model.get_layer('dense_layer').get_weights()
height, width, channels = 32, 32, 3
input_shape = (height, width, channels)

fig, axes = plt.subplots(5, 2, figsize=(8, 8))

for i, ax in enumerate(axes.flat):
  if i >= weights.shape[1]:
    break

  weight_vector = weights[:, i]
  weight_image = weight_vector.reshape(input_shape)

  min_val = weight_image.min()
  max_val = weight_image.max()
  weight_image = (weight_image - min_val) / (max_val - min_val + 1e-5) # Avoid division by 0

  ax.imshow(weight_image)
  ax.set_title(f'Class {i}')
  ax.axis('off')

plt.suptitle('MLP Dense Layer Weights as RGB Images', fontsize=16)
plt.tight_layout()
plt.show()