# Theme 3: Inspecting Neural Network Functionality (Julia/Flux Edition)

## 1. Setup & Data Preparation (Julia)
**Objective**: Fetch and prepare the CIFAR-10 small dataset using Julia.
- **Implementation**: We used `Images.jl` and `FileIO.jl` to load images.
- **Data Shape**: Unlike Python's `(Batch, Height, Width, Channels)` format, standard Flux operations typically expect `(Width, Height, Channels, Batch)`.
    - Initial Load: `(32, 32, 3, N)`
    - This column-major layout is native to Julia.

## 2. Initial Run (Baseline)
We implemented a linear classifier using `Flux.jl`.
- **Model**: `Chain(Flux.flatten, Dense(3072, 10, softmax))`.
- **Training**: Using `Adam(3e-5)` optimizer and `crossentropy` loss.
- **Expected Outcome**: Similar to the Python version, a simple linear model achieves ~35% accuracy locally.

## 3. Code Inspection (Julia Context)
**Q: How many and what types of layers does the neural network of the program have?**
**A:** In the Flux implementation:
```julia
model = Chain(
    Flux.flatten,
    Dense(3072, 10, softmax)
)
```
- `Flux.flatten`: Reshapes the 4D input array into a 2D matrix `(3072, BatchSize)`. It does not have trainable parameters.
- `Dense`: Connectivity layer.
*Total: 1 Trainable layer.*

**Q: What is the size of the linear layer?**
**A:**
- **Inputs**: 3072.
- **Outputs**: 10.
- **Weight Matrix**: `(10, 3072)`. Note the transpose compared to Keras `(3072, 10)`. Flux stores weights as `(Out, In)`.
- **Total Parameters**: $10 \times 3072 + 10 = 30,730$.

**Q: What kind of data normalization is done?**
**A:**
```julia
(data .- 128.0f0) ./ 128.0f0
```
Standardization logic remains the same (mapping 0..255 to approx -1..1).

## 4. Training Behavior (100 Epochs)
Training for 100 epochs typically results in overfitting for this simple architecture.
- **Flux Implementation Detail**: We used a custom training loop with `Flux.train!` and explicit evaluation steps.
- **Plotting**: Used `Plots.jl` to visualize the divergence between Training and Validation accuracy.

## 5. Visualization of Weights
We visualized the weights of the Dense layer.
- **Accessing Weights**: `model[2].weight` gives the matrix.
- **Reshaping**: We reshaped rows `W[i, :]` back to `(32, 32, 3)`.
- **Interpretation**: The visualizations confirm the "template matching" nature of the single-layer perceptron.

## 6. Regularization (L2)
In Flux, L2 regularization can be added to the optimizer (WeightDecay) or explicitly to the loss function. We chose the explicit approach to match the exercise logic:
```julia
reg = l2_lambda * sum(abs2, model[2].weight)
loss = crossentropy(pred, y) + reg
```
- **Effect**: Reduces the magnitude of weights, creating "smoother" weight images and improving test accuracy by reducing overfitting.

## 7. Deepening the Network (2 Layers)
Adding a hidden layer:
```julia
model_2l = Chain(
    Flux.flatten,
    Dense(input_dim, 100, relu),
    Dense(100, 10, softmax)
)
```
- **Result**: The non-linear activation (`relu`) allows the network to capture more complex features, boosting accuracy significantly (typically >45%).

## 8. Activations
We demonstrated activations using Julia's broadcasting dot syntax:
- `relu.(x)`
- `sigmoid.(x)`
- `leakyrelu.(x, 0.1)`

**Comparison**:
- **ReLU**: Fast, avoids vanishing gradient for positive values.
- **Sigmoid**: Saturates at 0 and 1, can cause vanishing gradients in deep nets.
- **LeakyReLU**: Fixes the "dying ReLU" problem by allowing small gradients for negative inputs.
