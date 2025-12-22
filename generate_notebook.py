import json
import os

# Notebook structure
notebook = {
 "cells": [],
 "metadata": {
  "kernelspec": {
   "display_name": "Julia 1.9.3",
   "language": "julia",
   "name": "julia-1.9"
  },
  "language_info": {
   "file_extension": ".jl",
   "mimetype": "application/julia",
   "name": "julia",
   "version": "1.9.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

def add_markdown(source):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True)
    })

def add_code(source):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True)
    })

# --- Content Definitions ---

# 1. Introduction
add_markdown("""# Theme 3: Inspecting Neural Network Functionality (Julia/Flux)

This notebook implements the exercises for Theme 3 using the Julia programming language and the Flux.jl machine learning library. We will inspect the functionality of a neural network in image classification using the CIFAR-10 dataset.

## Setup
We need the following packages: `Flux`, `Images`, `FileIO`, `Statistics`, `Plots`, `LinearAlgebra`, `Random`.
""")

add_code("""using Pkg
Pkg.add(["Flux", "Images", "FileIO", "Plots", "OneHotArrays", "FileIO", "ImageIO", "ImageMagick"])
# Note: ImageMagick/ImageIO might be needed for PNG loading depending on OS
""")

add_code("""using Flux
using Flux: onehotbatch, onecold, crossentropy, train!, @epochs, params
using Images
using FileIO
using Statistics
using Plots
using LinearAlgebra
using Random
""")

# 2. Data Loading
add_markdown("""## 1. Data Preparation

We define helper functions to load the CIFAR-10 small dataset.
**Note**: Flux typically expects image data in the format `(Width, Height, Channels, BatchSize)`. We ensure our data loader adheres to this.
""")

add_code("""function load_labels_jl(filename)
    if !isfile(filename)
        println("Warning: Label file $filename not found.")
        return Int[]
    end
    lines = readlines(filename)
    return parse.(Int, lines)
end

function load_image_batch(folder, count, img_size=(32,32))
    # Initialize array: (Width, Height, Channels, BatchSize)
    data = zeros(Float32, img_size[1], img_size[2], 3, count)
    
    for i in 0:(count-1)
        fname = joinpath(folder, "image_" * lpad(i, 4, "0") * ".png")
        if !isfile(fname)
            continue
        end
        img = load(fname)
        # Permute from (H, W) standard to (W, H) if needed, or just ensure channel dim is last in intermediate 
        # Flux Images are typically (W, H, C, N). 
        # Image loading gives (C, H, W) often.
        mat = channelview(img) 
        mat = permutedims(mat, (3, 2, 1)) # -> (W, H, C)
        
        # Scaling: (x - 128) / 128 (performed later in batch, or here raw 0-255)
        # Load gives 0..1. We want 0..255 for the formula equivalent.
        data[:, :, :, i+1] = float.(mat) .* 255
    end
    return data
end

function normalize_dataset(data)
    return (data .- 128.0f0) ./ 128.0f0
end

function get_data()
    base_res = "resources"
    train_labels_path = joinpath(base_res, "training", "labels.csv")
    train_img_path = joinpath(base_res, "training")
    test_labels_path = joinpath(base_res, "testing", "labels.csv")
    test_img_path = joinpath(base_res, "testing")

    y_train_raw = load_labels_jl(train_labels_path)
    # Only load what we have
    if isempty(y_train_raw)
        println("Error: No training labels found. Check path.")
        return nothing
    end
    
    x_train_raw = load_image_batch(train_img_path, length(y_train_raw))
    x_train = normalize_dataset(x_train_raw)
    
    y_test_full = load_labels_jl(test_labels_path)
    x_test_full_raw = load_image_batch(test_img_path, length(y_test_full))
    x_test_norm = normalize_dataset(x_test_full_raw)
    
    # Validation Split
    splitpoint = 2000
    x_val = x_test_norm[:, :, :, 1:splitpoint]
    y_val_raw = y_test_full[1:splitpoint]
    
    x_test = x_test_norm[:, :, :, splitpoint+1:end]
    y_test_raw = y_test_full[splitpoint+1:end]
    
    # One Hot Encoding
    classes = sort(unique(y_train_raw))
    y_train = onehotbatch(y_train_raw, classes)
    y_val = onehotbatch(y_val_raw, classes)
    y_test = onehotbatch(y_test_raw, classes)
    
    return x_train, y_train, x_val, y_val, x_test, y_test, length(classes)
end

# Load the data
print("Loading data... ")
data = get_data()
if data !== nothing
    x_train, y_train, x_val, y_val, x_test, y_test, class_count = data
    println("Done.")
    println("Train shape: ", size(x_train))
    println("Val shape: ", size(x_val))
    println("Test shape: ", size(x_test))
end
""")

# 3. Task 3
add_markdown("""## 2. Task 3: Linear Model (5 Epochs)

We define a simple linear model: `Input -> Flatten -> Dense -> Softmax`.
**Question**: How many parameters?
**Answer**: $3072 \times 10 + 10 = 30,730$.
""")

add_code("""input_dim = 32 * 32 * 3

model_t3 = Chain(
    Flux.flatten,
    Dense(input_dim, class_count, softmax)
)

loss(x, y) = crossentropy(model_t3(x), y)
opt = Adam(3e-5)
data_loader = Flux.DataLoader((x_train, y_train), batchsize=32, shuffle=true)

function evaluate_acc(m, x, y)
    y_hat = m(x)
    return mean(onecold(y_hat) .== onecold(y))
end

println("Training Task 3 (5 epochs)...")
for epoch in 1:5
    Flux.train!(loss, params(model_t3), data_loader, opt)
    val_acc = evaluate_acc(model_t3, x_val, y_val)
    println("Epoch $epoch: Val Acc = $val_acc")
end

test_acc_t3 = evaluate_acc(model_t3, x_test, y_test)
println("Task 3 Test Accuracy: $test_acc_t3")
""")

# 4. Task 4
add_markdown("""## 3. Task 4: Extended Training (100 Epochs)

We train the same model for 100 epochs to observe overfitting behavior.
""")

add_code("""model_t4 = Chain(
    Flux.flatten,
    Dense(input_dim, class_count, softmax)
)
loss4(x, y) = crossentropy(model_t4(x), y)
opt4 = Adam(3e-5)

history_train = Float64[]
history_val = Float64[]

println("Training Task 4 (100 epochs)...")
for epoch in 1:100
    Flux.train!(loss4, params(model_t4), data_loader, opt4)
    push!(history_train, evaluate_acc(model_t4, x_train, y_train))
    push!(history_val, evaluate_acc(model_t4, x_val, y_val))
    if epoch % 10 == 0
        println("Epoch $epoch: Val Acc = $(history_val[end])")
    end
end

p4 = plot(history_train, label="Train Acc", title="Task 4: Accuracy vs Epochs", xlabel="Epoch", ylabel="Accuracy", linewidth=2)
plot!(p4, history_val, label="Val Acc", linewidth=2)
display(p4)
""")

add_markdown("""**Observation**: The training accuracy should continue to rise while validation accuracy plateaus, indicating overfitting. The model is memorizing the noise in the training set.""")

# 5. Task 5
add_markdown("""## 4. Task 5: Weight Visualization

We visualize the learned weights of the Dense layer. The weights, when reshaping back to image dimensions, often look like "templates" of the classes they represent.
""")

add_code("""function visualize_weights(model, layer_idx, title)
    W = model[layer_idx].weight
    n_classes = size(W, 1)
    
    plots = []
    for i in 1:n_classes
        w_vec = W[i, :]
        w_img = reshape(w_vec, 32, 32, 3)
        
        # Normalize 0-1
        w_min, w_max = extrema(w_img)
        w_norm = (w_img .- w_min) ./ (w_max - w_min + 1e-5)
        
        # Permute to (H, W, C) for plotting (Plots.jl expects H,W for images sometimes, or just standard RGB array)
        # Image structure in Julia is typically handled by `colorview(RGB, ...)` expecting (Channels, Height, Width) usually?
        # Let's try (3, 32, 32)
        img_c = permutedims(w_norm, (3, 2, 1))
        p = plot(colorview(RGB, img_c), axis=false, title="Class $(i-1)", ticks=false)
        push!(plots, p)
    end
    display(plot(plots..., layout=(2, 5), size=(800, 300), title=title))
end

visualize_weights(model_t4, 2, "Task 5: Weights")
""")

# 6. Task 6
add_markdown("""## 5. Task 6: L2 Regularization

We add L2 regularization to the loss function to penalize large weights.
`Loss = CrossEntropy + lambda * sum(weights^2)`
""")

add_code("""model_l2 = Chain(
    Flux.flatten,
    Dense(input_dim, class_count, softmax)
)

l2_lambda = 0.03
function loss_l2(x, y)
    weights = model_l2[2].weight
    return crossentropy(model_l2(x), y) + l2_lambda * sum(abs2, weights)
end

opt_l2 = Adam(3e-5)

println("Training Task 6 (L2 Regularization)...")
for epoch in 1:100
    Flux.train!(loss_l2, params(model_l2), data_loader, opt_l2)
end

test_acc_l2 = evaluate_acc(model_l2, x_test, y_test)
println("Task 6 Test Accuracy: $test_acc_l2")

visualize_weights(model_l2, 2, "Task 6: L2 Weights")
""")

add_markdown("""**Observation**: The weights should look smoother and less noisy compared to Task 5 ("ghostly templates"). The accuracy should generalize better than the unregularized model.""")

# 7. Task 7
add_markdown("""## 6. Task 7: 2-Layer Network

We introduce a hidden layer with 100 neurons and ReLU activation.
""")

add_code("""dense_size = 100
model_2l = Chain(
    Flux.flatten,
    Dense(input_dim, dense_size, relu),
    Dense(dense_size, class_count, softmax)
)

loss_2l(x, y) = crossentropy(model_2l(x), y)
opt_2l = Adam(3e-5)

println("Training Task 7 (2 Layers)...")
for epoch in 1:100
    Flux.train!(loss_2l, params(model_2l), data_loader, opt_2l)
end

test_acc_2l = evaluate_acc(model_2l, x_test, y_test)
println("Task 7 Test Accuracy: $test_acc_2l")
""")

add_markdown("""**Observation**: The addition of a non-linear layer (`ReLU`) usually correlates with a significant jump in accuracy (e.g., from ~35-40% to ~45-50%).""")

# 8. Activations
add_markdown("""## 7. Task 8: Activation Functions Demo

Comparing ReLU, Sigmoid, and LeakyReLU on a sample matrix.
""")

add_code("""input_mat = Float32[-1 0 1 2; -2 0 1 2; 10 15 20 30; -20 -10 -5 -1]

println("Input Matrix:")
display(input_mat)

println("\\nReLU:")
display(relu.(input_mat))

println("\\nSigmoid:")
display(sigmoid.(input_mat))

println("\\nLeakyReLU (0.1):")
display(leakyrelu.(input_mat, 0.1))
""")

# output
with open('theme3_julia.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
