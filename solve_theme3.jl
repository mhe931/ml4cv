using Flux
using Flux: onehotbatch, onecold, crossentropy, train!, @epochs, params
using Images
using FileIO
using Statistics
using Plots
using LinearAlgebra
using Random

# --- Helper Functions ---

function load_labels_jl(filename)
    if !isfile(filename)
        println("Warning: Label file $filename not found.")
        return Int[]
    end
    lines = readlines(filename)
    # Python code parses lines to Int.
    labels = parse.(Int, lines)
    return labels
end

function load_image_batch(folder, count, img_size=(32,32))
    # Initialize array: (Width, Height, Channels, BatchSize)
    # Flux expects (W, H, C, N)
    data = zeros(Float32, img_size[1], img_size[2], 3, count)
    
    for i in 0:(count-1)
        # Format filename to match image_0000.png
        fname = joinpath(folder, "image_" * lpad(i, 4, "0") * ".png")
        if !isfile(fname)
            continue
        end
        img = load(fname)
        # Resize if necessary, but CIFAR is 32x32
        # channelview returns (C, H, W) or (C, W, H) depending on implementation sometimes, 
        # usually (C, H, W) for standard Images.
        # We need to ensure we align with dims (W, H, C) for the loop or permute.
        # Easier: Get channelview, permute to (W, H, C)
        # In Julia Images, standard is (H, W).
        # We want to match (32, 32, 3).
        mat = channelview(img) # -> (3, 32, 32) usually
        mat = permutedims(mat, (3, 2, 1)) # -> (32, 32, 3) (Width, Height, Chan)
        
        # Scaling: Image loading usually gives 0..1. 
        # Python code assumes 0..255 input, then (x-128)/128.
        # (x_255 - 128)/128 = x_255/128 - 1 ~= 2 * x_norm - 1
        data[:, :, :, i+1] = float.(mat) .* 255
    end
    return data
end

function normalize_dataset(data)
    # Match Python logic: (x - 128) / 128
    return (data .- 128.0f0) ./ 128.0f0
end

function get_data()
    base_res = "resources"
    
    # Paths
    train_labels_path = joinpath(base_res, "training", "labels.csv")
    train_img_path = joinpath(base_res, "training")
    test_labels_path = joinpath(base_res, "testing", "labels.csv")
    test_img_path = joinpath(base_res, "testing")

    # Load
    y_train_raw = load_labels_jl(train_labels_path)
    x_train_raw = load_image_batch(train_img_path, length(y_train_raw))
    
    y_test_full = load_labels_jl(test_labels_path)
    x_test_full = load_image_batch(test_img_path, length(y_test_full))
    
    # Normalize
    x_train = normalize_dataset(x_train_raw)
    x_test_norm = normalize_dataset(x_test_full)
    
    # Split Test/Val (First 2000 is Val in Python script logic? 
    # Python: data[splitpoint:] -> 2000 to end (Test)
    # Python: data[:splitpoint] -> 0 to 2000 (Val)
    # Let's match:
    # x_val = x_test_norm[0:2000] (Julia 1:2000)
    # x_test = x_test_norm[2000:] (Julia 2001:end)
    
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

function evaluate_acc(model, x, y_onehot)
    y_pred = model(x)
    pred_indices = onecold(y_pred)
    true_indices = onecold(y_onehot)
    return mean(pred_indices .== true_indices)
end

function visualize_weights_jl(model, layer_index, title, filename)
    # Weights are in the specific Dense layer
    # Flux Chain: layers indexable
    layer = model[layer_index] # Assuming layer_index points to the Dense layer
    W = layer.weight # Shape: (Out, In) -> (10, 3072)
    
    # We want to visualize rows of W. Each row corresponds to a class.
    # But wait, Python code: weight_vector = weights[:, i]. 
    # Python Keras Dense weights: (Input, Output) -> (3072, 10).
    # Flux Dense weights: (Output, Input) -> (10, 3072).
    # So we want W[i, :].
    
    n_classes = size(W, 1)
    p = plot(layout=(2, 5), size=(800, 400), title=title)
    
    for i in 1:n_classes
        # Extract weights for class i
        w_vec = W[i, :]
        # Reshape to (32, 32, 3) matching (W, H, C)
        # Flux flatten was (W, H, C, N) -> (WHC, N)
        # So we can reshape back directly.
        w_img = reshape(w_vec, 32, 32, 3)
        
        # Normalize to 0..1 for display
        w_min, w_max = extrema(w_img)
        w_norm = (w_img .- w_min) ./ (w_max - w_min + 1e-5)
        
        # Permute for plotting: Plots expects (H, W) or uses coordinate system
        # Images uses (H, W). Our data is (32, 32, 3) -> (W, H, C) presumably.
        # Let's permute to (H, W, C) for colorview
        img_color = colorview(RGB, permutedims(w_norm, (3, 2, 1)))
        
        plot!(p[i], img_color, axis=false, title="Class $(i-1)")
    end
    savefig(p, filename)
    println("Saved $filename")
end

# --- Tasks ---

function main()
    println("--- Loading Data ---")
    x_train, y_train, x_val, y_val, x_test, y_test, class_count = get_data()
    
    input_dim = 32 * 32 * 3
    
    # Task 3: 1 Layer Model (Flatten -> Dense -> Softmax)
    # In Flux, Dense includes activation. 
    # Chain(Flatten, Dense)
    println("\n--- Task 3 (5 Epochs) ---")
    model_t3 = Chain(
        Flux.flatten,
        Dense(input_dim, class_count, softmax)
    )
    
    loss(x, y) = crossentropy(model_t3(x), y)
    opt = Adam(3e-5)
    data_loader = Flux.DataLoader((x_train, y_train), batchsize=32, shuffle=true)
    
    # Train 5 epochs
    for epoch in 1:5
        Flux.train!(loss, params(model_t3), data_loader, opt)
        acc = evaluate_acc(model_t3, x_val, y_val)
        println("Epoch $epoch Val Acc: $acc")
    end
    test_acc = evaluate_acc(model_t3, x_test, y_test)
    println("Task 3 Test Accuracy: $test_acc")
    
    # Task 4: 100 Epochs
    println("\n--- Task 4 (100 Epochs) ---")
    model_t4 = Chain(
        Flux.flatten,
        Dense(input_dim, class_count, softmax)
    )
    loss4(x, y) = crossentropy(model_t4(x), y)
    opt4 = Adam(3e-5)
    
    history_train = Float64[]
    history_val = Float64[]
    
    for epoch in 1:100
        Flux.train!(loss4, params(model_t4), data_loader, opt4)
        # Evaluate occasionally to save time? Or every epoch
        train_a = evaluate_acc(model_t4, x_train, y_train)
        val_a = evaluate_acc(model_t4, x_val, y_val)
        push!(history_train, train_a)
        push!(history_val, val_a)
    end
    
    # Plot history
    p4 = plot(history_train, label="Train Acc", title="Task 4: Accuracy", xlabel="Epochs", ylabel="Accuracy")
    plot!(p4, history_val, label="Val Acc")
    savefig(p4, "task4_accuracy_jl.png")
    println("Saved task4_accuracy_jl.png")
    
    # Task 5: Visualize
    println("\n--- Task 5 (Visualization) ---")
    visualize_weights_jl(model_t4, 2, "Task 5 Weights", "task5_weights_jl.png")
    
    # Task 6: L2 Regularization
    println("\n--- Task 6 (L2 Reg) ---")
    model_l2 = Chain(
        Flux.flatten,
        Dense(input_dim, class_count, softmax)
    )
    
    # Manual L2: lambda * sum(w^2)
    # Python used 0.03.
    l2_lambda = 0.03
    function loss_l2(x, y)
        # Extract weights from the Dense layer (layer 2)
        w = model_l2[2].weight
        ce = crossentropy(model_l2(x), y)
        reg = l2_lambda * sum(abs2, w)
        return ce + reg
    end
    
    opt_l2 = Adam(3e-5)
    
    hist_l2_val = Float64[]
    for epoch in 1:100
        Flux.train!(loss_l2, params(model_l2), data_loader, opt_l2)
        val_a = evaluate_acc(model_l2, x_val, y_val)
        push!(hist_l2_val, val_a)
    end
    
    visualize_weights_jl(model_l2, 2, "Task 6 L2 Weights", "task6_weights_jl.png")
    test_acc_l2 = evaluate_acc(model_l2, x_test, y_test)
    println("Task 6 Test Accuracy: $test_acc_l2")
    
    # Task 7: 2 Layers
    println("\n--- Task 7 (2 Dense Layers) ---")
    dense_size = 100
    model_2l = Chain(
        Flux.flatten,
        Dense(input_dim, dense_size, relu),
        Dense(dense_size, class_count, softmax)
    )
    
    loss_2l(x, y) = crossentropy(model_2l(x), y)
    opt_2l = Adam(3e-5)
    
    for epoch in 1:100
        Flux.train!(loss_2l, params(model_2l), data_loader, opt_2l)
    end
    test_acc_2l = evaluate_acc(model_2l, x_test, y_test)
    println("Task 7 Test Accuracy: $test_acc_2l")
    
    # Task 8: Activations
    println("\n--- Task 8 (Activations Demo) ---")
    input_mat = Float32[-1 0 1 2; -2 0 1 2; 10 15 20 30; -20 -10 -5 -1]
    
    println("ReLU:")
    display(relu.(input_mat))
    
    println("Sigmoid:")
    display(sigmoid.(input_mat))
    
    println("LeakyReLU (0.1):")
    display(leakyrelu.(input_mat, 0.1))
end

main()
