import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates a Grad-CAM heatmap for a given input array (1D signal).

    Args:
        img_array: Input tensor of shape (1, seq_length, num_features).
        model: The Keras model.
        last_conv_layer_name: Name of the last convolutional layer.
        pred_index: Index of the class to visualize. If None, uses the predicted class.

    Returns:
        heatmap: The 1D heatmap of shape (seq_length,).
    """
    # 1. Create a model that maps the input image to the activations
    #    of the last conv layer as well as the output predictions
    try:
        target_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        # If the specific layer name is not found, try to find the last Conv1D layer
        conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv1D)]
        if not conv_layers:
             raise ValueError("No Conv1D layers found in the model.")
        target_layer = conv_layers[-1]
        print(f"Layer '{last_conv_layer_name}' not found. Using last Conv1D layer: '{target_layer.name}'")

    grad_model = tf.keras.models.Model(
        [model.inputs], [target_layer.output, model.output]
    )

    # 2. Compute the gradient of the top predicted class for our input image
    #    with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # 3. Vector of weights: mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    # 4. Multiply each channel in the feature map array
    #    by "how important this channel is" with regard to the top predicted class
    #    then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 5. Apply ReLU to the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_gradcam_visualization(img_array, heatmap, save_path, alpha=0.4):
    """
    Overlays the Grad-CAM heatmap on the 1D signal and saves the plot.

    Args:
        img_array: Input tensor of shape (1, seq_length, num_features).
        heatmap: The 1D heatmap (activations).
        save_path: Path to save the resulting image.
        alpha: Transparency of the heatmap overlay.
    """
    # img_array shape is (1, 900, 2) -> we take the first feature (RRI) for plotting usually
    # or plot both. Let's assume RRI is index 0.
    signal = img_array[0, :, 0] # RRI
    
    # Resize heatmap to match signal length
    # We use cv2.resize for convenience, treating it as a 1D image (height 1)
    # heatmap is currently shape (14,) or similar.
    # We need to resize it to (900,)
    heatmap = np.uint8(255 * heatmap)
    
    # Expand dims to use cv2.resize which expects at least 2D
    heatmap = np.expand_dims(heatmap, axis=0) # (1, 14)
    heatmap = cv2.resize(heatmap, (signal.shape[0], 1)) # (900, 1)
    heatmap = np.squeeze(heatmap) # (900,)

    # Create figure
    plt.figure(figsize=(10, 4))
    
    # Plot original signal
    x = np.arange(len(signal))
    plt.plot(x, signal, label='ECG Signal (RRI)', color='black', alpha=0.8, linewidth=1)

    # Color mapping for heatmap
    # We can scatter plot points colored by heatmap intensity, or use fill_between
    # A simple way is to use a scatter plot with c=heatmap
    plt.scatter(x, signal, c=heatmap, cmap='jet', alpha=alpha, label='Grad-CAM', s=10)
    
    plt.colorbar(label='Attention')
    plt.title('Grad-CAM: Model Attention on ECG Signal')
    plt.xlabel('Time Step')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
