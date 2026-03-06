import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from backend.src import config

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


def get_heatmap_for_minute(raw_signal, heatmap):
    """
    Resize heatmap to raw signal length and crop to the target (central) minute.
    Returns a 1D numpy array of length (60 * config.FS) for use in frontend overlays.

    Args:
        raw_signal: Full raw ECG segment for the window (e.g. 5 minutes).
        heatmap: 1D heatmap from make_gradcam_heatmap (model sequence length).

    Returns:
        heatmap_minute: 1D array of heatmap values for the central minute only.
    """
    heatmap = np.uint8(255 * heatmap)
    heatmap = np.expand_dims(heatmap, axis=0)
    heatmap = cv2.resize(heatmap, (raw_signal.shape[0], 1))
    heatmap = np.squeeze(heatmap)

    fs = config.FS
    start_sec = config.BEFORE * 60
    end_sec = (config.BEFORE + 1) * 60
    start_idx = int(start_sec * fs)
    end_idx = int(end_sec * fs)
    start_idx = max(0, start_idx)
    end_idx = min(len(raw_signal), end_idx)

    heatmap_minute = heatmap[start_idx:end_idx]
    # Normalize to [0, 1] for frontend color mapping
    if heatmap_minute.max() > heatmap_minute.min():
        heatmap_minute = (heatmap_minute.astype(np.float64) - heatmap_minute.min()) / (
            heatmap_minute.max() - heatmap_minute.min()
        )
    else:
        heatmap_minute = np.zeros_like(heatmap_minute, dtype=np.float64)
    return heatmap_minute


def save_gradcam_visualization(raw_signal, heatmap, save_path, alpha=0.4):
    """
    Overlays the Grad-CAM heatmap on the RAW ECG signal (cropped to the target minute)
    and saves the plot as a grid of 30 subplots (2 seconds each).
    
    Args:
        raw_signal: The raw ECG signal array (1D) for the full window.
        heatmap: The 1D heatmap (activations) derived from the processed input.
        save_path: Path to save the resulting image.
        alpha: Transparency of the heatmap overlay.
    """
    # 1. Resize heatmap to match the FULL raw signal length first
    heatmap = np.uint8(255 * heatmap)
    heatmap = np.expand_dims(heatmap, axis=0)
    heatmap = cv2.resize(heatmap, (raw_signal.shape[0], 1))
    heatmap = np.squeeze(heatmap)

    # 2. Crop to the target minute (the central minute)
    # The window structure is determined by config.BEFORE and config.AFTER.
    fs = config.FS
    start_sec = config.BEFORE * 60
    end_sec = (config.BEFORE + 1) * 60
    
    start_idx = int(start_sec * fs)
    end_idx = int(end_sec * fs)
    
    # Safety check
    start_idx = max(0, start_idx)
    end_idx = min(len(raw_signal), end_idx)
    
    signal_minute = raw_signal[start_idx:end_idx]
    heatmap_minute = heatmap[start_idx:end_idx]
    
    # 3. Create Grid Plot (30 subplots: 6 rows x 5 columns)
    # Each subplot covers 60s / 30 = 2 seconds.
    rows = 6
    cols = 5
    num_plots = rows * cols
    seconds_per_plot = 2.0
    samples_per_plot = int(seconds_per_plot * fs)
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15), sharey=True)
    axes = axes.flatten() # Flatten to 1D array for easy iteration
    
    # Global title
    fig.suptitle('Grad-CAM: Target Minute Zoomed (2s segments)', fontsize=16)

    for i in range(num_plots):
        ax = axes[i]
        
        # Segment indices relative to the minute crop
        seg_start = i * samples_per_plot
        seg_end = min((i + 1) * samples_per_plot, len(signal_minute))
        
        if seg_start >= len(signal_minute):
            ax.axis('off') # Hide unused subplots if any
            continue
            
        segment_signal = signal_minute[seg_start:seg_end]
        segment_heatmap = heatmap_minute[seg_start:seg_end]
        
        # Time axis for this segment (absolute time within the minute)
        t_start = i * seconds_per_plot
        t_axis = np.linspace(t_start, t_start + seconds_per_plot, len(segment_signal))
        
        # Plot raw signal
        ax.plot(t_axis, segment_signal, color='black', alpha=0.8, linewidth=1)
        
        # Scatter heatmap
        # Increased dot size slightly for visibility on smaller plots
        ax.scatter(t_axis, segment_signal, c=segment_heatmap, cmap='jet', alpha=alpha, s=20)
        
        # Formatting
        ax.set_title(f"{t_start:.0f}s - {t_start+seconds_per_plot:.0f}s", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Only show x labels on bottom row
        if i >= num_plots - cols:
            ax.set_xlabel("Time (s)", fontsize=9)
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    plt.savefig(save_path, dpi=150)
    plt.close()