import os
import glob
import cv2
import numpy as np
import torch
from upscaler import UpscalerCNN

def load_model(weights_path, upscale_factor=2, num_channels=3, base_filter=64, device="cpu"):
    """
    Load the UpscalerCNN model with the specified weights.
    """
    # Instantiate the model
    model = UpscalerCNN(upscale_factor, num_channels, base_filter)
    
    # Load the model weights
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    model.to(device)  # Move the model to the specified device
    print(f"Model loaded from {weights_path}")
    return model

def upscale_images(model, input_folder="LR", output_folder="results", device="cpu"):
    """
    Upscale images from the input folder and save the results in the output folder.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each image in the input folder
    for img_path in glob.glob(f"{input_folder}/*"):
        # Get the base name of the image
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"Processing image: {base_name}")

        # Read and preprocess the image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Skipping invalid image: {img_path}")
            continue
        img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).unsqueeze(0).to(device)  # Convert to tensor

        # Perform upscaling
        with torch.no_grad():
            output = model(img).squeeze(0).cpu().numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # Convert back to HWC format
        output = (output * 255.0).clip(0, 255).astype(np.uint8)  # Convert to uint8

        # Save the upscaled image
        output_path = os.path.join(output_folder, f"{base_name}_upscaled.png")
        cv2.imwrite(output_path, output)
        print(f"Upscaled image saved to: {output_path}")

def main():
    # Define paths
    weights_path = "models/upscaler_model.pth"  # Path to the saved model weights
    input_folder = "LR"  # Folder containing low-resolution images
    output_folder = "results"  # Folder to save upscaled images

    # Check if the weights file exists
    if not os.path.exists(weights_path):
        print(f"Model weights not found at {weights_path}. Please train the model first.")
        return

    # Determine the device (CPU or CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    model = load_model(weights_path, device=device)

    # Upscale images
    upscale_images(model, input_folder=input_folder, output_folder=output_folder, device=device)

if __name__ == "__main__":
    main()