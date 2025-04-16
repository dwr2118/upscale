import torch
import torch.nn as nn
import os
import cv2
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from torchvision.transforms import ToTensor
import random
import time

import glob

# Set default tensor type
torch.set_default_dtype(torch.float32)


# ==== RESIDUAL BLOCK ====
class ResidualBlock(nn.Module):
    """
    Residual Block with two conv layers
    Helps preserve info and eases training of deep models
    """
    def __init__(self, num_features):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out + residual

# ==== UPSCALER CNN ====
class UpscalerCNN(nn.Module):
    """
    A CNN for image upscaling that:
      - Extracts features from the low res img,
      - Uses several residual blocks to refine features,
      - Upsamples the features using PixelShuffle,
      - Reconstructs a high-resolution image
    """
    
    def __init__(self, upscale_factor, num_channels, base_filter):
        super(UpscalerCNN, self).__init__()

        # Resize factor
        self.upscale_factor = upscale_factor

        # Input conv layer w/ a relatively large kernel helps initial feature extraction
        self.input_conv = nn.Conv2d(num_channels, base_filter, kernel_size=9, padding=4)
        self.relu = nn.ReLU(inplace=True)
        
        # 5 res blocks to refine features
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(base_filter) for _ in range(5)]
        )
        
        # A conv layer applied after the res blocks
        # We add output of this layer to the output from input_conv (res connection)
        self.mid_conv = nn.Conv2d(base_filter, base_filter, kernel_size=3, padding=1)
        self.bn_mid = nn.BatchNorm2d(base_filter)
        
        # Upsampling layers:
        # Increase the num of channels to base_filter * (upscale_factor^2) to prepare for PixelShuffle
        self.upsample = nn.Sequential(
            nn.Conv2d(base_filter, base_filter * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.ReLU(inplace=True)
        )
        
        # Output convolution layer to reconstruct the img
        self.output_conv = nn.Conv2d(base_filter, num_channels, kernel_size=9, padding=4)
    
    def forward(self, x):
        # Extract features from the low res img
        out1 = self.relu(self.input_conv(x))
        
        # Pass through the residual blocks
        out = self.res_blocks(out1)
        
        # Process features further & add the initial features (global res connection)
        out = self.bn_mid(self.mid_conv(out))
        out = out1 + out
        
        # Upsample the feature maps
        out = self.upsample(out)
        
        # Reconstruct the high res img
        out = self.output_conv(out)
        return out

# ==== DATASET ====
class UpscalerDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        low_res_img, high_res_img = self.image_paths[idx]

        low_img = Image.open(low_res_img).convert("RGB")
        high_img = Image.open(high_res_img).convert("RGB")

        # Resize images because input images and output images are not uniform
        # respectively
        # High resolution: target dimensions 300 (height) x 168 (width)
        # Low resolution: target dimensions 150 (height) x 84 (width)
        high_img = high_img.resize((300, 168), Image.BICUBIC)
        low_img = low_img.resize((150, 84), Image.BICUBIC)


        norm_low_img = np.array(low_img).astype(np.float32) / 255.0  # Normalize RGB to [0, 1]
        norm_high_img = np.array(high_img).astype(np.float32) / 255.0 # Normalize RGB to [0, 1]

        low_tensor = torch.from_numpy(norm_low_img).permute(2, 0, 1)  # Shape: [3, H, W]
        high_tensor = torch.from_numpy(norm_high_img).permute(2, 0, 1)  # Shape: [3, H, W]

        return low_tensor, high_tensor

# ==== TEST, EVALUATE, AND SAVE MODEL IMAGES ====
def evaluate_and_save(model, test_loader, device, output_folder="PredictedImg"):
    # make the PredictedColorizedImg directory if it exist
    os.makedirs(output_folder, exist_ok=True)

    # Evaluate the model 
    model.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for i, (low_res_batch, high_res_batch) in enumerate(test_loader):
            low_res_batch = low_res_batch.to(device)
            high_res_batch = high_res_batch.to(device)
            preds = model(low_res_batch)
            loss = nn.functional.mse_loss(preds, high_res_batch, reduction='mean')
            total_loss += loss.item()
            # count += low_res_batch.size(0)
            count += 1

            # Save RGB colorized results
            for j in range(low_res_batch.size(0)):
                rgb = preds[j].cpu().numpy().transpose(1, 2, 0) # shape [C, H, W]
                rgb_img = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                img = Image.fromarray(rgb_img)
                img.save(os.path.join(output_folder, f"test_img_{i*10 + j}.png"))

    # Print out mse loss of the testing
    mse = total_loss / count
    print(f"Test MSE: {mse:.6f}")
    output_file = "cpu_upscaler.txt"  # Specify the file name
    with open(output_file, "a") as f:
        print(f"Test MSE: {mse:.6f}", file=f)

def degrade_image_quality(image, quality=30, resize_scale=0.5):
    # Resize the image (optional degradation)
    if resize_scale < 1.0:
        new_width = int(image.shape[1] * resize_scale)
        new_height = int(image.shape[0] * resize_scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Encode to JPEG to reduce quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, compressed_img = cv2.imencode('.jpg', image, encode_param)

    # Decode back into image format
    degraded = cv2.imdecode(compressed_img, 1)
    return degraded

def augment_images(input_dir='HiResImages', output_dir='Augmented', quality=30, resize_scale=0.5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    processed_count = 0

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)

        # Extract the file name and extension
        name, ext = os.path.splitext(filename)
        
        # Create the output file name with '_aug' appended
        output_filename = f"{name}_aug{ext}"
        output_path = os.path.join(output_dir, output_filename)

        # Read the image
        image = cv2.imread(input_path)
        if image is None:
            print(f"Skipping non-image or unreadable file: {filename}")
            continue

        # Degrade image quality
        degraded = degrade_image_quality(image, quality=quality, resize_scale=resize_scale)

        # Save the degraded image
        cv2.imwrite(output_path, degraded)
        processed_count += 1
        
    print(f"Processed {processed_count} images")

# ==== MAIN ====
def main():
    # Get the current working directory (where the script is being run from)
    base_dir = os.getcwd()

    # Construct the path to the 'face_images' directory
    high_res_img = os.path.join(base_dir, "high_res_img")

    # Fetch all image filenames from the face_images folder
    high_res_paths = [os.path.join(high_res_img, fname)
                            for fname in os.listdir(high_res_img)
                            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Split 90% of the face images to augment, 10% to remain the same
    # train_size = 0.9
    # test_size = 0.1
    # to_augment, to_original = random_split(high_res_paths, [train_size, test_size])

    # Construct path to augmented directory
    low_res_img = os.path.join(base_dir, "low_res_img/")

    # Delete all files inside the augmented directory if there are any
    if os.path.exists(low_res_img):
        files = glob.glob(os.path.join(low_res_img, '*'))
        for f in files:
            os.remove(f)

    # Generated new augmented images
    augment_images("high_res_img", "low_res_img")

    # Fetch all image file names in the augmented path
    aug_paths = [os.path.join(low_res_img, fname)
                            for fname in os.listdir(low_res_img)
                            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Kinda redundant, but just changed variable name to store it in
    # train_aug_paths = aug_paths

    dataset = list(zip(aug_paths, high_res_paths))

    # train_res_paths, test_res_paths

    train_size = 0.9
    test_size = 0.1
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


    # Grab the L*, ab* colorized dataset
    train_dataset = UpscalerDataset(train_dataset)
    
    # Redundant again, but keep the last 10% original into the test dataset
    # test_high_res_paths = to_original

    # Grab the L*, ab* of the actual face images
    test_dataset = UpscalerDataset(test_dataset)

    # DataLoad both the training and testing data set
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

    # Grab custom made Colorization CNN model
    upscale_factor = 2 # Images are reduced by 50% size, so need to double output
    num_channels = 3 # RGB has 3 channels
    base_filter = 64 # base filter used for upsampling with pixel shuffling
    model = UpscalerCNN(upscale_factor, num_channels, base_filter)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Determine whether we're running the model on cuda or cpu 
    device = torch.device("cpu")
    model.to(device)
    
    train_start_time = time.time()
    
    # Running with 10 epochs
    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0
        # Train model first
        model.train()
        
        # For each low_res_batch, high_res_batch from the train_loader
        # Check how accurate the images are to the colored one
        for low_res_batch, high_res_batch in train_loader:
            low_res_batch, high_res_batch = low_res_batch.to(device), high_res_batch.to(device)

            # clears gradient from prev step
            optimizer.zero_grad()

            # make prediction
            preds = model(low_res_batch)

            # Calculate loss
            loss = criterion(preds, high_res_batch)

            # perform backpropagation
            loss.backward()

            # Update model's parameters 
            optimizer.step()

            total_loss += loss.item()
        
        # Print out epoch results
        # print(f"Epoch {epoch+1}/{num_epochs}: Loss = {total_loss/loss.item():.4f}")
        print(f"Epoch {epoch+1}/{num_epochs}: Training Loss = {total_loss/len(train_loader):.4f}")
        # print(f"Epoch {epoch+1}/{num_epochs}: Training Loss = {total_loss:.4f}")
        
        output_file = "cpu_upscaler.txt"  # Specify the file name
        with open(output_file, "a") as f:
            print(f"--- CPU Colorization Results ---", file=f)
            print(f"Epoch {epoch+1}/{num_epochs}: Training Loss = {total_loss/len(train_loader):.4f}", file=f)
        
    train_end_time = time.time()
    print(f"Training time: {train_end_time - train_start_time:.2f} seconds")
    # Write the runtime to a file
    output_file = "cpu_runtime_log.txt"  # Specify the file name
    with open(output_file, "a") as f:
        print(f"Training time: {train_end_time - train_start_time:.2f} seconds", file=f)
        print(f"---------------------------------------", file=f)
    
    # Construct path to the predicted colorized img directory
    predict_dir = os.path.join(base_dir, "PredictedImg/")
    # Create folders if they do not exist
    os.makedirs(predict_dir, exist_ok=True)

    # Clear all files inside this folder from previous run
    if os.path.exists(predict_dir):
        files = glob.glob(os.path.join(predict_dir, '*'))
        for f in files:
            os.remove(f)

    # After training, save predictions
    evaluate_and_save(model, test_loader, device=device)

if __name__ == "__main__":
    # Record the start time
    start_time = time.time()
    
    # Call the main function
    main()
    
    # Record the end time
    end_time = time.time()
    
    # Calculate the total runtime
    total_time = end_time - start_time
    
    # Write the runtime to a file
    output_file = "cpu_runtime_log.txt"  # Specify the file name
    with open(output_file, "a") as f:
        print(f"---------------------------------------", file=f)
        print(f"Total runtime: {total_time:.2f} seconds", file=f)
    
    # Print to console total runtime too 
    print(f"Total runtime: {total_time:.2f} seconds")