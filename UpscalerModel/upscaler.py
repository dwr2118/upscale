import torch
import torch.nn as nn
import os
import cv2
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import KFold
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
    Helps preserve info and eases training of deep g_models
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

# ==== DISCRIMINATOR CNN ====
# Source for understanding Discriminator:
# https://www.geoffreylitt.com/2017/06/04/enhance-upscaling-images-with-generative-adversarial-neural-networks
class DiscriminatorCNN(nn.Module):
    """
    Uses a PatchGAN discriminator approach 
    Classifies each N x N patch as real or fake.
    Given an input of 3 x H x W and and output of 1 x (H/16) x (W/16) 
    (for H,W divisible by 16)
    """
    def __init__(self, in_channels=3, base_filters=64):
        super(DiscriminatorCNN, self).__init__()

        def conv_block(in_c, out_c, stride=2, batch_norm=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            # input is (in_channels) x H x W
            conv_block(in_channels,       base_filters,  stride=2,  batch_norm=False),  # 64 x H/2 x W/2
            conv_block(base_filters,      base_filters*2,          stride=2),          # 128 x H/4 x W/4
            conv_block(base_filters*2,    base_filters*4,          stride=2),          # 256 x H/8 x W/8
            conv_block(base_filters*4,    base_filters*8,          stride=1),          # 512 x H/8 x W/8
            # final conv to get 1-channel output (the “realness” score per patch)
            nn.Conv2d(base_filters*8, 1, kernel_size=4, stride=1, padding=1)          # 1 x H/8-1 x W/8-1
        )

    def forward(self, x):
        """
        x: tensor of shape (B, in_channels, H, W)
        returns: tensor of shape (B, 1, H_out, W_out) w/ logits
        """
        return self.model(x)

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
        # High resolution: target dimensions 168 (height) x 300 (width)
        # Low resolution: target dimensions 84 (height) x 150 (width)
        high_img = high_img.resize((300, 168), Image.BICUBIC)
        low_img = low_img.resize((150, 84), Image.BICUBIC)


        norm_low_img = np.array(low_img).astype(np.float32) / 255.0  # Normalize RGB to [0, 1]
        norm_high_img = np.array(high_img).astype(np.float32) / 255.0 # Normalize RGB to [0, 1]

        low_tensor = torch.from_numpy(norm_low_img).permute(2, 0, 1)  # Shape: [3, H, W]
        high_tensor = torch.from_numpy(norm_high_img).permute(2, 0, 1)  # Shape: [3, H, W]

        return low_tensor, high_tensor

# ==== TEST, EVALUATE, AND SAVE g_model IMAGES ====
def evaluate_and_save(g_model, test_loader, device, output_folder="PredictedImg"):
    # make the PredictedColorizedImg directory if it exist
    os.makedirs(output_folder, exist_ok=True)

    # Evaluate the g_model 
    g_model.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for i, (low_res_batch, real_high_res_batch) in enumerate(test_loader):
            low_res_batch = low_res_batch.to(device)
            real_high_res_batch = real_high_res_batch.to(device)
            preds = g_model(low_res_batch)
            loss = nn.functional.mse_loss(preds, real_high_res_batch, reduction='mean')
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
    output_file = "upscaler_res.txt"  # Specify the file name
    with open(output_file, "a") as f:
        print(f"Test MSE: {mse:.6f}", file=f)

# ==== DATA AUGMENTATION ====
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

def augment_images(input_dir, input_files='HiResImages', output_dir='Augmented', quality=30, resize_scale=0.5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    processed_count = 0
    for filename in input_files:

        # Get base name of file
        base = os.path.basename(filename)
        input_path = os.path.join(input_dir, base)

        # Extract the file name and extension
        name, ext = os.path.splitext(base)

        # Read the image
        image = cv2.imread(input_path)
        if image is None:
            print(f"Skipping non-image or unreadable file: {filename}")
            continue
        
        # Loop through degradation levels: 30, 40, 50, 60, 70
        for i, quality in enumerate(range(30, 71, 10), start=1):
            # Create the output file name with '_aug' appended
            output_filename = f"{name}_aug_{str(i).zfill(2)}_q{quality}{ext}"
            output_path = os.path.join(output_dir, output_filename)
        
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
    
    
    # Split 90% of the face images to augment, 10% to augment for testing
    train_size = 0.9
    test_size = 0.1
    to_augment, to_augment_for_test = random_split(high_res_paths, [train_size, test_size])

    # Construct path to augmented directory
    low_res_img = os.path.join(base_dir, "low_res_img/")
    low_res_img_test = os.path.join(base_dir, "low_res_img_test/")

    # Delete all files inside the augmented directory if there are any
    if os.path.exists(low_res_img):
        files = glob.glob(os.path.join(low_res_img, '*'))
        for f in files:
            os.remove(f)

     # Delete all files inside the augmented directory if there are any
    if os.path.exists(low_res_img_test):
        files = glob.glob(os.path.join(low_res_img_test, '*'))
        for f in files:
            os.remove(f)

    # Generated new augmented images
    augment_images("high_res_img", to_augment, low_res_img)

    # Fetch all image file names in the augmented path
    aug_paths = [os.path.join(low_res_img, fname)
                            for fname in os.listdir(low_res_img)
                            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Connect all low res training files with the original high res one
    train_dataset = []

    next_img = 0
    count = 0
    for train_img in aug_paths:
        train_dataset.append((train_img, to_augment[next_img]))
        if count != 4:
            count += 1
        else:
            next_img += 1
            count = 0

    augment_images("high_res_img", to_augment_for_test, low_res_img_test)

    aug_paths_test = [os.path.join(low_res_img_test, fname)
                            for fname in os.listdir(low_res_img_test)
                            if fname.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Connect all low res testing files with the original high res one
    test_dataset = []
    next_img = 0
    count = 0
    for test_img in aug_paths_test:
        test_dataset.append((test_img, to_augment_for_test[next_img]))
        if count != 4:
            count += 1
        else:
            next_img += 1
            count = 0

    # Grab the RGB tensor from the anime dataset
    train_dataset = UpscalerDataset(train_dataset)

    # Grab the RGB tensor from the anime dataset
    test_dataset = UpscalerDataset(test_dataset)

    # DataLoad the testing data set for later evaluation
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)
    # # train_loader = DataLoader(train_dataset, batch_size=5, shuffle = False)
    # # test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

    # # Grab custom made Upscaler CNN Generator Model and Discriminator Model
    # upscale_factor = 2 # Images are reduced by 50% size, so need to double output
    # num_channels = 3 # RGB has 3 channels
    # base_filter = 64 # base filter used for upsampling with pixel shuffling
    # g_model = UpscalerCNN(upscale_factor, num_channels, base_filter)
    # d_model = DiscriminatorCNN()

    # # Generator and Discriminator Model
    # G_optimizer = torch.optim.Adam(g_model.parameters(), lr=1e-4)
    # D_optimizer = torch.optim.Adam(d_model.parameters(), lr=1e-4)

    # adversarial loss uses binary cross entropy
    adv_loss = nn.BCEWithLogitsLoss()
    # pixel loss for generator uses MSE
    pixel_loss = nn.MSELoss()

    # lambda term for adversarial loss in generator
    l_adv = 1e-3

    # # Determine whether we're running the g_model on cuda or cpu 
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # g_model.to(device)
    # d_model.to(device)
    
    # train_start_time = time.time()


    # 5 fold cross validation applied
    n_splits = 5
    kfold = KFold(n_splits=n_splits, shuffle = True, random_state=42)

    # Running with 10 epochs
    num_epochs = 10
    
    # Training time results
    upscaler_val_f = "upscaler_validate.txt"  # Specify the file name
    upscaler_epoch_f = "upscaler_epoch.txt"

    val_f = open(upscaler_val_f, "a")
    epoch_f = open(upscaler_epoch_f, "a")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):

        # Have both 
        train_sub = Subset(train_dataset, train_idx)
        val_sub = Subset(train_dataset, val_idx)
        # DataLoad both the training and testing data set
        train_loader = DataLoader(train_sub, batch_size=5, shuffle = True)
        val_loader = DataLoader(val_sub, batch_size=5, shuffle = False)

        # Grab custom made Upscaler CNN Generator Model and Discriminator Model
        upscale_factor = 2 # Images are reduced by 50% size, so need to double output
        num_channels = 3 # RGB has 3 channels
        base_filter = 64 # base filter used for upsampling with pixel shuffling
        g_model = UpscalerCNN(upscale_factor, num_channels, base_filter)
        d_model = DiscriminatorCNN()

        # Generator and Discriminator Model
        G_optimizer = torch.optim.Adam(g_model.parameters(), lr=1e-4)
        D_optimizer = torch.optim.Adam(d_model.parameters(), lr=1e-4)

        # Determine whether we're running the g_model on cuda or cpu 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        g_model.to(device)
        d_model.to(device)

        train_start_time = time.time()

        for epoch in range(num_epochs):
            # total_loss = 0
            # Train g_model first
            g_model.train()
            d_model.train()
            
            # For each low_res_batch, real_high_res_batch from the train_loader
            # Check how accurate the images are to the colored one
            for low_res_batch, real_high_res_batch in train_loader:
                low_res_batch, real_high_res_batch = low_res_batch.to(device), real_high_res_batch.to(device)

                # make generated image from Generator Model
                pred_gen_img = g_model(low_res_batch)

                """ First train on Discriminator """
                # Clear gradient from previous step
                D_optimizer.zero_grad()

                # Predict on real high res image and adversarial loss
                pred_real_img = d_model(real_high_res_batch)
                loss_real_img = adv_loss(pred_real_img, torch.ones_like(pred_real_img))

                # Predict on fake high res image and adversarial loss
                pred_fake_img = d_model(pred_gen_img.detach())
                loss_fake_img = adv_loss(pred_fake_img, torch.zeros_like(pred_fake_img))

                # Add up total loss and propagate the loss back and update the step
                loss_discriminator = 0.5 * (loss_real_img + loss_fake_img)
                loss_discriminator.backward()
                D_optimizer.step()


                """ Then train on Generator """

                # Clears gradient from prev step
                G_optimizer.zero_grad()

                # Calculate loss
                loss_pixel = pixel_loss(pred_gen_img, real_high_res_batch)

                pred_fake_for_G = d_model(pred_gen_img)
                loss_adv = adv_loss(pred_fake_for_G, torch.ones_like(pred_fake_for_G))

                loss_generator = loss_pixel + l_adv * loss_adv

                # perform backpropagation
                loss_generator.backward()

                # Update g_model's parameters 
                G_optimizer.step()

                # ---- VALIDATE ----
                g_model.eval()
                val_pixel_loss = 0.0
                with torch.no_grad():
                    for low_res, high_res in val_loader:
                        low_res, high_res = low_res.to(device), high_res.to(device)
                        gen = g_model(low_res)
                        val_pixel_loss += pixel_loss(gen, high_res).item()
                val_pixel_loss /= len(val_loader)

                # total_loss += loss_pixel.item()
            
            # Print out epoch results
            # print(f"Epoch {epoch+1}/{num_epochs}: Training Loss = {total_loss/len(train_loader):.4f}")
            # print(f"Fold {fold + 1} [Epoch {epoch+1}/{num_epochs}] "
            # f"D Loss: {loss_discriminator.item():.4f}, G Loss: {loss_generator.item():.4f} "
            # f"(pix: {loss_pixel.item():.4f}, adv: {loss_adv.item():.4f}) ")
            
                # print(f"--- Upscaler Results ---", file=f)
                # print(f"Epoch {epoch+1}/{num_epochs}: Training Loss = {total_loss/len(train_loader):.4f}", file=f)

                print(f"--- Validation Results ---", file=val_f)
                print(f"Fold {fold + 1} [Epoch {epoch+1}/{num_epochs}] "
                f"Val MSE = {val_pixel_loss:.4f}", file=val_f)
            
            print(f"Fold {fold + 1} [Epoch {epoch+1}/{num_epochs}] "
            f"D Loss: {loss_discriminator.item():.4f}, G Loss: {loss_generator.item():.4f} "
            f"(pix: {loss_pixel.item():.4f}, adv: {loss_adv.item():.4f}) ", file=epoch_f)
        
    train_end_time = time.time()
    print(f"Training time: {train_end_time - train_start_time:.2f} seconds")
    # Write the runtime to a file
    output_file = "upscaler_runtime_log.txt"  # Specify the file name
    with open(output_file, "a") as f:
        print(f"Training time: {train_end_time - train_start_time:.2f} seconds", file=f)
        print(f"---------------------------------------", file=f)
        f.close()
        
    epoch_f.close()
    val_f.close()
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
    evaluate_and_save(g_model, test_loader, device=device)

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
    output_file = "upscaler_runtime_log.txt"  # Specify the file name
    with open(output_file, "a") as f:
        print(f"---------------------------------------", file=f)
        print(f"Total runtime: {total_time:.2f} seconds", file=f)
    
    # Print to console total runtime too 
    print(f"Total runtime: {total_time:.2f} seconds")