# Image Upscaler UI

This project is a web application built using Streamlit that allows users to upscale low-resolution images using the ESRGAN model. The application provides an intuitive interface for uploading images and processing them on an AWS EC2 instance equipped with a GPU.

# Motivation, Vision, and Technical Implementation of Image Upscaler

The following [slides](https://docs.google.com/presentation/d/1s0bnAiOKTtljY6oqF3TKh6VBiRwCB7PGg7YUc4OLIbQ/edit?usp=sharing) is how we approached designing and modularized the process of creating the AI Image Upscaler model and having support for an intuitive interface for uploading images or videos to generate high quality images or videos.

# Demo

[![Demo of Application](https://img.youtube.com/vi/acGnMiEcAGg/maxresdefault.jpg)](https://www.youtube.com/watch?v=acGnMiEcAGg)

## Project Structure

```
gan-upscale
├── UpscalerModel
│   ├── upscaler.py                # Script for training and saving the upscaler model
│   ├── upscaler_test.py           # Script for testing the upscaler model on images
│   ├── models                     # Directory containing the trained model weights
│   │   └── upscaler_model.pth     # Pre-trained model weights
│   └── LR                         # Directory for low-resolution input images
│   └── LR_Videos                  # Directory for low-resolution input videos
│   └── results                    # Directory for upscaled images
│   └── results_videos             # Directory for upscaled videos 
│
├── VideoProcessing
│   ├── VideoDataSetProcessing.py  # Functions for video deconstruction and reconstruction
│   └── destructed                 # Directory for destructed video frames and audio
│   └── reconstructed              # Directory for reconstructed video from audio and upscaled frames
│
├── image-upscaler-ui
│   ├── src
│   │   ├── app.py                 # Main entry point for the Streamlit application
│   │   ├── aws_utils.py           # Utility functions for AWS interactions
│   │   ├── local_utils.py         # Utility functions for local processing
│   │   └── credentials.py         # File to store SSH credentials
│   └── requirements.txt           # Python dependencies for the UI
│
└── README.md                      # Project documentation
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd gan-upscale
```

### 2. Install Dependencies
It is recommended to use a virtual environment. You can create one using `venv` or `conda`:
```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
# or
venv\Scripts\activate     # On Windows

pip install -r image-upsclaer-ui/requirements.txt
```

### 3. Configure SSH Credentials
Update the image-upscaler-ui/src/credentials.py file with your EC2 instance details:
```bash
SSH_HOST = "your-ec2-public-ip"
SSH_USER = "your-ssh-username"
SSH_KEY_PATH = "/path/to/your/private-key.pem"
```

### 4. Prepare the EC2 Instance
To run the upscaling model on an EC2 instance, follow these steps:

#### a. Get EC2 Instance Info
1. Go to Courseworks > Deep Learning > Go to Announcements
2. Find the most recent Announcement containing the ec2 instance's public ip

#### b. Connect to the EC2 Instance
SSH into the instance: 
```bash
ssh -i /path/to/your/private-key.pem your-ssh-username@your-ec2-public-ip
```

#### c. Install Required Software
Once connected, install the necessary dependencies:
```bash
pip install torch torchvision
```

#### d. Clone the Repository on EC2
Clone the gan-upscale repository on the EC2 instance:
```bash
git clone https://github.com/your-repo/gan-upscale.git
cd gan-upscale
```

#### e. Prepare Directories
Clone the ESRGAN repository and place the model weights in the appropriate directory:
Ensure the following directories exist on the EC2 instance:

- ```UpscalerModel/LR```: For low-resolution input images.
- ```UpscalerModel/results```: For upscaled image outputs.
- ```VideoProcessing/destructed```: For destructed video frames and audio.
- ```VideoProcessing/reconstructed```: For reconstructed upscaled videos.

### 5. Run the Application

Once the ESRGAN repository and model weights are set up on the EC2 instance, the directory structure should look like this:

```
ESRGAN/
├── LICENSE
├── LR/                   # Folder for low-resolution input images
├── QA.md
├── README.md
├── RRDBNet_arch.py
├── __pycache__/          # Python cache files
├── figures/              # Example figures (optional)
├── models/               # Folder containing the ESRGAN model weights
│   └── RRDB_ESRGAN_x4.pth
├── net_interp.py
├── results/              # Folder where upscaled images will be saved
├── test.py               # Script to run the ESRGAN model
└── transer_RRDB_models.py
```

#### f. Test the Setup
Run the ESRGAN test script to ensure everything is working:
```bash
python test.py
```
You should see the results/ directory populated with the higher resolution images. 

### 5. Run the Application
Start the Streamlit application locally:
```bash
streamlit run image-upscaler-ui/src/app.py
```

## Usage Guideliness
Image Upscaling
1. Upload a Low-Resolution Image:
   Use the Streamlit interface to upload a low-resolution image.
2. Process the Image:
   - The application will transfer the image to the EC2 instance.
   - The upscaler model will process the image on the EC2 instance.
   - The upscaled image will be downloaded back to your local machine.
3. View and Download Results:
   - The low-resolution and upscaled images will be displayed side-by-side.
   - A download button will be provided to save the upscaled image.

Video Upscaling
1. Upload a Low-Resolution Video:
   Use the Streamlit interface to upload a low-resolution video.
2. Process the Video:
   - The application will deconstruct the video into frames and audio.
   - The frames will be processed using the upscaler model on the EC2 instance.
   - The upscaled frames and audio will be reconstructed into a high-resolution video.
3. View and Download Results:
   The upscaled video will be available for download.

### Image and Video Upscaling Process
The application utilizes a trained upscaler model for enhancing the resolution of images and videos. When a file is uploaded, it is processed to improve its resolution while maintaining quality. The upscaled results are then made available for download.

For videos, the process involves:

1. Deconstructing the video into individual frames and extracting the audio.
2. Upscaling the frames using the upscaler model.
3. Reconstructing the video from the upscaled frames and original audio.

This workflow ensures high-quality results for both images and videos.

### Pretrained Weights and Model Training
The application uses pretrained weights located in the ```models/``` directory within ```UpscalerModel/```. These weights are used by default for upscaling images and videos.
If you need to create new weights or would like to perform the time-consuming task of training the model, you can run the following script:
```bash
python UpscalerModel/upscaler.py
```
This script will train the upscaler model using the provided dataset and save the new weights in the models/ directory.

To utilize the model outside of the user interface, input your low resolution 
images into ```UpscalerModel/LR/``` and run: 
```bash
python UpscalerModel/upscaler_test.py 
```
The upscaled images will then populate the ```UpscalerModel/results/``` folder. 
