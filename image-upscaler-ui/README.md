# Image Upscaler UI

This project is a web application built using Streamlit that allows users to upscale low-resolution images using the ESRGAN model. The application provides an intuitive interface for uploading images and processing them on an AWS EC2 instance equipped with a GPU.

## Project Structure

```
image-upscaler-ui
├── src
│   ├── app.py          # Main entry point for the Streamlit application
│   ├── aws_utils.py    # Utility functions for AWS interactions
│   ├── local_utils.py  # Utility functions for local processing
│   ├── credentials.py  # File to store SSH credentials
│   └── types
│       └── index.py    # Type definitions and interfaces
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd image-upscaler-ui
```

### 2. Install Dependencies
It is recommended to use a virtual environment. You can create one using `venv` or `conda`:
```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
# or
venv\Scripts\activate     # On Windows

pip install -r requirements.txt
```

### 3. Configure SSH Credentials
Update the `src/credentials.py` file with your EC2 instance details:
```python
# filepath: src/credentials.py
SSH_HOST = "your-ec2-public-ip"
SSH_USER = "your-ssh-username"
SSH_KEY_PATH = "/path/to/your/private-key.pem"
```

### 4. Prepare the EC2 Instance
To run the ESRGAN model on an EC2 instance, follow these steps:

#### a. Launch an EC2 Instance
1. Go to Courseworks > Deep Learning > Go to Announcements
2. Find the most recent Announcement containing the ec2 instance's public ip

#### b. Connect to the EC2 Instance
SSH into the instance: 
```bash
ssh -i /path/to/your/private-key.pem your-ssh-username@your-ec2-public-ip
```

#### c. Install Required Software
Once connected, install the necessary dependency:
```bash
pip install torch torchvision gdown
```

#### d. Download ESRGAN Model Weights
Use `gdown` to download the pre-trained ESRGAN model weights:
```bash
gdown https://drive.google.com/uc?id=1TPrz5QKd8DHHt1k8SRtm6tMiPjz_Qene -O RRDB_ESRGAN_x4.pth
```

#### e. Clone the ESRGAN Repository
Clone the ESRGAN repository and place the model weights in the appropriate directory:
```bash
git clone https://github.com/xinntao/ESRGAN.git
cd ESRGAN
mv ../RRDB_ESRGAN_x4.pth models/
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
streamlit run src/app.py
```

## Usage Guidelines

1. **Upload a Low-Resolution Image:**
   Use the Streamlit interface to upload a low-resolution image.

2. **Process the Image:**
   - The application will transfer the image to the EC2 instance.
   - The ESRGAN model will upscale the image on the EC2 instance.
   - The upscaled image will be downloaded back to your local machine.

3. **View and Download Results:**
   - The low-resolution and upscaled images will be displayed side-by-side.
   - A download button will be provided to save the upscaled image.

## Image Upscaling Process

The application utilizes the ESRGAN model for image upscaling. When an image is uploaded, it is processed to enhance its resolution while maintaining quality. The upscaled images are then made available for download.