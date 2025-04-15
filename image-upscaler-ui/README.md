# Image Upscaler UI

This project is a web application built using Streamlit that allows users to upscale low-resolution images using the ESRGAN model. The application provides an intuitive interface for uploading images and processing them either locally or on an AWS server equipped with a GPU.

## Project Structure

```
image-upscaler-ui
├── src
│   ├── app.py          # Main entry point for the Streamlit application
│   ├── aws_utils.py    # Utility functions for AWS interactions
│   ├── local_utils.py   # Utility functions for local processing
│   └── types
│       └── index.py    # Type definitions and interfaces
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd image-upscaler-ui
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment. You can create one using `venv` or `conda`.
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure AWS Credentials:**
   Ensure that your AWS credentials are configured properly. You can set them up using the AWS CLI or by creating a `~/.aws/credentials` file.

4. **Run the application:**
   Start the Streamlit application by running:
   ```bash
   streamlit run src/app.py
   ```

## Usage Guidelines

- Upload a low-resolution image using the provided interface.
- Choose whether to process the image locally or on the AWS server.
- If processing on AWS, the application will handle the upload of the image, trigger the model execution, and download the upscaled image back to your local environment.
- The results will be saved in the `results/` folder.

## Image Upscaling Process

The application utilizes the ESRGAN model for image upscaling. When an image is uploaded, it is processed to enhance its resolution while maintaining quality. The upscaled images are then made available for download.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.