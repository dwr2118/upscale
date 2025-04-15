import streamlit as st
import os
from aws_utils import run_model_on_ec2
from local_utils import save_low_res_image

# Import credentials
try:
    from credentials import SSH_HOST, SSH_USER, SSH_KEY_PATH
except ImportError:
    st.error("Error: `credentials.py` file is missing or improperly configured.")
    st.stop()

# Validate credentials
if not SSH_HOST or not SSH_USER or not SSH_KEY_PATH:
    st.error("Error: SSH credentials are not properly defined in `credentials.py`.")
    st.stop()

def main():
    st.title("Image Upscaler using ESRGAN")
    
    # File uploader for low-resolution images
    uploaded_file = st.file_uploader("Choose a low-resolution image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Save the uploaded image locally
        save_low_res_image(uploaded_file, "LR")
        
        if st.button("Upscale Image"):
            
            st.write("Processing on EC2 instance via SSH...")
            run_model_on_ec2(
                ssh_host=SSH_HOST,
                ssh_user=SSH_USER,
                ssh_key_path=SSH_KEY_PATH,
                local_image_path="LR/" + uploaded_file.name,
                
                remote_image_path="/home/diegorivaslazala/final/ESRGAN/LR/" + uploaded_file.name,
                remote_results_path="/home/diegorivaslazala/final/ESRGAN/results/",
                local_results_path="results/"
            )
            st.success("Image upscaling completed! Check the results folder.")
            
            # Display the low-resolution and upscaled images
            low_res_path = "LR/" + uploaded_file.name
            upscaled_image_name = uploaded_file.name.split('.')[0]  # Extract the base name without extension
            results_folder = "results/"
            
            # Find the upscaled image in the results folder
            upscaled_image_path = None
            for file in os.listdir(results_folder):
                if file.startswith(upscaled_image_name):
                    upscaled_image_path = os.path.join(results_folder, file)
                    break
            
            if upscaled_image_path:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("### Low-Resolution Image:")
                    st.image(low_res_path, caption="Low-Resolution Image", use_container_width=True)
                
                with col2:
                    st.write("### Upscaled Image:")
                    st.image(upscaled_image_path, caption="Upscaled Image", use_container_width=True)
            
                # Add a download button for the upscaled image
                with open(upscaled_image_path, "rb") as file:
                    btn = st.download_button(
                        label="Download Upscaled Image",
                        data=file,
                        file_name=os.path.basename(upscaled_image_path),
                        mime="image/png"
                    )
            else:
                st.error("Upscaled image not found in the results folder.")

if __name__ == "__main__":
    main()