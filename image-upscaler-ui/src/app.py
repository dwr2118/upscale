import streamlit as st
import os
from aws_utils import run_model_on_ec2, run_video_model_on_ec2
from local_utils import save_low_res_image, save_low_res_video

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
    st.title("Image and Video Upscaler using ESRGAN")
    
    # File uploader for low-resolution images or videos
    uploaded_file = st.file_uploader("Choose a low-resolution image or video...", type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"])
    
    #TODO: need to handle file names with spaces in them 
    
    if uploaded_file is not None:
        # Determine if the uploaded file is an image or a video
        file_extension = os.path.splitext(uploaded_file.name)[-1].lower()
        is_video = file_extension in ('.mp4', '.avi', '.mov', '.mkv')
        
        if is_video:
            st.write("Detected a video file. Processing will use the video upscaling pipeline.")
            
            # Save the uploaded video locally
            save_low_res_video(uploaded_file, "LR_Videos")
            
            if st.button("Upscale Video"):
                st.write("Processing video on EC2 instance via SSH...")
                run_video_model_on_ec2(
                    ssh_host=SSH_HOST,
                    ssh_user=SSH_USER,
                    ssh_key_path=SSH_KEY_PATH,
                    local_video_path="LR_Videos/" + uploaded_file.name,
                    remote_video_path=f"/home/{SSH_USER}/gan-upscale/UpscalerModel/LR_Videos/" + uploaded_file.name,
                    remote_results_path=f"/home/{SSH_USER}/gan-upscale/UpscalerModel/results_videos/",
                    local_results_path="results_videos/"
                )
                st.success("Video upscaling completed! Check the results_videos/ folder.")
                
                # Add a download button for the upscaled video
                upscaled_video_path = "results_videos/" + "reconstructed_" + os.path.splitext(uploaded_file.name)[0] + ".mp4"

                # Display the low-resolution and upscaled videos
                col1, col2 = st.columns(2)

                with col1:
                    st.write("### Low-Resolution Video:")
                    low_res_video_path = "LR_Videos/" + uploaded_file.name
                    st.video(low_res_video_path)

                with col2:
                    st.write("### Upscaled Video:")
                    st.video(upscaled_video_path)

                # Add a download button for the upscaled video
                with open(upscaled_video_path, "rb") as file:
                    btn = st.download_button(
                        label="Download Upscaled Video",
                        data=file,
                        file_name=os.path.basename(upscaled_video_path),
                        mime="video/mp4"
                    )
        else:
            st.write("Detected an image file. Processing will use the image upscaling pipeline.")
            
            # Save the uploaded image locally
            save_low_res_image(uploaded_file, "LR")
            
            if st.button("Upscale Image"):
                st.write("Processing on EC2 instance via SSH...")
                run_model_on_ec2(
                    ssh_host=SSH_HOST,
                    ssh_user=SSH_USER,
                    ssh_key_path=SSH_KEY_PATH,
                    local_image_path="LR/" + uploaded_file.name,
                    remote_image_path=f"/home/{SSH_USER}/gan-upscale/UpscalerModel/LR/" + uploaded_file.name,
                    remote_results_path="/home/{SSH_USER}/gan-upscale/UpscalerModel/results/",
                    local_results_path="results/"
                )
                st.success("Image upscaling completed! Check the results/ folder.")
                
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
                    # Headers in one row
                    col1_header, col2_header = st.columns(2)
                    with col1_header:
                        st.write("### Low-Resolution Image:")
                    
                    with col2_header:
                        st.write("### Upscaled Image:")
                    
                    # Images in another row
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(low_res_path, use_container_width=False)
                    
                    with col2:
                        st.image(upscaled_image_path, use_container_width=False)
                    
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