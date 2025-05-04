import time


def run_model_on_ec2(ssh_host, ssh_user, ssh_key_path, local_image_path, remote_image_path, remote_results_path, local_results_path):
    import paramiko
    import os
    
    # Extract the base name of the uploaded image
    image_name = os.path.basename(local_image_path).split('.')[0]  # Get the name without extension

    # Ensure the local results directory exists
    if not os.path.exists(local_results_path):
        os.makedirs(local_results_path)

    # Connect to EC2 instance
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ssh_host, username=ssh_user, key_filename=ssh_key_path)

    # Transfer the image to the EC2 instance
    sftp = ssh.open_sftp()
    sftp.put(local_image_path, remote_image_path)
    sftp.close()

    # Run the model on the EC2 instance
    command = f"cd /home/{ssh_user}/final/gan-upscale/UpscalerModel && python upscaler_test.py"  # Navigate to the model directory and run the script
    stdin, stdout, stderr = ssh.exec_command(command)
    print(stdout.read().decode())
    print(stderr.read().decode())

    # Download the results back to the local machine
    sftp = ssh.open_sftp()
    for file in sftp.listdir(remote_results_path):
        if file.startswith(image_name):  # Only download files that start with the uploaded image's name
            sftp.get(os.path.join(remote_results_path, file), os.path.join(local_results_path, file))
    sftp.close()
    
    # Clean up the remote directories
    # cleanup_command = f"rm -rf {remote_image_path} {remote_results_path}/*"
    # ssh.exec_command(cleanup_command)

    ssh.close()

def run_video_model_on_ec2(ssh_host, ssh_user, ssh_key_path, local_video_path, remote_video_path, remote_results_path, local_results_path):
    import paramiko
    import os

    # Ensure the local results directory exists
    if not os.path.exists(local_results_path):
        os.makedirs(local_results_path)

    # Connect to EC2 instance
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ssh_host, username=ssh_user, key_filename=ssh_key_path)

    # Transfer the video to the EC2 instance
    sftp = ssh.open_sftp()
    sftp.put(local_video_path, remote_video_path)
    sftp.close()
    
    # deconstruct the video frame by frame first 
    # Remove the video name from the remote_video_path to get just the directory
    remote_video_dir = os.path.dirname(remote_video_path)
    command = f"cd /home/{ssh_user}/final/gan-upscale/VideoProcessing && python VideoDataSetProcessing.py -d {remote_video_dir}"
    stdin, stdout, stderr = ssh.exec_command(command)
    print(stdout.read().decode())
    print(stderr.read().decode())
    
    # # Extract the video name without extension from the path
    video_name = os.path.basename(remote_video_path).split('.')[0]
    location_of_frames = f"/home/{ssh_user}/final/gan-upscale/VideoProcessing/destructed/{video_name}/{video_name}_frames/"
    location_of_HR_frames = f"/home/{ssh_user}/final/gan-upscale/VideoProcessing/destructed/{video_name}/HR_{video_name}_frames/"
    
    print("Starting upscaling process...")
    # run the upscaling model on the video frames 
    command = f"cd /home/{ssh_user}/final/gan-upscale/UpscalerModel && python upscaler_test.py --input {location_of_frames} --output {location_of_HR_frames}"
    stdin, stdout, stderr = ssh.exec_command(command)
    print(stdout.read().decode())
    print(stderr.read().decode())
    
    # Wait until the number of files in the HR frames directory matches the original frames directory
    while True:
        stdin, stdout, stderr = ssh.exec_command(f"ls -1 {location_of_frames} | wc -l")
        num_original_frames = int(stdout.read().decode().strip())

        stdin, stdout, stderr = ssh.exec_command(f"ls -1 {location_of_HR_frames} | wc -l")
        num_hr_frames = int(stdout.read().decode().strip())

        if num_original_frames == num_hr_frames:
            break

        time.sleep(5)  # Wait for 5 seconds before checking again
    
    # move all of the content in the old directory to the new directory
    command = f"mv {location_of_HR_frames} {location_of_frames}"
    
    # Reconstruct the video from the upscaled frames
    command = f"cd /home/{ssh_user}/final/gan-upscale/VideoProcessing && python VideoDataSetProcessing.py -r destructed/"
    stdin, stdout, stderr = ssh.exec_command(command)
    print(stdout.read().decode())
    print(stderr.read().decode())
    
    # # Wait until the reconstructed video appears in the remote results directory
    while True:
        print("Waiting for the reconstructed video...")
        stdin, stdout, stderr = ssh.exec_command(f"cd /home/{ssh_user}/final/gan-upscale/VideoProcessing && ls -1 reconstructed/ | wc -l")
        num_files_in_results = int(stdout.read().decode().strip())
        print(num_files_in_results)

        if num_files_in_results > 0:
            break

        time.sleep(10)  # Wait for 10 seconds before checking again
    
    # Move the upscaled video to the results directory
    command = f"mv /home/{ssh_user}/final/gan-upscale/VideoProcessing/reconstructed/reconstructed_{video_name}.mp4 {remote_results_path}/"
    stdin, stdout, stderr = ssh.exec_command(command)
    print(stdout.read().decode())
    print(stderr.read().decode())
    
    # Download the results back to the local machine
    sftp = ssh.open_sftp()
    for file in sftp.listdir(remote_results_path):
        sftp.get(os.path.join(remote_results_path, file), os.path.join(local_results_path, file))
    sftp.close()
    
    # Clean up the remote directories
    cleanup_command = f"rm -rf {remote_video_path} {remote_results_path}/*"
    ssh.exec_command(cleanup_command)

    ssh.close()