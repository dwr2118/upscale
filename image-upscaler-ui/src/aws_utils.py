def run_model_on_ec2(ssh_host, ssh_user, ssh_key_path, local_image_path, remote_image_path, remote_results_path, local_results_path):
    import paramiko
    import os
    
    # Extract the base name of the uploaded image
    image_name = os.path.basename(local_image_path).split('.')[0]  # Get the name without extension

    # Connect to EC2 instance
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ssh_host, username=ssh_user, key_filename=ssh_key_path)

    # Transfer the image to the EC2 instance
    sftp = ssh.open_sftp()
    sftp.put(local_image_path, remote_image_path)
    sftp.close()

    # Run the model on the EC2 instance
    command = f"cd /home/{ssh_user}/final/ESRGAN && python test.py"  # Navigate to the model directory and run the script
    stdin, stdout, stderr = ssh.exec_command(command)
    print(stdout.read().decode())
    print(stderr.read().decode())

    # Download the results back to the local machine
    sftp = ssh.open_sftp()
    for file in sftp.listdir(remote_results_path):
        if file.startswith(image_name):  # Only download files that start with the uploaded image's name
            sftp.get(os.path.join(remote_results_path, file), os.path.join(local_results_path, file))
    sftp.close()

    ssh.close()