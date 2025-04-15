def save_low_res_image(uploaded_file, lr_folder):
    import os
    from PIL import Image

    if not os.path.exists(lr_folder):
        os.makedirs(lr_folder)

    file_path = os.path.join(lr_folder, uploaded_file.name)
    image = Image.open(uploaded_file)
    image.save(file_path)

    return file_path

def call_local_model(lr_folder, output_folder):
    import subprocess

    command = ["python", "test.py", "--input", lr_folder, "--output", output_folder]
    subprocess.run(command, check=True)