import os
import cv2

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

if __name__ == "__main__":
    augment_images()
