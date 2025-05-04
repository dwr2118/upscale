import os
import cv2
import argparse
from moviepy import VideoFileClip, ImageSequenceClip, AudioFileClip

def destruct_video(video_path):
    base_name = os.path.basename(video_path)
    title, _ = os.path.splitext(base_name)

    frames_dir = os.path.join("destructed", f"{title}", f"{title}_frames")
    os.makedirs(frames_dir, exist_ok=True)

    clip = VideoFileClip(video_path)

    audio_path = os.path.join("destructed", f"{title}", f"{title}_audio.mp3")
    clip.audio.write_audiofile(audio_path)

    vidcap = cv2.VideoCapture(video_path)
    frame_count = 0
    success, image = vidcap.read()
    while success:
        frame_filename = os.path.join(frames_dir, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, image)
        success, image = vidcap.read()
        frame_count += 1

    print(f"Done! Extracted {frame_count} frames and saved audio to {audio_path}")

def destruct_all_videos_in_directory(directory):
    supported_formats = ('.mp4', '.avi', '.mov', '.mkv')
    for filename in os.listdir(directory):
        if filename.lower().endswith(supported_formats):
            video_path = os.path.join(directory, filename)
            print(f"Processing {video_path}...")
            destruct_video(video_path)

def reconstruct_video(base_dir, desired_fps=30):
    title = os.path.basename(base_dir)
    frames_dir = os.path.join(base_dir, f"{title}_frames")
    audio_path = os.path.join(base_dir, f"{title}_audio.mp3")
    
    if not os.path.exists(frames_dir):
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Include both .jpg and .png files
    frame_files = sorted([
        os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".png"))
    ])
    if not frame_files:
        raise ValueError(f"No frames found in {frames_dir}")

    first_frame = cv2.imread(frame_files[0])
    height, width, _ = first_frame.shape
    fps = desired_fps  # Change if you know the original FPS (You might have to do math to figure it out based on original)

    clip = ImageSequenceClip(frame_files, fps=fps)
    clip = clip.with_audio(AudioFileClip(audio_path))
    
    reconstruction_dir = "reconstructed"
    os.makedirs(reconstruction_dir, exist_ok=True)
    
    output_filename = os.path.join(reconstruction_dir, f"reconstructed_{title}.mp4")
    clip.write_videofile(output_filename, codec="libx264", audio_codec="aac")

    print(f"Reconstructed video saved as {output_filename}")

def reconstruct_all_videos_in_directory(parent_dir="destructed"):
    if not os.path.exists(parent_dir):
        raise FileNotFoundError(f"Directory not found: {parent_dir}")
    
    for title in os.listdir(parent_dir):
        base_dir = os.path.join(parent_dir, title)
        if os.path.isdir(base_dir):
            print(f"Reconstructing video from {base_dir}...")
            try:
                reconstruct_video(base_dir)
            except (FileNotFoundError, ValueError) as e:
                print(f"Skipping {title}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Process videos for deconstruction or reconstruction.")
    parser.add_argument("-d", "--destruct", type=str, help="Directory containing videos to destruct.")
    parser.add_argument("-r", "--reconstruct", type=str, help="Directory containing destructed content to reconstruct.")

    args = parser.parse_args()

    if args.destruct and args.reconstruct:
        print("Error: Please provide only one of -d (destruct) or -r (reconstruct) at a time.")
        return

    if args.destruct:
        if not os.path.exists(args.destruct):
            print(f"Error: Directory not found: {args.destruct}")
            return
        print(f"Destructing videos in directory: {args.destruct}")
        destruct_all_videos_in_directory(args.destruct)

    elif args.reconstruct:
        if not os.path.exists(args.reconstruct):
            print(f"Error: Directory not found: {args.reconstruct}")
            return
        print(f"Reconstructing videos from directory: {args.reconstruct}")
        reconstruct_all_videos_in_directory(args.reconstruct)

    else:
        print("Error: Please provide either -d (destruct) or -r (reconstruct).")

if __name__ == "__main__":
    main()