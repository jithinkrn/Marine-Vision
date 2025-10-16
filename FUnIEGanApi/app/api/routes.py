from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse
import shutil
import os
from app.api.funiegan_predict import (
    enhance_image,
)  # Import the FUnIE-GAN enhancement function
import cv2  # OpenCV for video processing
import uuid
import subprocess

router = APIRouter()

UPLOAD_DIR = "uploaded_videos"
PROCESSED_DIR = "processed_videos"
FRAMES_DIR = "video_frames"

# Ensure the directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)


def delete_all_items_in_directory(directory):
    """Delete all files and subdirectories in the specified directory."""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


def extract_frames(video_path, frames_dir):
    """Extract frames from the video."""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_path = os.path.join(frames_dir, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    cap.release()


def compile_video(frames_dir, output_video_path, original_video_path):
    """Compile frames back into a video, preserving frame rate."""
    cap = cv2.VideoCapture(original_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Actual Code
    # frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # Modified for resizing
    frame_size = (640, 360)
    cap.release()

    out = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size
    )

    # Read frames from frames_dir
    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    for frame in frames:
        frame_path = os.path.join(frames_dir, frame)
        img = cv2.imread(frame_path)
        # height, width, channels = img.shape
        # cropped_image = img[:, left_crop_value : width - right_crop_value]
        # img = img[top_crop_value:height - bottom_crop_value, :]
        out.write(img)
    out.release()


def convert_to_h264_high_profile(input_video_path):
    """Convert video to H.264 High Profile using FFmpeg and replace the input file."""
    temp_output_video_path = input_video_path.replace(".mp4", "_temp.mp4")

    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_video_path,
        "-c:v",
        "libx264",
        "-profile:v",
        "high",
        "-preset",
        "slow",
        "-b:v",
        "3000k",
        temp_output_video_path,
    ]
    subprocess.run(command, check=True)

    if os.path.exists(input_video_path):
        os.remove(input_video_path)
    os.rename(temp_output_video_path, input_video_path)


@router.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    # Generate a unique identifier for this video
    video_id = str(uuid.uuid4())

    # Clear previous files
    delete_all_items_in_directory(UPLOAD_DIR)
    delete_all_items_in_directory(PROCESSED_DIR)
    delete_all_items_in_directory(FRAMES_DIR)

    # Save the uploaded video
    video_path = os.path.join(UPLOAD_DIR, f"{video_id}_{file.filename}")
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract frames from the video
    frames_output_dir = os.path.join(FRAMES_DIR, video_id)
    os.makedirs(frames_output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    # Actual code
    # original_size = (
    #     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    #     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    # )

    # Modified for resizing
    original_size = (
        640,
        360,
    )
    cap.release()

    extract_frames(video_path, frames_output_dir)

    # Enhance each frame using the FUnIE-GAN model
    processed_frames_dir = os.path.join(FRAMES_DIR, f"processed_{video_id}")
    os.makedirs(processed_frames_dir, exist_ok=True)

    for frame_file in os.listdir(frames_output_dir):
        frame_path = os.path.join(frames_output_dir, frame_file)
        processed_frame_path = os.path.join(processed_frames_dir, frame_file)

        enhance_image(
            input_path=frame_path,
            output_path=processed_frame_path,
            original_size=original_size,
        )

    processed_video_path = os.path.join(PROCESSED_DIR, f"processed_{file.filename}")
    compile_video(processed_frames_dir, processed_video_path, video_path)
    convert_to_h264_high_profile(processed_video_path)

    # Return the processed video as a response
    return FileResponse(processed_video_path, media_type="video/mp4")
