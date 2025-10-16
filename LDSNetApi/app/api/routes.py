from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse
import shutil
import os
from app.api.ldsNetPredict import enhance_image
from fastapi.responses import StreamingResponse
import cv2  # OpenCV for video processing
import uuid
import shutil
from moviepy.editor import ImageSequenceClip

router = APIRouter()

UPLOAD_DIR = "uploaded_videos"
PROCESSED_DIR = "processed_videos"
FRAMES_DIR = "video_frames"

# Ensure the upload and processed directories exist when the application starts
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

# Additional check for processed_videos directory (if needed)
if not os.path.exists(PROCESSED_DIR):
    os.makedirs(PROCESSED_DIR)

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

if not os.path.exists(FRAMES_DIR):
    os.makedirs(FRAMES_DIR)


def delete_all_items_in_directory(directory):
    """
    Delete all files and subdirectories in the specified directory.
    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Delete the file or symbolic link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Delete the directory and its contents
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


def extract_frames(video_path, frames_dir):
    """
    Extract frames from the video and save them in frames_dir.
    """
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
    """
    Compile frames back into a video using moviepy, preserving the frame rate of the original video.
    """
    # Extract the original video's FPS and frame size using OpenCV
    cap = cv2.VideoCapture(original_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second from original video
    cap.release()

    # Get the list of frame files (JPEG images) sorted by name
    frames = sorted(
        [
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir)
            if f.endswith(".jpg")
        ]
    )

    # Create the video using moviepy's ImageSequenceClip
    clip = ImageSequenceClip(frames, fps=fps)

    # Write the video to the output path
    clip.write_videofile(output_video_path, codec="libx264", fps=fps)

    print(f"Video compiled successfully: {output_video_path}")


@router.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    # Generate a unique identifier for this video
    video_id = str(uuid.uuid4())

    # Clear previous files from directories
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

    # Open video to get the frame size
    cap = cv2.VideoCapture(video_path)
    original_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()

    extract_frames(video_path, frames_output_dir)

    # Enhance each frame using the LDS-Net model
    processed_frames_dir = os.path.join(FRAMES_DIR, f"processed_{video_id}")
    os.makedirs(processed_frames_dir, exist_ok=True)

    for frame_file in os.listdir(frames_output_dir):
        frame_path = os.path.join(frames_output_dir, frame_file)
        processed_frame_path = os.path.join(processed_frames_dir, frame_file)

        # Pass the original frame size to ensure the enhanced image is resized correctly
        enhance_image(
            input_path=frame_path,
            output_path=processed_frame_path,
            original_size=original_size,
        )

    # Compile the enhanced frames back into a video
    processed_video_path = os.path.join(PROCESSED_DIR, f"processed_{file.filename}")
    compile_video(processed_frames_dir, processed_video_path, video_path)

    # Return the processed video as a response
    return FileResponse(processed_video_path, media_type="video/mp4")
