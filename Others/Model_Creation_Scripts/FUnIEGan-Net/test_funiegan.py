import os
import time
import argparse
import numpy as np
from PIL import Image
from glob import glob
from os.path import join, exists
import cv2  # OpenCV for video processing

# pytorch libs
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms

# options
parser = argparse.ArgumentParser()
parser.add_argument("--video_path", type=str, default="/path/to/input_video.mp4", help="Path to input video file")
parser.add_argument("--save_dir", type=str, default="./output_video", help="Directory to save the processed video")
parser.add_argument("--model_name", type=str, default="funiegan", help="Model name (funiegan or ugan)")
parser.add_argument(
    "--model_path",
    type=str,
    default="/home/ashwin/Project/FunieGAN/FUnIE-GAN/PyTorch/checkpoints/FunieGAN/EUVP/generator_45.pth",
    help="Path to the pre-trained model",
)
opt = parser.parse_args()

# checks
assert exists(opt.model_path), "Model not found"
os.makedirs(opt.save_dir, exist_ok=True)
is_cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor

# model architecture
if opt.model_name.lower() == "funiegan":
    from nets import funiegan

    model = funiegan.GeneratorFunieGAN()
elif opt.model_name.lower() == "ugan":
    from nets.ugan import UGAN_Nets

    model = UGAN_Nets(base_model="pix2pix").netG
else:
    raise ValueError("Unknown model name. Supported: 'funiegan', 'ugan'")

# load model weights
model.load_state_dict(torch.load(opt.model_path))
if is_cuda:
    model.cuda()
model.eval()
print(f"Loaded model from {opt.model_path}")

# transformation settings
img_width, img_height, channels = 1920, 1088, 3  # Original size with padding to 1088
transforms_ = [
    transforms.Pad((0, 16), padding_mode="reflect"),  # Adjust padding as needed
    transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
transform = transforms.Compose(transforms_)

# Load video using OpenCV
cap = cv2.VideoCapture(opt.video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get frames per second of the video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

assert frame_width == 1920 and frame_height == 1080, "Video frame size must be 1920x1080"

# Define the video writer for the final output
output_video_path = join(opt.save_dir, "output_video.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (1920, 1080))

# testing loop
times = []
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop when the video ends

    frame_idx += 1
    print(f"Processing frame {frame_idx}/{total_frames}")

    # Convert frame to PIL image and apply the transformation
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inp_img = transform(pil_img)
    inp_img = Variable(inp_img).type(Tensor).unsqueeze(0)

    # Generate enhanced image
    start_time = time.time()
    with torch.no_grad():
        gen_img = model(inp_img)
    times.append(time.time() - start_time)

    # Convert generated image back to numpy and resize to 1920x1080
    gen_img = gen_img.squeeze(0).cpu().detach()
    # Denormalize the image (reverse the normalization applied during transformation)
    gen_img = (gen_img * 0.5) + 0.5  # This will bring values from [-1, 1] to [0, 1]
    # Convert to numpy array
    gen_img_np = gen_img.permute(1, 2, 0).numpy()  # C x H x W -> H x W x C
    gen_img_np = (gen_img_np * 255).clip(0, 255).astype(np.uint8)
    # Ensure the image is in RGB format
    gen_img_np = cv2.cvtColor(gen_img_np, cv2.COLOR_RGB2BGR)

    final_frame = gen_img_np[:, 100:, :]  # Take only the top 1080 pixels
    # Ensure the generated frame is correctly resized to 1920x1080 (or adjust the model output)
    final_frame = cv2.resize(final_frame, (640, 480))  # Resize to the target dimensions

    # Write the processed frame to the output video
    out.write(final_frame)

    # Optionally show the frame during processing
    cv2.imshow("Processed Frame", final_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Runtime statistics
if len(times) > 1:
    print(f"\nTotal frames processed: {len(times)}")
    total_time, mean_time = np.sum(times[1:]), np.mean(times[1:])
    print(f"Time taken: {total_time:.2f} sec at {1.0 / mean_time:.3f} fps")
    print(f"Saved final video at {output_video_path}")
