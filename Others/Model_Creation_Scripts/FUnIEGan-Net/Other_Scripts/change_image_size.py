import os
from PIL import Image

# Define the paths
input_folder = "/home/ashwin/Project/DATASETS/Bx_Data/Paired/underwater_imagenet/reduced_size_test/trainB"
output_folder = "/home/ashwin/Project/DATASETS/Bx_Data/Paired/underwater_imagenet/reduced_size_test/trainB"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Set the desired dimensions
width, height = 600, 400

# Loop over all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff")):
        # Open an image file
        img = Image.open(os.path.join(input_folder, filename))

        # Resize the image
        img = img.resize((width, height), Image.Resampling.LANCZOS)

        # Save the resized image to the output folder
        img.save(os.path.join(output_folder, filename))

print("All images have been resized and saved.")
