import torch
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
import os
import numpy as np
from app.api.nets import seapixgan
#import pickle


# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the SeapixGAN model
model_path = os.path.join(current_dir, "models/best_fine_tuned_generator.pth")

# Define the model
model = seapixgan.SeapixGANGenerator()

# Load the model state dictionary (without weights_only=True)
model.load_state_dict(torch.load(model_path))

model.eval()

# # GPU check
is_cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor
if is_cuda:
    model.cuda()

def preprocess_image(image_file, target_size=(256, 256)):
    """Preprocess the image by resizing, normalizing, and converting to Tensor."""
    img = Image.open(image_file).convert("RGB")
    transform_list = [
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    transform = transforms.Compose(transform_list)
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor

def predict_image(generator, img_tensor):
    """Generate the enhanced image using SeapixGAN."""
    img_tensor = Variable(img_tensor).type(Tensor)
    with torch.no_grad():
        generated_img = generator(img_tensor)
    generated_img = (generated_img.squeeze(0) * 0.5) + 0.5  # Denormalize [-1, 1] -> [0, 1]
    return generated_img.cpu().detach().permute(1, 2, 0).numpy()

def save_image(image_array, output_path):
    """Save the enhanced image."""
    img = Image.fromarray((image_array * 255).astype(np.uint8))
    img.save(output_path)

def enhance_image(input_path, output_path, original_size=(1920, 1080)):
    """
    Enhance the image using SeapixGAN and resize to the original size.
    """
    # Preprocess the input image
    img_tensor = preprocess_image(input_path)

    # Generate the enhanced image
    enhanced_image_np = predict_image(model, img_tensor)

    # Convert to image and resize to the original dimensions
    enhanced_img = Image.fromarray((enhanced_image_np * 255).astype(np.uint8))
#   enhanced_img = enhanced_img.resize(original_size, Image.LANCZOS)
    enhanced_img = enhanced_img.resize(original_size)

    # Save the image
    enhanced_img.save(output_path)