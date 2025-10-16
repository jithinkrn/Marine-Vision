import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model as tf_load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import os
import io
from tensorflow.keras.models import model_from_json
import cv2
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Configure CPU optimization
NUM_CORES = multiprocessing.cpu_count()
tf.config.threading.set_intra_op_parallelism_threads(NUM_CORES)
tf.config.threading.set_inter_op_parallelism_threads(NUM_CORES)

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct absolute paths to the model files
# model_json_path = os.path.join(current_dir, "models/deep_sesr_2x_1d.json")
# model_weights_path = os.path.join(current_dir, "models/deep_sesr_2x_1d.h5")

'''
# Load the model architecture from the JSON file
with open(json_path, "r") as json_file:
    model_json = json_file.read()
    model = model_from_json(model_json)

# Load the model weights from the .h5 file
model.load_weights(weights_path)
'''


@tf.keras.utils.register_keras_serializable()
class PixelShuffle(tf.keras.layers.Layer):
    def __init__(self, scale, **kwargs):
        super(PixelShuffle, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return tf.nn.depth_to_space(inputs, self.scale)

    def get_config(self):
        config = super(PixelShuffle, self).get_config()
        config.update({"scale": self.scale})
        return config


# Model parameters
# MODEL_HR_SIZE = (720, 1280)
MODEL_HR_SIZE = (360, 640)  # height, width for model's input size
MODEL_LR_SIZE = (360, 640)  # height, width for 2x super-resolution


# def load_model(model_json_path, model_weights_path):
def load_deep_sesr_model(model_path):
    """Load the model with custom objects"""
    custom_objects = {'PixelShuffle': PixelShuffle}
    with ThreadPoolExecutor(max_workers=NUM_CORES) as executor:
        model = tf_load_model(model_path, custom_objects=custom_objects)
    return model


# Load model globally
model_path = os.path.join(current_dir, "models/deep_sesr_640x360_final.keras")
model = load_deep_sesr_model(model_path)


def preprocess_image(image_file, target_size=(640, 360)):
    """Preprocess the image by loading, resizing, normalizing, and adding a batch dimension."""

    """Parallel preprocessing"""
    def resize_image(img):
        return cv2.resize(img, target_size)

    def convert_color(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with ThreadPoolExecutor(max_workers=NUM_CORES) as executor:
        if isinstance(image_file, np.ndarray):
            # Parallel processing for numpy array input
            resized = executor.submit(resize_image, image_file).result()
            img = executor.submit(convert_color, resized).result()
        else:
            # Parallel processing for file input
            img = Image.open(image_file)
            img = img.resize(target_size)
            img = np.array(img)

        # Parallel preprocessing
        img = executor.submit(lambda x: x.astype(np.float32), img).result()
        img = executor.submit(lambda x: (x / 127.5) - 1.0, img).result()
        return np.expand_dims(img, axis=0)


def predict_image(generator, img_array):
    """Generate the enhanced image using the provided model."""
    # generated_img = generator.predict(img_array)
    # generated_img = 0.5 * generated_img + 0.5  # Denormalize to [0, 1]
    # generated_img = np.clip((generated_img + 1.0) *
    #                        127.5, 0, 255).astype(np.uint8)

    # generated_img = np.squeeze(generated_img)  # Remove batch dimension

    # Predict
    # gen_op = generator.predict(img_array)
    # Predict in parallel
    gen_op = generator.predict(img_array)
    gen_hr = gen_op[1]  # We're interested in the HR output

    # Postprocess
    # Remove batch dimension and deprocess
    gen_hr = np.clip((gen_hr[0] + 1.0) * 127.5, 0, 255).astype(np.uint8)

    return gen_hr


def save_image(image_array, output_path):
    """Save the generated image to the specified path."""
    img = Image.fromarray((image_array * 255).astype(np.uint8))
    img.save(output_path)


def enhance_image(input_path, output_path, original_size):
    """
    Enhances the image by passing it through the model and resizes
    the output to the original size.
    """
    # Preprocess the input image (resize to model's input size)
    try:
        img_array = preprocess_image(input_path)

    # Generate the enhanced image using the globally loaded model
        enhanced_image = predict_image(model, img_array)

    # Convert the enhanced image array to an image and resize it back to the original size
    # enhanced_img = Image.fromarray((enhanced_image * 255).astype(np.uint8))
    # Since the predict_image now returns uint8 values, we don't need to multiply by 255
        enhanced_img = cv2.resize(
            enhanced_image, (original_size))
    # Convert color space if needed
        enhanced_image_bgr = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)

    # Save the result
        cv2.imwrite(output_path, enhanced_image_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        return True
    except Exception as e:
        print(f"Error in enhance_image: {str(e)}")
        return False
