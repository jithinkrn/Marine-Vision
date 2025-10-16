import numpy as np

# from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import os
import io
from tensorflow.keras.saving import register_keras_serializable, load_model
from keras.config import enable_unsafe_deserialization
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Conv2D

enable_unsafe_deserialization()


# Utility functions to handle output shapes for pooling layers. This to be used by the model
@register_keras_serializable()
def output_shape_avg(input_shape):
    return input_shape[:-1] + (1,)


@register_keras_serializable()
def output_shape_max(input_shape):
    return input_shape[:-1] + (1,)


# Custom Layer to be used by the model to adjust contrast of the input
@register_keras_serializable()
class ContrastAdjustLayer(Layer):
    def __init__(self, contrast_factor=1.2, **kwargs):
        super(ContrastAdjustLayer, self).__init__(**kwargs)
        self.contrast_factor = contrast_factor

    def call(self, inputs):
        # Adjust contrast in a differentiable way (using multiplication)
        mean = tf.reduce_mean(
            inputs, axis=[1, 2], keepdims=True
        )  # Compute the mean pixel value
        adjusted = (inputs - mean) * self.contrast_factor + mean  # Adjust contrast
        return adjusted


# Custom layer to be used by the model to adjust brightness of the input
@register_keras_serializable()
class BrightnessAdjustLayer(Layer):
    def __init__(self, brightness_factor=0.1, **kwargs):
        super(BrightnessAdjustLayer, self).__init__(**kwargs)
        self.brightness_factor = brightness_factor

    def call(self, inputs):
        # Adjust brightness by adding a factor to the inputs
        adjusted = inputs + self.brightness_factor
        return adjusted


# Custom Layer to be used by the model for global_histogram equalization
@register_keras_serializable()
class GlobalHistEqLayer(Layer):
    def __init__(self, **kwargs):
        super(GlobalHistEqLayer, self).__init__(**kwargs)
        self.conv = Conv2D(3, (1, 1), padding="same")

    def build(self, input_shape):
        self.conv.build(input_shape)
        self._trainable_weights = self.conv.trainable_weights
        super(GlobalHistEqLayer, self).build(input_shape)

    def call(self, inputs):

        # Convert 32-channel input to 3-channel
        x = self.conv(inputs)

        # Apply histogram equalization to the luminance channel in YUV color space
        yuv = tf.image.rgb_to_yuv(x)
        y_channel = tf.image.per_image_standardization(
            yuv[..., 0:1]
        )  # Equalize the Y channel
        yuv = tf.concat([y_channel, yuv[..., 1:]], axis=-1)
        output = tf.image.yuv_to_rgb(yuv)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (3,)


# Custom Layer to be used by the model for Average Pooling custom
@register_keras_serializable()
class AvgPoolLayer(Layer):
    def call(self, inputs):
        return K.mean(inputs, axis=-1, keepdims=True)


# Max Pooling custom layer to be used by the model
@register_keras_serializable()
class MaxPoolLayer(Layer):
    def call(self, inputs):
        return K.max(inputs, axis=-1, keepdims=True)


# The SobelConvLayer is designed to apply a Sobel filter to the input, which is commonly used in edge detection to highlight gradients or edges in an image.
@register_keras_serializable()
class SobelConvLayer(Layer):
    def __init__(self, **kwargs):
        super(SobelConvLayer, self).__init__(**kwargs)
        self.sobel_kernel_x = tf.constant(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32
        )
        self.sobel_kernel_y = tf.constant(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32
        )
        self.epsilon = 1e-6  # Small constant to prevent NaNs

    def build(self, input_shape):
        input_channels = int(input_shape[-1])
        self.sobel_kernel_x = tf.tile(
            self.sobel_kernel_x[:, :, tf.newaxis, tf.newaxis], [1, 1, input_channels, 1]
        )
        self.sobel_kernel_y = tf.tile(
            self.sobel_kernel_y[:, :, tf.newaxis, tf.newaxis], [1, 1, input_channels, 1]
        )
        self.built = True

    def call(self, inputs):
        sobel_x = tf.nn.depthwise_conv2d(
            inputs, self.sobel_kernel_x, strides=[1, 1, 1, 1], padding="SAME"
        )
        sobel_y = tf.nn.depthwise_conv2d(
            inputs, self.sobel_kernel_y, strides=[1, 1, 1, 1], padding="SAME"
        )

        # Add epsilon to prevent sqrt of zero
        gradient_magnitude = tf.sqrt(
            tf.square(sobel_x) + tf.square(sobel_y) + self.epsilon
        )
        return gradient_magnitude


# Prepare a dictionary of all custom objects to pass during model loading
custom_objects = {
    "ContrastAdjustLayer": ContrastAdjustLayer,
    "BrightnessAdjustLayer": BrightnessAdjustLayer,
    "GlobalHistEqLayer": GlobalHistEqLayer,
    "AvgPoolLayer": AvgPoolLayer,
    "MaxPoolLayer": MaxPoolLayer,
    "SobelConvLayer": SobelConvLayer,
    "output_shape_avg": output_shape_avg,
    "output_shape_max": output_shape_max,
}

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute paths to the model files
model_path = os.path.join(current_dir, "models/lds_net_model.keras")

# Load the model globally when the application starts (for performance optimization)
lds_net_model = load_model(model_path, custom_objects=custom_objects)


def preprocess_image(image_file, target_size=(256, 256)):
    """Preprocess the image by loading, resizing, normalizing, and adding a batch dimension."""
    img = Image.open(image_file)
    img = img.resize(target_size)
    img_array = img_to_array(img)
    img_array = (img_array / 127.5) - 1.0  # Normalize to [-1, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


def predict_image(generator, img_array):
    """Generate the enhanced image using the provided model."""
    generated_img = generator.predict(img_array)
    generated_img = 0.5 * generated_img + 0.5  # Denormalize to [0, 1]
    generated_img = np.squeeze(generated_img)  # Remove batch dimension
    return generated_img


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
    img_array = preprocess_image(input_path)

    # Generate the enhanced image using the globally loaded model
    enhanced_image = predict_image(lds_net_model, img_array)

    # Convert the enhanced image array to an image and resize it back to the original size
    enhanced_img = Image.fromarray((enhanced_image * 255).astype(np.uint8))
    enhanced_img = enhanced_img.resize(original_size)

    # Save the resized enhanced image
    enhanced_img.save(output_path)
