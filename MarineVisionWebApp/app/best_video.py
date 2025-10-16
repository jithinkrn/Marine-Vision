from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2
import cv2
import numpy as np
import os
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.models import Model
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model


def extract_frames(video_path, target_size=(128, 128)):
    """Extract frames from the video and resize them to a consistent shape."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop if no more frames

        # Resize the frame to the target size
        frame = cv2.resize(frame, target_size)
        frames.append(frame)

    cap.release()
    return np.array(frames)  # Return as a NumPy array


# Calculate reconstruction error
def calculate_reconstruction_error(autoencoder, frames):
    reconstructions = autoencoder.predict(frames)
    errors = np.mean(np.square(frames - reconstructions), axis=(1, 2, 3))
    return np.median(errors)  # Use median to handle outliers


# Calculate SSIM between original and reconstructed frames
def calculate_ssim(frames, reconstructions):
    # Explicitly set win_size and data_range for SSIM calculation
    ssim_scores = [
        ssim(f, r, win_size=11, channel_axis=-1, data_range=1.0)
        for f, r in zip(frames, reconstructions)
    ]
    return np.mean(ssim_scores)  # Higher SSIM indicates better similarity


# Extract latent space features from encoder
def extract_latent_space(autoencoder, frames):
    # Extract latent space using the correct layer index
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.layers[4].output)
    return encoder.predict(frames)


def cluster_latent_space(latent_vectors):
    # Get the number of samples
    num_samples = latent_vectors.shape[0]

    # Ensure the number of clusters does not exceed the number of samples
    n_clusters = min(5, num_samples)  # Use 5 clusters or fewer if samples are limited

    if n_clusters < 2:
        print("Not enough samples for clustering. Returning inertia as 0.")
        return 0  # Return 0 if clustering is not possible

    # Perform clustering with the adjusted cluster count
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(
        latent_vectors.reshape(len(latent_vectors), -1)
    )
    return kmeans.inertia_


def add_perturbation(value, epsilon=1e-6):
    return value + np.random.uniform(-epsilon, epsilon)


def compute_composite_score(reconstruction_error, ssim_score, clustering_inertia):
    # Add small noise to avoid identical values
    reconstruction_error = add_perturbation(reconstruction_error)
    ssim_score = add_perturbation(ssim_score)
    clustering_inertia = add_perturbation(clustering_inertia)

    metrics = np.array(
        [
            [reconstruction_error],
            [-ssim_score],  # Negate SSIM as higher is better
            [clustering_inertia],
        ]
    )

    scaler = MinMaxScaler()
    normalized_metrics = scaler.fit_transform(metrics).flatten()

    print(f"Normalized Metrics: {normalized_metrics}")  # Debugging output

    return np.sum(normalized_metrics)


# Create overlapping sequences from frames
def create_overlapping_sequences(frames, seq_length=10, stride=5):
    sequences = [
        frames[i : i + seq_length]
        for i in range(0, len(frames) - seq_length + 1, stride)
    ]
    return np.array(sequences)


# Helper function to extract video paths from a directory
def get_video_paths(directory):
    """Retrieve paths of all video files from a given directory."""
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]


model_path = os.path.join(os.path.dirname(__file__), "model", "best_autoencoder.keras")


# Load the trained model
def load_trained_model(model_path):
    try:
        model = load_model(model_path)
        print(f"Model loaded successfully from {model_path}.")
        return model
    except Exception as e:
        print(f"Error loading the model: {e}")
        return None


autoencoder = load_trained_model(model_path)


# Function to evaluate video and return its composite score
def evaluate_video(video_path, input_shape):

    frames = extract_frames(video_path) / 255.0

    if frames.shape[0] < input_shape[0]:
        print(f"Warning: Not enough frames in {video_path}. Skipping.")
        return None

    # Create sequences of frames
    sequences = [
        frames[i : i + input_shape[0]]
        for i in range(0, frames.shape[0] - input_shape[0] + 1, input_shape[0])
    ]
    sequences = np.array(sequences)

    # Predict reconstructed sequences
    reconstructions = autoencoder.predict(sequences)

    # Calculate metrics
    recon_error = np.mean(
        np.square(sequences - reconstructions), axis=(1, 2, 3, 4)
    ).mean()
    ssim_score = np.mean(
        [calculate_ssim(seq, recon) for seq, recon in zip(sequences, reconstructions)]
    )
    latent_vectors = extract_latent_space(autoencoder, sequences)
    inertia = cluster_latent_space(latent_vectors)

    return compute_composite_score(recon_error, ssim_score, inertia)


def evaluate_output_videos(video_paths):
    input_shape = (10, 128, 128, 3)
    video_scores = []
    for path in video_paths:
        score = evaluate_video(path, input_shape)
        if score is not None:
            # Extract model name based on the new naming convention
            filename = os.path.basename(path)
            model_name = filename.split("_")[
                0
            ]  # Extract part before the first underscore
            print(
                f"Extracted Model Name: {model_name}"
            )  # Print the extracted model name

            video_scores.append((model_name, score))

    # Rank videos by composite score
    ranked_videos = sorted(video_scores, key=lambda x: x[1])
    for video, score in ranked_videos:
        print(f"Video: {video}, Composite Score: {score}")
    return video_scores
