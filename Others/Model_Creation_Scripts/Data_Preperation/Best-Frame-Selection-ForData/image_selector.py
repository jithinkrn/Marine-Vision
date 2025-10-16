import os
import shutil
import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Path to your image folders
trainA_ICM_folder = "trainA_ICM"
trainB_ICM_folder = "trainB_ICM"
trainA_RGHS_folder = "trainA_RGHS"
trainA_UCM_folder = "trainA_UCM"
trainB_selected_folder = "trainB_selected"
trainB_final_folder = "trainB_final_4"
trainA_RGHS_renamed_folder = "trainA_RGHS_renamed"

# Load the ResNet50 model without the top layer (used for classification)
resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# Function to preprocess and extract ResNet features
def extract_resnet_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = resnet_model.predict(img_array)
    return features.flatten()

# Function to check if an image is predominantly yellow
def is_yellow_dominant(img, yellow_threshold=0.2):
    # Convert image to HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for yellow
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    
    # Create a mask for yellow pixels
    yellow_mask = cv2.inRange(hsv_img, yellow_lower, yellow_upper)

    # Calculate the percentage of yellow pixels
    yellow_pixels = np.sum(yellow_mask > 0)
    total_pixels = img.shape[0] * img.shape[1]

    yellow_percentage = yellow_pixels / total_pixels
    return yellow_percentage > yellow_threshold

# Custom function to extract sharpness, contrast, and brightness features
def extract_quality_features(img_path):
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate sharpness using Sobel filter
    sharpness = np.max(cv2.Sobel(gray_img, cv2.CV_64F, 1, 1))
    
    # Calculate contrast
    contrast = np.std(gray_img)
    
    # Calculate brightness
    brightness = np.mean(gray_img)
    
    # Check for unwanted colors (yellow)
    if is_yellow_dominant(img):
        return sharpness, contrast, brightness, 100  # Return high unwanted percentage for yellow dominant images
    
    return sharpness, contrast, brightness, 0  # No unwanted colors detected

# Function to calculate quality score
def calculate_quality_score(sharpness, contrast, brightness, sharpness_weight=1, contrast_weight=0.5, brightness_weight=0.2):
    return (sharpness_weight * sharpness) + (contrast_weight * contrast) + (brightness_weight * brightness)

# Function to perform grid search for DBSCAN parameters
def grid_search_dbscan(features, eps_values, min_samples_values):
    best_eps = None
    best_min_samples = None
    best_score = -1

    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(features)

            # Ignore noise
            if len(set(cluster_labels)) > 1:  # At least two clusters
                score = silhouette_score(features, cluster_labels)
                if score > best_score:
                    best_score = score
                    best_eps = eps
                    best_min_samples = min_samples

    return best_eps, best_min_samples

# Function to select the best image from different folders for the same frame
def select_best_image(frame):
    images = {
        "ICM_A": f"{trainA_ICM_folder}/{frame}_ICM.jpg",
        "ICM_B": f"{trainB_ICM_folder}/{frame}_ICM.jpg",
        "RGHS_A": f"{trainA_RGHS_renamed_folder}/{frame}_RGHS.jpg",
        "UCM_A": f"{trainA_UCM_folder}/{frame}_UCM.jpg",
        "Selected_B": f"{trainB_selected_folder}/{frame}.jpeg"
    }
    
    valid_images = []
    features = []
    all_scores = []
    
    for label, img_path in images.items():
        if os.path.exists(img_path):
            sharpness, contrast, brightness, unwanted_percentage = extract_quality_features(img_path)
            if unwanted_percentage > 0:  # Skip if unwanted percentage is high
                continue
            
            resnet_features = extract_resnet_features(img_path)
            combined_features = np.concatenate(([sharpness, contrast, brightness], resnet_features))
            features.append(combined_features)
            valid_images.append((label, img_path))
            
            score = calculate_quality_score(sharpness, contrast, brightness)
            all_scores.append((label, img_path, score))
        else:
            all_scores.append((label, "Missing", 0))
    
    if not valid_images:
        return None, all_scores
    
    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=2)
    cluster_labels = dbscan.fit_predict(normalized_features)
    
    # Select the best image from the largest cluster
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    largest_cluster = unique_labels[np.argmax(counts)]
    
    best_score = -np.inf
    best_image = None
    
    for i, (label, img_path) in enumerate(valid_images):
        if cluster_labels[i] == largest_cluster:
            score = next(score for l, _, score in all_scores if l == label)
            if score > best_score:
                best_score = score
                best_image = img_path
    
    return best_image, all_scores

# Main function to process all frames
def process_all_frames():
    frame = 1
    all_scores = []
    
    while True:
        # Select best image for the current frame
        best_image, frame_scores = select_best_image(frame)
        if not best_image and all(score == 0 for _, _, score in frame_scores):
            break
        
        if best_image:
            print(f"Best image for frame {frame}: {best_image}")
            
            # Move the best image to trainB_final folder
            best_image_name = os.path.basename(best_image)
            final_image_path = os.path.join(trainB_final_folder, best_image_name)
            shutil.copy(best_image, final_image_path)
        else:
            print(f"No valid images found for frame {frame}")
        
        # Append scores for all images in this frame
        for label, img_path, score in frame_scores:
            all_scores.append({"Frame": frame, "Image": label, "Path": img_path, "Score": score})
        
        # Increment frame number
        frame += 5
    
    # Save scores to a CSV file
    df_scores = pd.DataFrame(all_scores)
    df_scores.to_csv("image_scores_4.csv", index=False)
    print("Scores saved to image_scores_4.csv")

# Run the processing function
if __name__ == "__main__":
    process_all_frames()
