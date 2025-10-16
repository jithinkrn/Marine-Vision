from flask import (
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory,
    flash,
)

import csv
from datetime import datetime
import os
import requests  # Import the requests library to make API calls
from werkzeug.utils import secure_filename
from app import app
from app.best_video import evaluate_output_videos
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


# # Local run API end points
# LDS_NET_API_VIDEO_URL = (
#     "http://localhost:8003/process-video/"  # LDSNet API URL (localhost)
# )

# DEEP_SESR_API_VIDEO_URL = (
#     "http://localhost:8002/process-video/"  # DeepSESRApi API URL (Docker)
# )

# FUNIEGAN_API_VIDEO_URL = (
#     "http://localhost:8001/process-video/"  # FUnIEGanApi API URL (Docker)
# )

# SEAPIXGAN_API_VIDEO_URL = (
#     "http://localhost:8000/process-video/"  # SeapixGanApi API URL (localhost)
# )

# Docker run API end points
LDS_NET_API_VIDEO_URL = (
    "http://ldsnetapi:8003/process-video/"  # LDS-NetApi API URL (Docker)
)

DEEP_SESR_API_VIDEO_URL = (
    "http://deepseesrapi:8002/process-video/"  # DeepSESRApi API URL (localhost)
)

FUNIEGAN_API_VIDEO_URL = (
    "http://funieganapi:8001/process-video/"  # FUnIEGanApi API URL (localhost)
)

SEAPIXGAN_API_VIDEO_URL = (
    "http://seapixganapi:8000/process-video/"  # SeapixGanApi API URL (Docker)
)

# Path to the feedback CSV file
CSV_FILE = "feedback.csv"


# Ensure the CSV file has headers if it doesn't exist
def initialize_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Timestamp",
                    "Video Name",
                    "LDS-Net Score",
                    "SeaPixGan Score",
                    "FUnieGan Score",
                    "DeepSESR Score",
                    "System Best Model",
                    "Human Best Model",
                ]
            )


# Initialize the CSV file with headers (if necessary)
initialize_csv()


# Helper function to check allowed video files
def allowed_video_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_VIDEO_EXTENSIONS"]
    )


# Function to process video with LDS-Net
def process_video_with_lds_net(filepath, filename):

    start_time = time.time()
    print(f"Processing LDS-Net started at {datetime.now()}")

    # Send the video to the LDSNetApi for processing
    with open(filepath, "rb") as video_file:
        files = {"file": video_file}
        response = requests.post(LDS_NET_API_VIDEO_URL, files=files)

    if response.status_code == 200:
        # Save the processed video received from the API
        result_filename = f"LDS-Net_{filename}"
        save_filepath = os.path.join(app.config["VIDEO_RESULT_FOLDER"], result_filename)

        # Write the processed video content to the results folder
        with open(save_filepath, "wb") as f:
            f.write(response.content)

        result_filepath = result_filename

        elapsed_time = time.time() - start_time
        print(f"LDS-Net processing completed in {elapsed_time:.2f} seconds")

        # Return the result filepath

        return result_filepath, None
    else:
        # Return an error if processing failed
        return None, "Error processing video with LDS-Net"


# Function to process video with SeepSESR
def process_video_with_deep_sesr(filepath, filename):

    start_time = time.time()
    print(f"Processing DeepSESR started at {datetime.now()}")

    # Send the video to the LDSNetApi for processing
    with open(filepath, "rb") as video_file:
        files = {"file": video_file}
        response = requests.post(DEEP_SESR_API_VIDEO_URL, files=files)

    if response.status_code == 200:
        # Save the processed video received from the API
        result_filename = f"DeepSESR_{filename}"
        save_filepath = os.path.join(app.config["VIDEO_RESULT_FOLDER"], result_filename)

        # Write the processed video content to the results folder
        with open(save_filepath, "wb") as f:
            f.write(response.content)

        result_filepath = result_filename

        elapsed_time = time.time() - start_time
        print(f"DeepSESR processing completed in {elapsed_time:.2f} seconds")

        # Return the result filepath
        return result_filepath, None
    else:
        # Return an error if processing failed
        return None, "Error processing video with DeepSESR"


# Function to process video with FUnieGan
def process_video_with_funiegan(filepath, filename):
    start_time = time.time()
    print(f"Processing FUnieGan started at {datetime.now()}")
    # Send the video to the LDSNetApi for processing
    with open(filepath, "rb") as video_file:
        files = {"file": video_file}
        response = requests.post(FUNIEGAN_API_VIDEO_URL, files=files)

    if response.status_code == 200:
        # Save the processed video received from the API
        result_filename = f"FUnieGan_{filename}"
        save_filepath = os.path.join(app.config["VIDEO_RESULT_FOLDER"], result_filename)

        # Write the processed video content to the results folder
        with open(save_filepath, "wb") as f:
            f.write(response.content)

        result_filepath = result_filename

        elapsed_time = time.time() - start_time
        print(f"FUnieGan processing completed in {elapsed_time:.2f} seconds")
        # Return the result filepath
        return result_filepath, None
    else:
        # Return an error if processing failed
        return None, "Error processing video with FunieGan"


# Function to process video with SeaPixGan
def process_video_with_seapixgan(filepath, filename):
    start_time = time.time()
    print(f"Processing SeapixGan started at {datetime.now()}")
    # Send the video to the LDSNetApi for processing
    with open(filepath, "rb") as video_file:
        files = {"file": video_file}
        response = requests.post(SEAPIXGAN_API_VIDEO_URL, files=files)

    if response.status_code == 200:
        # Save the processed video received from the API
        result_filename = f"SeaPixGan_{filename}"
        save_filepath = os.path.join(app.config["VIDEO_RESULT_FOLDER"], result_filename)

        # Write the processed video content to the results folder
        with open(save_filepath, "wb") as f:
            f.write(response.content)

        result_filepath = result_filename

        elapsed_time = time.time() - start_time
        print(f"SeapixGan processing completed in {elapsed_time:.2f} seconds")
        # Return the result filepath
        return result_filepath, None
    else:
        # Return an error if processing failed
        return None, "Error processing video with Seapix-Gan"


# Function to delete all files in a directory
def delete_all_files_in_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)  # Remove the file
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")


# Update process_all_videos to accept selected models
def process_selected_videos(filepath, filename, selected_models):
    """Runs selected video processing functions in parallel, with DeepSESR run sequentially if selected."""

    # Available processing functions with their names as keys
    model_functions = {
        "LDS-Net": process_video_with_lds_net,
        "FunieGan": process_video_with_funiegan,
        "Sea-pixGan": process_video_with_seapixgan,
        "DeepSESR": process_video_with_deep_sesr,  # Sequential model
    }

    # Filter the models to run in parallel (excluding DeepSESR)
    parallel_functions = [
        model_functions[model] for model in selected_models if model != "DeepSESR"
    ]

    results = {}
    start_time = time.time()
    print(f"Parallel video processing started at {datetime.now()}")

    # Run selected models in parallel
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(func, filepath, filename): func.__name__
            for func in parallel_functions
        }

        for future in as_completed(futures):
            func_name = futures[future]
            try:
                result, error = future.result()
                results[func_name] = (result, error)
            except Exception as exc:
                results[func_name] = (None, str(exc))

    # Run DeepSESR sequentially if selected
    if "DeepSESR" in selected_models:
        print(f"Running {model_functions['DeepSESR'].__name__} sequentially...")
        try:
            result, error = model_functions["DeepSESR"](filepath, filename)
            results[model_functions["DeepSESR"].__name__] = (result, error)
        except Exception as exc:
            results[model_functions["DeepSESR"].__name__] = (None, str(exc))

    total_elapsed_time = time.time() - start_time
    print(f"Total video processing completed in {total_elapsed_time:.2f} seconds")

    return results


@app.route("/", methods=["GET", "POST"])
@app.route("/upload_video", methods=["GET", "POST"])
def upload_video():
    upload_time = time.time()
    print(f"Request received at {datetime.now()}")

    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if file and allowed_video_file(file.filename):
            # Get selected models from the form data
            selected_models = request.form.getlist("model")
            # Ensure at least one model is selected
            if len(selected_models) < 1:
                flash("Please select at least one model.")
                return redirect(request.url)

            # Delete old files
            delete_all_files_in_directory(app.config["VIDEO_UPLOAD_FOLDER"])
            delete_all_files_in_directory(app.config["VIDEO_RESULT_FOLDER"])

            # Save uploaded video
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["VIDEO_UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Process the selected models
            results = process_selected_videos(filepath, filename, selected_models)

            # Collect results and errors
            errors = []
            video_paths = []

            for func_name, (result, error) in results.items():
                if error:
                    errors.append(f"{func_name} failed: {error}")
                else:
                    video_paths.append(
                        os.path.join(app.config["VIDEO_RESULT_FOLDER"], result)
                    )

            # If there are errors, flash them
            if errors:
                for error in errors:
                    flash(error)
                return redirect(request.url)

            print("Evaluating best video selection...")
            selection_start_time = time.time()

            # Evaluate the videos only if all processing succeeded
            evaluation_results = evaluate_output_videos(video_paths)

            # Find the best model and create a scores dictionary
            best_model = min(evaluation_results, key=lambda x: x[1])
            scores = {model: score for model, score in evaluation_results}

            selection_elapsed_time = time.time() - selection_start_time
            print(
                f"Best video selection completed in {selection_elapsed_time:.2f} seconds"
            )

            # Flash success message and render the result
            flash("Video successfully processed and returned")
            total_response_time = time.time() - upload_time
            print(f"Response retuned at {total_response_time:.2f} seconds")
            return render_template(
                "upload.html",
                input_video=filename,
                ldsnet_result_video=results.get("process_video_with_lds_net", (None,))[
                    0
                ],
                seapixgan_result_video=results.get(
                    "process_video_with_seapixgan", (None,)
                )[0],
                funiegan_result_video=results.get(
                    "process_video_with_funiegan", (None,)
                )[0],
                deepsesr_result_video=results.get(
                    "process_video_with_deep_sesr", (None,)
                )[0],
                ldsnet_score=float(scores.get("LDS-Net", 0)),
                seapixgan_score=float(scores.get("SeaPixGan", 0)),
                funiegan_score=float(scores.get("FUnieGan", 0)),
                deepsesr_score=float(scores.get("DeepSESR", 0)),
                best_model=best_model[0],
                best_score=best_model[1],
            )
        else:
            flash("Invalid file type. Please upload a valid video file.")
            return redirect(request.url)
    return render_template("upload.html")


@app.route("/submit_feedback", methods=["POST"])
def submit_feedback():

    # Collect data from the form
    feedback = request.form.get(
        "human_best_video"
    )  # e.g., "LDS-Net:ldsnet_result_video.mp4"
    human_best_model, video_name = feedback.split(":")  # Extract model and video name

    # Collect all scores and system-best model
    ldsnet_score = request.form.get("ldsnet_score")
    seapixgan_score = request.form.get("seapixgan_score")
    funiegan_score = request.form.get("funiegan_score")
    deepsesr_score = request.form.get("deepsesr_score")
    system_best_model = request.form.get("system_best_model")

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save feedback to CSV
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                timestamp,
                video_name,
                ldsnet_score,
                seapixgan_score,
                funiegan_score,
                deepsesr_score,
                system_best_model,
                human_best_model,
            ]
        )

    flash("Thank you for your feedback!")
    return redirect(url_for("upload_video"))


# Serve uploaded videos
@app.route("/uploads/videos/<filename>")
def uploaded_video(filename):
    return send_from_directory(app.config["VIDEO_UPLOAD_SERVE_FOLDER"], filename)


# Serve result videos (if applicable in the future)
@app.route("/results/videos/<filename>")
def result_video(filename):
    return send_from_directory(app.config["VIDEO_RESULT_SERVE_FOLDER"], filename)
