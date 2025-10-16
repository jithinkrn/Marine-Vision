import os
from flask import Flask

app = Flask(__name__, template_folder="templates")

# Set configuration values
app.config["SECRET_KEY"] = "supersecretkey"
app.config["VIDEO_UPLOAD_FOLDER"] = "app/videos/uploads/"
app.config["VIDEO_RESULT_FOLDER"] = "app/videos/results/"

app.config["VIDEO_UPLOAD_SERVE_FOLDER"] = "videos/uploads/"
app.config["VIDEO_RESULT_SERVE_FOLDER"] = "videos/results/"

app.config["ALLOWED_VIDEO_EXTENSIONS"] = {"mp4", "avi", "mov"}

# Ensure the upload and result directories exist
os.makedirs(app.config["VIDEO_UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["VIDEO_RESULT_FOLDER"], exist_ok=True)

from app import routes
