import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
import time

# Import our processing functions
from inference_utils import process_image, process_video, model

# --- Flask App Configuration ---
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/outputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB upload limit

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if a file was submitted
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']

        # Check if the filename is valid and allowed
        if file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)

        if file and model is not None:
            # Generate a unique filename to avoid conflicts
            original_filename = secure_filename(file.filename)
            timestamp = int(time.time())
            unique_filename = f"{timestamp}_{original_filename}"
            
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(input_path)
            
            # Define output path
            output_filename = f"processed_{unique_filename}"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            
            # Process based on file type
            file_extension = original_filename.rsplit('.', 1)[1].lower()
            if file_extension in {'png', 'jpg', 'jpeg'}:
                process_image(input_path, output_path)
                is_video = False
            elif file_extension == 'mp4':
                process_video(input_path, output_path)
                is_video = True
            
            # Redirect to the result page
            return render_template('result.html', filename=output_filename, is_video=is_video)

    return render_template('index.html')

# This route allows the result page to access and display the processed file
@app.route('/outputs/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')