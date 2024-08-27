from flask import Flask, render_template, request, send_file, url_for, redirect, jsonify, abort
import os
import subprocess
from pathlib import Path
import sqlite3
import cv2
import threading
import time
from biomass import calculate_biomass as calc_biomass, save_green_extracted_image, get_status_and_recommendations
from camera import capture_image
from bestprediction import select_best_prediction
from cropping import crop_image
import warnings
import numpy as np

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Directory where your database images are stored
DB_IMAGE_FOLDER = 'db_images'
LIVE_IMAGE_FOLDER = 'live_images'
PREDICTIONS_FOLDER = 'predictions'
BIOMASS_FOLDER = 'biomass'
Path(PREDICTIONS_FOLDER).mkdir(parents=True, exist_ok=True)
Path(BIOMASS_FOLDER).mkdir(parents=True, exist_ok=True)
Path(LIVE_IMAGE_FOLDER).mkdir(parents=True, exist_ok=True)

# SQLite database for storing predictions
DATABASE = 'predictions.db'

# Global variables to track the status and image details
prediction_running = threading.Event()
biomass_cal_running = threading.Event()
displayed_image = ""
current_image = ""
classification_image = ""
dev_image = ""
current_db_index = 0


def init_db():
    print("Initializing database...")
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    # Create the predictions table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id TEXT NOT NULL,
            prediction TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
    print("Database initialized.")


init_db()

@app.route('/')
def index():
    global displayed_image
    # Initial page load should not include monitoring info
    return render_template('index.html', displayed_image=displayed_image, json_info=None)

@app.route('/reset_image')
def reset_image():
    global displayed_image, current_db_index, current_image, classification_image, dev_image
    displayed_image = ""  # Clear the displayed image
    current_db_index = 0
    current_image = ""
    classification_image = ""
    dev_image = ""
    return redirect(url_for('index'))

@app.route('/load_image', methods=['POST'])
def load_image():
    global prediction_running, biomass_cal_running, current_image, current_db_index, classification_image, dev_image, displayed_image

    source_type = request.form.get('source_type')

    # Check if prediction or biomass calculation is running
    if prediction_running.is_set() or biomass_cal_running.is_set():
        return redirect(url_for('index', status="A prediction is already in progress. Please wait."))

    # Set calculation flags
    prediction_running.set()
    biomass_cal_running.set()
    # Reset global variables
    classification_image = None
    dev_image = None
    
    # Set current image based on the source type
    if source_type == 'db':
        image_list = sorted(os.listdir(DB_IMAGE_FOLDER))
        
        if not image_list:
            prediction_running.clear()
            biomass_cal_running.clear()
            return render_template('index.html', status="No images found in the database.")

        selected_image = image_list[current_db_index % len(image_list)]
        current_db_index += 1
        current_image = os.path.abspath(os.path.join(DB_IMAGE_FOLDER, selected_image))

    elif source_type in ['cam1', 'cam2']:
        timestamp = int(time.time())
        selected_image = f'{source_type}_{timestamp}.jpg'
        image_path = os.path.join(LIVE_IMAGE_FOLDER, selected_image)

        if source_type == 'cam1':
            camera_id = 0
        elif source_type == 'cam2':
            camera_id = 1
        captured_image_path = capture_image(image_path, camera_id=camera_id)

        if not captured_image_path:
            prediction_running.clear()
            biomass_cal_running.clear()
            return render_template('index.html', status="Error capturing live image")

        current_image = os.path.abspath(captured_image_path)
        print(f"Captured live image: {captured_image_path}")
    else:
        return abort(400)
    
    displayed_image = current_image
    
    # Start processing the image in a new thread
    threading.Thread(target=process_image).start()

    return redirect(url_for('index'))

def process_image():
    global current_image, classification_image, dev_image, prediction_running, biomass_cal_running

    try:
        image_id = os.path.basename(current_image).replace('.jpg', '')

        # Check if the predictions for this image already exist
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT prediction FROM predictions WHERE image_id = ?
        ''', (image_id,))
        predictions = cursor.fetchall()
        conn.close()

        # Paths for the annotated images
        classification_image_path = os.path.join(PREDICTIONS_FOLDER, 'results', 'sick_' + os.path.basename(current_image))
        dev_image_path = os.path.join(PREDICTIONS_FOLDER, 'results', 'all_' + os.path.basename(current_image))

        # Check if both the predictions and the annotated images exist
        if predictions:
            print(f"Predictions and annotated images already exist for {current_image}.")
            prediction_running.clear()
            classification_image = classification_image_path
            dev_image = dev_image_path
            if not os.path.exists(os.path.join(BIOMASS_FOLDER, image_id + '.txt')):
                start_calculate_biomass_thread()
            else:
                biomass_cal_running.clear()
        else:
            print(f"No existing predictions or annotated images for {current_image}. Running prediction and processing...")
            run_prediction() 
            start_calculate_biomass_thread()
    except Exception as e:
        print(f"An error occurred while processing the image: {e}")
        # Clear the flags to allow processing of the next image
        prediction_running.clear()
        biomass_cal_running.clear()



def get_week() -> int:
    global current_image
    basename = os.path.basename(current_image)
    switch_dict = {
        "image_0_20231222-230226.jpg": 1,
        "image_0_20240312-120507.jpg": 4,
        "image_0_20240318-120020.jpg":4,
        "image_0_20240322-060004.jpg":4,
        "image_0_20240327-060005.jpg":4,
        "image_0_20240605-060010.jpg":4,
        "image_0_20240815-060343.jpg":4,
        "image_1_20240201-060010.jpg":2,
        "image_1_20240209-120501.jpg":2,
        "image_2_20240405-140501.jpg":2,
        "image_2_20240601-060308.jpg":3,
        "image_3_20240624-140303.jpg":5,
        "image_3_20240605-060704.jpg":3
    }
    return switch_dict.get(basename, 3)

def run_prediction():
    global classification_image, dev_image, current_image
    command = [
        'python', 'run.py',
        '--source', current_image,
        '--weights', 'best.pt',
        '--save-txt',
        '--save-conf',
        '--project', PREDICTIONS_FOLDER,
        '--name', 'results'
    ]
    try:
        subprocess.run(command, check=True)

        # Save all predictions to the database
        prediction_txt_path = os.path.join(PREDICTIONS_FOLDER, 'results', 'labels', os.path.basename(current_image).replace('.jpg', '.txt'))
        with open(prediction_txt_path, 'r') as file:
            predictions = file.read().splitlines()

        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        for prediction in predictions:
            image_id = os.path.basename(current_image).replace('.jpg', '')
            prediction_with_id = f"{image_id} {prediction}"
            #print(f"Saving prediction for {image_id}: {prediction_with_id}")
            cursor.execute('''
                INSERT INTO predictions (image_id, prediction)
                VALUES (?, ?)
            ''', (image_id, prediction_with_id))

        conn.commit()
        conn.close()
        print(f"Predictions for {current_image} saved.")
        
        generate_images()
        
        prediction_running.clear()

    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
    except sqlite3.OperationalError as e:
        print(f"Database error occurred: {e}")
    finally:
        prediction_running.clear()


def apply_scaling_correction(img, difference):
    """
    Apply a white balance correction to the image based on the provided difference.
    """
    # Calculate the correction factor from the difference
    correction_factor = 1 + (difference / 255.0)
    print(f"Correction factor: {correction_factor}")

    # Convert the image to float32 for precise adjustment
    corrected_img = img.astype(np.float32)

    # Apply the correction factor to each channel (B, G, R)
    for c in range(3):  # Iterating over B, G, R channels
        corrected_img[:, :, c] *= correction_factor[c]

    # Clip values to be in the range [0, 255] and convert back to uint8
    corrected_img = np.clip(corrected_img, 0, 255).astype(np.uint8)

    return corrected_img

def start_calculate_biomass_thread():
    thread = threading.Thread(target=calculate_biomass)
    thread.start()
    
def calculate_biomass():
    global biomass_cal_running, current_image

    print(f"Attempting to load image from path: {current_image}")
    
    image = cv2.imread(current_image)

    if image is None:
        print(f"Error: Could not load image at {current_image}. Please check the file path and try again.")
        biomass_cal_running.clear()
        return

    # Extract camera ID from the image filename
    try:
        filename = os.path.basename(current_image)
        camera_id = int(filename.split('_')[1])
    except (IndexError, ValueError) as e:
        # continue without cropping
        warnings.warn(f"Warning extracting camera ID from filename {filename}: {e}", UserWarning)
        camera_id = None


    # Perform cropping on the image based on the camera ID
    if camera_id is None:
        cropped_image = image
    else:
        cropped_image = crop_image(image, camera_id)

    if cropped_image is None:
        warnings.warn(f"Error cropping image for camera ID {camera_id}. Using uncropped image.", UserWarning)
        cropped_image = image

    # Apply white balance correction
    Difference = np.array([0, 4, 52])
    corrected_image = apply_scaling_correction(cropped_image, Difference)

    # Calculate biomass using the corrected image
    biomass, mask = calc_biomass(corrected_image) # change cropped image to image to remove cropping and white balance.

    # Save biomass value
    image_id = os.path.basename(current_image).replace('.jpg', '')
    biomass_file_path = os.path.join(BIOMASS_FOLDER, f"{image_id}.txt")
    with open(biomass_file_path, 'w') as f:
        f.write(str(biomass))

    # Optionally save the green extracted image
    images_folder = os.path.join(BIOMASS_FOLDER, 'images')
    os.makedirs(images_folder, exist_ok=True)

    green_image_file_path = os.path.join(images_folder, f"{image_id}.jpg")
    save_green_extracted_image(corrected_image, mask, green_image_file_path)

    biomass_cal_running.clear()
    return


def generate_images():
    global classification_image, dev_image, current_image
    
    image_id = os.path.basename(current_image).replace('.jpg', '')

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT prediction FROM predictions WHERE image_id = ?
    ''', (image_id,))
    result = cursor.fetchall()
    conn.close()

    if not result:
        print(f"No predictions found for the selected image: {current_image}")
        return

    predictions = []
    for row in result:
        prediction = row[0].replace(f"{image_id} ", "")
        try:
            class_id, x_center, y_center, width, height, conf = map(float, prediction.split())
            box = [x_center - width / 2, y_center - height / 2, x_center + width / 2, y_center + height / 2]
            predictions.append({'class_id': class_id, 'box': box, 'score': conf})
        except ValueError:
            continue

    best_predictions = select_best_prediction(predictions)

    img = cv2.imread(current_image)
    
    output_img = img.copy()  # Image for output based on report_type
    annotate_image(best_predictions, output_img, 'classification_report')
    
    classification_image_path = os.path.join(PREDICTIONS_FOLDER, 'results', 'sick_' + os.path.basename(current_image))
    cv2.imwrite(classification_image_path, output_img)
    classification_image = classification_image_path

    output_img = img.copy()  # Image for output based on report_type
    annotate_image(best_predictions, output_img, 'developer_mode')
    
    dev_image_path = os.path.join(PREDICTIONS_FOLDER, 'results', 'all_' + os.path.basename(current_image))
    cv2.imwrite(dev_image_path, output_img)
    dev_image = dev_image_path


def annotate_image(best_predictions, output_img, report_type):
    for pred in best_predictions:
        abs_x_center = int((pred['box'][0] + pred['box'][2]) / 2 * output_img.shape[1])
        abs_y_center = int((pred['box'][1] + pred['box'][3]) / 2 * output_img.shape[0])
        abs_width = int((pred['box'][2] - pred['box'][0]) * output_img.shape[1])
        abs_height = int((pred['box'][3] - pred['box'][1]) * output_img.shape[0])

        top_left = (abs_x_center - abs_width // 2, abs_y_center - abs_height // 2)
        bottom_right = (abs_x_center + abs_width // 2, abs_y_center + abs_height // 2)

        if int(pred['class_id']) == 0:
            if report_type == 'developer_mode':
                color = (0, 255, 0)  # Green for Healthy
                label = f"Healthy {pred['score']:.2f}"
                cv2.rectangle(output_img, top_left, bottom_right, color, 2)
                cv2.putText(output_img, label, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            color = (0, 0, 255)  # Red for Sick
            label = f"Sick {pred['score']:.2f}"
            cv2.rectangle(output_img, top_left, bottom_right, color, 2)
            cv2.putText(output_img, label, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def load_monitoring_info():
    global current_image
    image_id = os.path.basename(current_image).replace('.jpg', '')

    # Check if the biomass file exists
    biomass_file_path = os.path.join(BIOMASS_FOLDER, image_id + '.txt')
    if not os.path.exists(biomass_file_path):
        print(f"Biomass file for {current_image} does not exist. Skipping monitoring info load.")
        return {}

    # Load biomass value
    try:
        with open(biomass_file_path, 'r') as f:
            biomass = float(f.read().strip())
    except (FileNotFoundError, ValueError, IOError) as e:
        print(f"Error loading biomass value: {e}")
        biomass = None
        
    # Retrieve predictions to calculate sick spots
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT prediction FROM predictions WHERE image_id = ?
        ''', (image_id,))
        predictions = cursor.fetchall()
        conn.close()

        sick_spots = 0
        for row in predictions:
            prediction = row[0].replace(f"{image_id} ", "")
            try:
                class_id, _, _, _, _, _ = map(float, prediction.split())
                if int(class_id) != 0:  # Assuming class_id 0 means healthy
                    sick_spots += 1
            except ValueError:
                continue
    except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
        print(f"Database error: {e}")
        sick_spots = None
        
    # Get the week number using the get_week function
    week_number = get_week()

    # Get status and recommendation based on biomass, sick spots, and week number
    result = get_status_and_recommendations(biomass, sick_spots, week_number)

    # Populate the info_json with all necessary details
    info_json = {
        "week": week_number,  # Now using dynamic week number
        "type": "Basil",  # Adjust as needed
        "status": result['status'],
        "recommendation": result['recommendation'],
        "biomass": biomass,
        "sick_spots": sick_spots
    }

    return info_json

@app.route('/report')
def report():
    global classification_image, dev_image, displayed_image
    report_type = request.args.get('type')
    
    if prediction_running.is_set() or biomass_cal_running.is_set():
        return render_template('index.html', status="Prediction or biomass calculation is still in progress. Please wait.")
    if not current_image:
        return render_template('index.html', status=f"No image selected for {report_type.replace('_', ' ')}")
    
    # Determine which image to display based on the report type
    if report_type == 'classification_report':
        if not classification_image:
            return render_template('index.html', status="No classification image available. Please run the prediction first.")
        else:
            displayed_image = classification_image
    elif report_type == 'developer_mode':
        if not dev_image:
            return render_template('index.html', status="No developer image available. Please run the prediction first.")
        else:
            displayed_image = dev_image
    else:
        raise ValueError(f"Invalid report type: {report_type}")

    # Load monitoring info only when a report is requested
    json_info = load_monitoring_info()

    # Render the template with the selected image and monitoring info
    return render_template('index.html', displayed_image=displayed_image, json_info=json_info)


@app.route('/images')
def send_image():
    filepath = request.args.get('filepath')
    # Since security is not a concern, directly use the filepath
    if not os.path.exists(filepath):
        return abort(404)  # Return a 404 error if the file doesn't exist
    
    return send_file(filepath)  # send_file directly serves the file


@app.route('/prediction_status')
def prediction_status_endpoint():
    global prediction_running, biomass_cal_running
    return jsonify({"prediction_running": prediction_running.is_set(), "biomass_cal_running": biomass_cal_running.is_set()})


if __name__ == '__main__':
    app.run(debug=True)
