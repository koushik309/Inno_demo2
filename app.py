from flask import Flask, render_template, request, send_file, url_for, redirect, jsonify, abort
import os
import subprocess
from pathlib import Path
import sqlite3
import cv2
import threading
import time
from biomass import calculate_biomass as calc_biomass, save_green_extracted_image, get_status_and_recommendation
from camera import capture_image
from bestprediction import select_best_prediction
import urllib.parse

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
            classification_image = classification_image_path
            dev_image = dev_image_path
            if not os.path.exists(os.path.join(BIOMASS_FOLDER, image_id + '.txt')):
                calculate_biomass()
        else:
            print(f"No existing predictions or annotated images for {current_image}. Running prediction and processing...")
            run_prediction() 
            calculate_biomass()

    finally:
        # Clear the flags to indicate the processing is complete
        prediction_running.clear()
        biomass_cal_running.clear()
        print(f"Processing for {current_image} completed.")


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
            print(f"Saving prediction for {image_id}: {prediction_with_id}")
            cursor.execute('''
                INSERT INTO predictions (image_id, prediction)
                VALUES (?, ?)
            ''', (image_id, prediction_with_id))

        conn.commit()
        conn.close()
        print(f"Predictions for {current_image} saved.")
        
        generate_images()

        # Calculate and save biomass if not already calculated
        if not os.path.exists(os.path.join(BIOMASS_FOLDER, os.path.basename(current_image).replace('.jpg', '.txt'))):
            calculate_biomass()

    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
    except sqlite3.OperationalError as e:
        print(f"Database error occurred: {e}")


def calculate_biomass():
    global biomass_cal_running, current_image

    # Ensure the image path is correct and print it for debugging
    print(f"Attempting to load image from path: {current_image}")
    
    image = cv2.imread(current_image)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Error: Could not load image at {current_image}. Please check the file path and try again.")
        biomass_cal_running.clear()
        return None, None, None, None

    biomass, mask = calc_biomass(image)  # Correct function call

    biomass_file_path = os.path.join(BIOMASS_FOLDER, os.path.basename(current_image).replace('.jpg', '.txt'))
    with open(biomass_file_path, 'w') as f:
        f.write(str(biomass))

    images_folder = os.path.join(BIOMASS_FOLDER, 'images')
    os.makedirs(images_folder, exist_ok=True)

    green_image_file_path = os.path.join(images_folder, os.path.basename(current_image))
    save_green_extracted_image(image, mask, green_image_file_path)

    # Calculate sick spots based on predictions
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT prediction FROM predictions WHERE image_id = ?
    ''', (os.path.basename(current_image).replace('.jpg', ''),))
    predictions = cursor.fetchall()
    conn.close()

    sick_spots = 0
    for row in predictions:
        prediction = row[0].replace(f"{os.path.basename(current_image).replace('.jpg', '')} ", "")
        try:
            class_id, _, _, _, _, _ = map(float, prediction.split())
            if int(class_id) != 0:  # Assuming class_id 0 means healthy
                sick_spots += 1
        except ValueError:
            continue

    # Get status and recommendation
    status, recommendation = get_status_and_recommendation(biomass, sick_spots)

    return status, recommendation, biomass, sick_spots


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
    with open(biomass_file_path, 'r') as f:
        biomass = float(f.read().strip())

    # Retrieve predictions to calculate sick spots
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

    # Get status and recommendation from existing data
    status, recommendation = get_status_and_recommendation(biomass, sick_spots)

    # Populate the info_json with all necessary details
    info_json = {
        "week": 3,  # Hardcoded for now; adjust as needed
        "type": "Basil",  # Adjust as needed
        "status": status,
        "recommendation": recommendation,
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
