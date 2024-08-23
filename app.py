from flask import Flask, render_template, request, send_from_directory, url_for, redirect, jsonify
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
import re

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
    selected_image = request.args.get('selected_image', "")
    result_image = request.args.get('result_image', "")
    result_image_biomass = request.args.get('result_image_biomass', "-")
    
    # Ensure result_image_biomass is either a float or "-"
    if result_image_biomass not in ["", "-", None]:
        try:
            result_image_biomass = float(result_image_biomass)
        except ValueError:
            result_image_biomass = "-"

    # Ensure path uses forward slashes
    if selected_image:
        selected_image = selected_image.replace('\\', '/')

    # Load monitoring info (placeholder values for now)
    info_json = {
        "status": "",
        "recommendation": "",
        "sick_spots": "",
        "type": "",
        "week": "",
        "biomass": "-"
    }

    return render_template('index.html', 
                           selected_image=selected_image, 
                           result_image=result_image, 
                           result_image_biomass="-",  
                           status="",  
                           recommendation="",  
                           sick_count="",  
                           typ="",  
                           week="")


@app.route('/load_image', methods=['POST'])
def load_image():
    global prediction_running, biomass_cal_running, current_image, current_db_index, classification_image, dev_image

    source_type = request.form.get('source_type')

    # Check if prediction or biomass calculation is running
    if prediction_running.is_set() or biomass_cal_running.is_set():
        return redirect(url_for('index', status="A prediction is already in progress. Please wait."))

    # set calculation flags
    prediction_running.set()
    biomass_cal_running.set()
    # reset global variables
    classification_image = None
    dev_image = None
    
    # set current image
    if source_type == 'db_image':
        # Process the DB image
        image_list = sorted(os.listdir(DB_IMAGE_FOLDER))
        
        if not image_list:
            prediction_running.clear()
            biomass_cal_running.clear()
            return render_template('index.html', status="No images found in the database.")

        selected_image = image_list[current_db_index % len(image_list)]
        current_db_index += 1
        current_image = os.path.abspath(os.path.join(DB_IMAGE_FOLDER, selected_image))

    else:
        # Capture live image (either level 1 or level 2)
        timestamp = int(time.time())
        selected_image = f'{source_type}_{timestamp}.jpg'
        image_path = os.path.join(LIVE_IMAGE_FOLDER, selected_image)

        # Choose the correct camera ID based on the source_type
        camera_id = 0 if source_type == 'live_image' else 1
        captured_image_path = capture_image(image_path, camera_id=camera_id)

        if not captured_image_path:
            prediction_running.clear()
            biomass_cal_running.clear()
            return render_template('index.html', status="Error capturing live image")

        current_image = os.path.abspath(captured_image_path)
        print(f"Captured live image: {captured_image_path}")

    # Process the selected or captured image
    threading.Thread(target=process_image, args=(source_type,)).start()

    return redirect(url_for('index', selected_image=current_image))


def process_image(source_type):
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
                calculate_biomass(current_image)
        else:
            print(f"No existing predictions or annotated images for {current_image}. Running prediction and processing...")
            run_prediction(current_image, source_type) 
            calculate_biomass(current_image)

    finally:
        # Clear the flags to indicate the processing is complete
        prediction_running.clear()
        biomass_cal_running.clear()
        print(f"Processing for {current_image} completed.")


def run_prediction(image_path, source_type):
    global classification_image, dev_image, current_image
    command = [
        'python', 'run.py',
        '--source', image_path,
        '--weights', 'best.pt',
        '--save-txt',
        '--save-conf',
        '--project', PREDICTIONS_FOLDER,
        '--name', 'results'
    ]
    try:
        subprocess.run(command, check=True)

        # Save all predictions to the database
        prediction_txt_path = os.path.join(PREDICTIONS_FOLDER, 'results', 'labels', os.path.basename(image_path).replace('.jpg', '.txt'))
        with open(prediction_txt_path, 'r') as file:
            predictions = file.read().splitlines()

        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        for prediction in predictions:
            image_id = os.path.basename(image_path).replace('.jpg', '')
            prediction_with_id = f"{image_id} {prediction}"
            print(f"Saving prediction for {image_id}: {prediction_with_id}")
            cursor.execute('''
                INSERT INTO predictions (image_id, prediction)
                VALUES (?, ?)
            ''', (image_id, prediction_with_id))

        conn.commit()
        conn.close()
        print(f"Predictions for {image_path} saved.")
        
        # Generate images only once, based on the predictions
        generate_images('classification_report', image_path)
        generate_images('developer_mode', image_path)

        # Calculate and save biomass if not already calculated
        if not os.path.exists(os.path.join(BIOMASS_FOLDER, os.path.basename(image_path).replace('.jpg', '.txt'))):
            calculate_biomass(image_path)

    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
    except sqlite3.OperationalError as e:
        print(f"Database error occurred: {e}")


def calculate_biomass(image_path):
    global biomass_cal_running

    # Ensure the image path is correct and print it for debugging
    print(f"Attempting to load image from path: {image_path}")
    
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print(f"Error: Could not load image at {image_path}. Please check the file path and try again.")
        biomass_cal_running.clear()
        return None, None, None, None

    biomass, mask = calc_biomass(image)  # Correct function call

    biomass_file_path = os.path.join(BIOMASS_FOLDER, os.path.basename(image_path).replace('.jpg', '.txt'))
    with open(biomass_file_path, 'w') as f:
        f.write(str(biomass))

    images_folder = os.path.join(BIOMASS_FOLDER, 'images')
    os.makedirs(images_folder, exist_ok=True)

    green_image_file_path = os.path.join(images_folder, os.path.basename(image_path))
    save_green_extracted_image(image, mask, green_image_file_path)

    # Calculate sick spots based on predictions
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT prediction FROM predictions WHERE image_id = ?
    ''', (os.path.basename(image_path).replace('.jpg', ''),))
    predictions = cursor.fetchall()
    conn.close()

    sick_spots = 0
    for row in predictions:
        prediction = row[0].replace(f"{os.path.basename(image_path).replace('.jpg', '')} ", "")
        try:
            class_id, _, _, _, _, _ = map(float, prediction.split())
            if int(class_id) != 0:  # Assuming class_id 0 means healthy
                sick_spots += 1
        except ValueError:
            continue

    # Get status and recommendation
    status, recommendation = get_status_and_recommendation(biomass, sick_spots)

    return status, recommendation, biomass, sick_spots


def generate_images(report_type, image_path):
    global classification_image, dev_image
    
    image_id = os.path.basename(image_path).replace('.jpg', '')

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT prediction FROM predictions WHERE image_id = ?
    ''', (image_id,))
    result = cursor.fetchall()
    conn.close()

    if not result:
        print(f"No predictions found for the selected image: {image_path}")
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

    img = cv2.imread(image_path)
    output_img = img.copy()  # Image for output based on report_type

    sick_spots = 0
    for pred in best_predictions:
        abs_x_center = int((pred['box'][0] + pred['box'][2]) / 2 * img.shape[1])
        abs_y_center = int((pred['box'][1] + pred['box'][3]) / 2 * img.shape[0])
        abs_width = int((pred['box'][2] - pred['box'][0]) * img.shape[1])
        abs_height = int((pred['box'][3] - pred['box'][1]) * img.shape[0])

        top_left = (abs_x_center - abs_width // 2, abs_y_center - abs_height // 2)
        bottom_right = (abs_x_center + abs_width // 2, abs_y_center + abs_height // 2)

        if int(pred['class_id']) == 0:
            if report_type == 'developer_mode':
                color = (0, 255, 0)  # Green for Healthy
                label = f"Healthy {pred['score']:.2f}"
                cv2.rectangle(output_img, top_left, bottom_right, color, 2)
                cv2.putText(output_img, label, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            sick_spots += 1
            color = (0, 0, 255)  # Red for Sick
            label = f"Sick {pred['score']:.2f}"
            cv2.rectangle(output_img, top_left, bottom_right, color, 2)
            cv2.putText(output_img, label, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save the image based on report_type
    if report_type == 'classification_report':
        classification_image_path = os.path.join(PREDICTIONS_FOLDER, 'results', 'sick_' + os.path.basename(image_path))
        cv2.imwrite(classification_image_path, output_img)
        classification_image = classification_image_path
    elif report_type == 'developer_mode':
        dev_image_path = os.path.join(PREDICTIONS_FOLDER, 'results', 'all_' + os.path.basename(image_path))
        cv2.imwrite(dev_image_path, output_img)
        dev_image = dev_image_path


def load_monitoring_info(selected_image):
    image_id = os.path.basename(selected_image).replace('.jpg', '')

    # Check if the biomass file exists
    biomass_file_path = os.path.join(BIOMASS_FOLDER, image_id + '.txt')
    if not os.path.exists(biomass_file_path):
        print(f"Biomass file for {selected_image} does not exist. Skipping monitoring info load.")
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


@app.route('/classification_report', methods=['POST'])
def classification_report():
    selected_image = request.form.get('selected_image', current_image)
    monitoring_info = load_monitoring_info(selected_image) # Load monitoring info only when button is pressed
    return report('classification_report', monitoring_info)


@app.route('/developer_mode', methods=['POST'])
def developer_mode():
    selected_image = request.form.get('selected_image', current_image)
    monitoring_info = load_monitoring_info(selected_image)  # Load monitoring info only when button is pressed
    return report('developer_mode', monitoring_info)


def report(report_type, monitoring_info):
    global classification_image, dev_image
    
    output_image = None
    
    if prediction_running.is_set() or biomass_cal_running.is_set():
        return render_template('index.html', status="Prediction or biomass calculation is still in progress. Please wait.")
    if not current_image:
        return render_template('index.html', status=f"No image selected for {report_type.replace('_', ' ')}")
    
    if report_type == 'classification_report':
        if not classification_image:
            return render_template('index.html', status="No classification image available. Please run the prediction first.")
        else:
            output_image = classification_image
    elif report_type == 'developer_mode':
        if not dev_image:
            return render_template('index.html', status="No developer image available. Please run the prediction first.")
        else:
            output_image = dev_image
    else:
        raise ValueError(f"Invalid report type: {report_type}")

    return render_template('index.html',
                           result_image=output_image,
                           selected_image=current_image,
                           result_image_biomass=monitoring_info.get("biomass", "-"),
                           status=monitoring_info.get("status", ""),
                           recommendation=monitoring_info.get("recommendation", ""),
                           sick_count=monitoring_info.get("sick_spots", ""),
                           typ=monitoring_info.get("type", ""),
                           week=monitoring_info.get("week", ""))


@app.route('/images/<path:filepath>')
def send_image(filepath):
    directory = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    return send_from_directory(directory, filename)


@app.route('/prediction_status')
def prediction_status_endpoint():
    global prediction_running, biomass_cal_running
    return jsonify({"prediction_running": prediction_running.is_set(), "biomass_cal_running": biomass_cal_running.is_set()})


if __name__ == '__main__':
    app.run(debug=True)
