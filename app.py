from flask import Flask, render_template, request, send_from_directory, url_for, redirect, jsonify
import os
import subprocess
from pathlib import Path
import sqlite3
import cv2
import threading
import time
from biomass import calculate_biomass, save_green_extracted_image
from monitoring import generate_monitoring_info
from camera import capture_image
from bestprediction import select_best_prediction
import re

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Directory where your database images are stored
DB_IMAGE_FOLDER = 'cropped_images'
LIVE_IMAGE_FOLDER = 'live_images'
PREDICTIONS_FOLDER = 'predictions'
BIOMASS_FOLDER = 'biomass'
Path(PREDICTIONS_FOLDER).mkdir(parents=True, exist_ok=True)
Path(BIOMASS_FOLDER).mkdir(parents=True, exist_ok=True)
Path(LIVE_IMAGE_FOLDER).mkdir(parents=True, exist_ok=True)

# SQLite database for storing predictions
DATABASE = 'predictions.db'

# Global variables to track the status and image details
prediction_running = False
biomass_cal_running = False
current_image = ""
current_image_type = ""
classification_image = ""
dev_image = ""

# Lock to ensure thread safety
prediction_lock = threading.Lock()

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
    status = request.args.get('status', "")
    recommendation = request.args.get('recommendation', "")
    selected_image_type = request.args.get('selected_image_type', "")
    sick_count = request.args.get('sick_count', "")
    typ = request.args.get('typ', "")
    week = request.args.get('week', "")

    # Ensure result_image_biomass is either a float or "-"
    if result_image_biomass not in ["", "-", None]:
        try:
            result_image_biomass = float(result_image_biomass)
        except ValueError:
            result_image_biomass = "-"

    return render_template('index.html', 
                           selected_image=selected_image, 
                           result_image=result_image, 
                           result_image_biomass=result_image_biomass, 
                           status=status, 
                           recommendation=recommendation, 
                           selected_image_type=selected_image_type, 
                           sick_count=sick_count, 
                           typ=typ, 
                           week=week)

@app.route('/load_image', methods=['POST'])
def load_image():
    global prediction_running, biomass_cal_running, current_image, current_image_type, prediction_lock

    source_type = request.form.get('source_type')

    with prediction_lock:
        if prediction_running:
            return redirect(url_for('index', result="A prediction is already in progress. Please wait."))

        # Mark that a prediction is running
        prediction_running = True

        if source_type == 'db_image':
            # Process the DB image
            image_list = sorted(os.listdir(DB_IMAGE_FOLDER))
            
            if not image_list:
                return render_template('index.html', result="No images found in the database.")

            # Find the index of the current image and select the next one
            if current_image in image_list:
                current_index = image_list.index(current_image)
                next_index = (current_index + 1) % len(image_list)
            else:
                next_index = 0  # Start from the first image if current_image is not in the list

            selected_image = image_list[next_index]
            image_path = os.path.join(DB_IMAGE_FOLDER, selected_image)

        else:
            # Capture live image (either level 1 or level 2)
            timestamp = int(time.time())
            selected_image = f'{source_type}_{timestamp}.jpg'
            image_path = os.path.join(LIVE_IMAGE_FOLDER, selected_image)

            # Choose the correct camera ID based on the source_type
            camera_id = 0 if source_type == 'live_image' else 1
            captured_image_path = capture_image(image_path, camera_id=camera_id)

            if not captured_image_path:
                with prediction_lock:
                    prediction_running = False
                return render_template('index.html', result="Error capturing live image")

            print(f"Captured live image: {captured_image_path}")

        # Process the selected or captured image
        threading.Thread(target=process_image, args=(image_path, selected_image)).start()

    return redirect(url_for('index', selected_image=selected_image, selected_image_type=source_type))

def process_image(image_path, selected_image):
    global current_image, current_image_type, prediction_running, biomass_cal_running, prediction_lock

    try:
        current_image = selected_image
        current_image_type = 'db_image' if 'db_images' in image_path else 'live_image'

        image_id = os.path.basename(selected_image).replace('.jpg', '')

        # Check if the predictions for this image are already in the database
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT prediction FROM predictions WHERE image_id = ?
        ''', (image_id,))
        predictions = cursor.fetchall()
        conn.close()

        if predictions:
            # Predictions exist, use them directly
            print(f"Predictions for {current_image} already exist in the database.")
            handle_existing_predictions(current_image, predictions)
        else:
            # Predictions do not exist, perform prediction and biomass calculation
            print(f"No existing predictions for {current_image}. Running prediction...")
            run_prediction(image_path)
            calculate_and_save_biomass(image_path)

    finally:
        with prediction_lock:
            prediction_running = False
            biomass_cal_running = False

        print(f"Processing for {selected_image} completed.")

def handle_existing_predictions(image_name, predictions):
    """
    Handle the existing predictions for an image.
    This function can be used to process and display predictions from the database.
    """
    print(f"Handling existing predictions for {image_name}.")
    for prediction in predictions:
        print(f"Prediction: {prediction}")
    
    # Optionally, update the classification image or status to reflect that it's already predicted
    global classification_image
    classification_image = image_name

def run_prediction(image_path):
    global prediction_running, current_image, classification_image, prediction_lock
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

        with prediction_lock:
            classification_image = os.path.basename(image_path)

    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
    except sqlite3.OperationalError as e:
        print(f"Database error occurred: {e}")
    finally:
        with prediction_lock:
            prediction_running = False

def calculate_and_save_biomass(image_path):
    global biomass_cal_running, current_image, prediction_lock
    image = cv2.imread(image_path)
    biomass, mask = calculate_biomass(image)

    biomass_file_path = os.path.join(BIOMASS_FOLDER, os.path.basename(image_path).replace('.jpg', '.txt'))
    with open(biomass_file_path, 'w') as f:
        f.write(str(biomass))

    images_folder = os.path.join(BIOMASS_FOLDER, 'images')
    os.makedirs(images_folder, exist_ok=True)

    green_image_file_path = os.path.join(images_folder, os.path.basename(image_path))
    save_green_extracted_image(image, mask, green_image_file_path)

    with prediction_lock:
        biomass_cal_running = False

def report(report_type):
    global prediction_running, biomass_cal_running, classification_image, prediction_lock
    selected_image = request.form.get('selected_image')
    selected_image_type = request.form.get('image_type')

    if prediction_running or biomass_cal_running:
        return render_template('index.html', result="Prediction or biomass calculation is still in progress. Please wait.")

    if not selected_image:
        return render_template('index.html', result=f"No image selected for {report_type.replace('_', ' ')}")

    image_id = os.path.basename(selected_image).replace('.jpg', '')

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT prediction FROM predictions WHERE image_id = ?
    ''', (image_id,))
    result = cursor.fetchall()
    conn.close()

    if not result:
        return render_template('index.html', result=f"No predictions found for the selected image")

    predictions = []
    for row in result:
        if len(row) < 1:
            continue
        prediction = row[0].replace(f"{image_id} ", "")
        
        # Attempt to split the prediction string correctly
        try:
            class_id, x_center, y_center, width, height, conf = map(float, prediction.split())
        except ValueError:
            # Handle cases where the string is not correctly formatted
            prediction_parts = re.findall(r'[0-9.]+', prediction)
            if len(prediction_parts) == 6:
                class_id, x_center, y_center, width, height, conf = map(float, prediction_parts)
            else:
                # Skip this prediction if it cannot be parsed correctly
                continue
        
        box = [x_center - width/2, y_center - height/2, x_center + width/2, y_center + height/2]
        predictions.append({'class_id': class_id, 'box': box, 'score': conf})

    # Select the best predictions using the select_best_prediction function
    best_predictions = select_best_prediction(predictions)

    img = cv2.imread(os.path.join(LIVE_IMAGE_FOLDER if selected_image_type == 'live_image' else DB_IMAGE_FOLDER, selected_image))

    sick_spots = 0
    for pred in best_predictions:
        abs_x_center = int((pred['box'][0] + pred['box'][2]) / 2 * img.shape[1])
        abs_y_center = int((pred['box'][1] + pred['box'][3]) / 2 * img.shape[0])
        abs_width = int((pred['box'][2] - pred['box'][0]) * img.shape[1])
        abs_height = int((pred['box'][3] - pred['box'][1]) * img.shape[0])

        top_left = (abs_x_center - abs_width // 2, abs_y_center - abs_height // 2)
        bottom_right = (abs_x_center + abs_width // 2, abs_y_center + abs_height // 2)

        if int(pred['class_id']) == 0:
            if report_type == 'developer_mode':  # Show "healthy" only in developer mode
                color = (0, 255, 0)  # Green for Healthy
                label = f"Healthy {pred['score']:.2f}"
                cv2.rectangle(img, top_left, bottom_right, color, 2)
                cv2.putText(img, label, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            sick_spots += 1
            color = (0, 0, 255)  # Red for Sick
            label = f"Sick {pred['score']:.2f}"
            cv2.rectangle(img, top_left, bottom_right, color, 2)
            cv2.putText(img, label, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Load biomass value
    biomass_file_path = os.path.join(BIOMASS_FOLDER, os.path.basename(selected_image).replace('.jpg', '.txt'))
    if os.path.exists(biomass_file_path):
        with open(biomass_file_path, 'r') as f:
            biomass = float(f.read().strip())
    else:
        biomass = 0.0

    # Use a dummy week number and type for demonstration
    week = 3
    typ = "Basil"

    # Generate monitoring information
    status, recommendation = generate_monitoring_info(biomass, sick_spots, week)

    if report_type == 'classification_report':
        new_image_path = os.path.join(PREDICTIONS_FOLDER, 'results', 'sick_' + selected_image)
    else:  # developer_mode
        new_image_path = os.path.join(PREDICTIONS_FOLDER, 'results', 'all_' + selected_image)

    cv2.imwrite(new_image_path, img)

    return redirect(url_for('index', selected_image=selected_image, result_image=os.path.basename(new_image_path), result_image_biomass=biomass, status=status, recommendation=recommendation, selected_image_type=selected_image_type, sick_count=sick_spots, week=week, typ=typ))


@app.route('/classification_report', methods=['POST'])
def classification_report():
    return report('classification_report')

@app.route('/developer_mode', methods=['POST'])
def developer_mode():
    return report('developer_mode')

@app.route('/images/<filename>')
def send_image(filename):
    return send_from_directory(DB_IMAGE_FOLDER, filename)

@app.route('/live_images/<filename>')
def send_live_image(filename):
    return send_from_directory(LIVE_IMAGE_FOLDER, filename)

@app.route('/prediction_images/<filename>')
def send_prediction_image(filename):
    return send_from_directory(os.path.join(PREDICTIONS_FOLDER, 'results'), filename)

@app.route('/prediction_status')
def prediction_status_endpoint():
    global prediction_running, biomass_cal_running
    return jsonify({"prediction_running": prediction_running, "biomass_cal_running": biomass_cal_running})

if __name__ == '__main__':
    app.run(debug=True)
