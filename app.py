from flask import Flask, render_template, request, send_from_directory, url_for, redirect, jsonify
import os
import subprocess
from pathlib import Path
import sqlite3
import cv2
import threading
import time
from biomass import calculate_biomass, save_green_extracted_image
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
    global prediction_running, biomass_cal_running, current_image, current_db_index, classification_image, dev_image

    source_type = request.form.get('source_type')

    # Check if prediction or biomass calculation is running
    if prediction_running.is_set() or biomass_cal_running.is_set():
        return redirect(url_for('index', result="A prediction is already in progress. Please wait."))

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
            return render_template('index.html', result="No images found in the database.")

        selected_image = image_list[current_db_index%len(image_list)]
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
            return render_template('index.html', result="Error capturing live image")

        current_image = os.path.abspath(captured_image_path)
        print(f"Captured live image: {captured_image_path}")

    # Process the selected or captured image
    threading.Thread(target=process_image).start()

    return redirect(url_for('index', selected_image=selected_image))


def process_image():
    global current_image, classification_image, dev_image, prediction_running, biomass_cal_running

    run_prediction()
    calculate_biomass()
    return
    ######################################## caching is complicated, let's skip it for now
    try:
        image_id = os.path.basename(current_image).replace('.jpg', '')

        # Check if the predictions for this image are available
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT prediction FROM predictions WHERE image_id = ?
        ''', (image_id,))
        predictions = cursor.fetchall()
        conn.close()
        
        # also check if the annotated images are available!
        # todo koushik
        # set image paths (as absolute paths)
        classification_image = None #todo koushik
        dev_image = None #todo koushik

        if predictions:
            # Predictions exist, use them directly
            print(f"Predictions for {current_image} already exist in the database.")
            handle_existing_predictions()
        else:
            # Predictions do not exist, perform prediction and biomass calculation
            print(f"No existing predictions for {current_image}. Running prediction...")
            run_prediction()
            calculate_biomass()

    finally:
        with prediction_lock:
            prediction_running = False
            biomass_cal_running = False

        print(f"Processing for {selected_image} completed.")


def handle_existing_predictions():
    """
    Handle the existing predictions for an image.
    This function can be used to process and display predictions from the database.
    """
    global current_image, classification_image
    print(f"Handling existing predictions for {image_name}.")
    for prediction in predictions:
        print(f"Prediction: {prediction}")
    
    # Optionally, update the classification image or status to reflect that it's already predicted
    global classification_image
    classification_image = image_name
    prediction_running.clear()


def run_prediction():
    global prediction_running, current_image, classification_image, dev_image
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
        load_monitoring_info()
        generate_images() # todo cals and set the annotated images


        with prediction_lock:
            classification_image = os.path.basename(image_path)
        
        with prediction_lock:
            classification_image = os.path.basename(image_path)

    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
    except sqlite3.OperationalError as e:
        print(f"Database error occurred: {e}")
    finally:
        prediction_running.clear()


def calculate_biomass():
    global biomass_cal_running, current_image
    
    image = cv2.imread(current_image)
    biomass, mask = calculate_biomass(current_image)

    biomass_file_path = os.path.join(BIOMASS_FOLDER, os.path.basename(current_image).replace('.jpg', '.txt'))
    with open(biomass_file_path, 'w') as f:
        f.write(str(biomass))

    images_folder = os.path.join(BIOMASS_FOLDER, 'images')
    os.makedirs(images_folder, exist_ok=True)

    green_image_file_path = os.path.join(images_folder, os.path.basename(current_image))
    save_green_extracted_image(image, mask, green_image_file_path)

    biomass_cal_running.clear()


def report(report_type):
    global prediction_running, biomass_cal_running, classification_image, dev_image
    
    output_image = None
    
    if prediction_running.is_set() or biomass_cal_running.is_set():
        return render_template('index.html', result="Prediction or biomass calculation is still in progress. Please wait.")
    if not current_image:
        return render_template('index.html', result=f"No image selected for {report_type.replace('_', ' ')}")
    
    if (report_type == 'classification_report'):
        if not classification_image:
            return render_template('index.html', result="No classification image available. Please run the prediction first.")
        else:
            output_image = classification_image
    elif (report_type == 'developer_mode'):
        if not dev_image:
            return render_template('index.html', result="No developer image available. Please run the prediction first.")
        else:
            output_image = dev_image
    else:
        raise ValueError(f"Invalid report type: {report_type}")

    info_json = load_monitoring_info()

    return redirect(url_for('index', result_image=output_image, display_info=info_json))


def generate_images():
    # takes global variables current_image, and sets classification_image and dev_image
    # todo koushik finish implementation
    
    image_id = os.path.basename(current_image).replace('.jpg', '')

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

        if report_type == 'classification_report':
        new_image_path = os.path.join(PREDICTIONS_FOLDER, 'results', 'sick_' + selected_image)
    else:  # developer_mode
        new_image_path = os.path.join(PREDICTIONS_FOLDER, 'results', 'all_' + selected_image)

    cv2.imwrite(new_image_path, img)
    
    classification_image = None #todo koushik
    dev_image = None #todo koushik


def load_monitoring_info():
    image_id = os.path.basename(current_image).replace('.jpg', '')  # use to access stored data
    # load all the necessary information and put together a json object to send to the frontend
    info_json = {}
    info_json["week"] = 3
    info_json["type"] = "Basil"
    info_json["status"] = None # todo koushik
    info_json["recommendation"] = None # todo koushik
    #status, recommendation = generate_monitoring_info(biomass, sick_spots, week)
    
    
    # I put a bunch of code here that has something todo with the monitoring info, please make it work
    # # Load biomass value
    # biomass_file_path = os.path.join(BIOMASS_FOLDER, image_id + '.txt')
    # if os.path.exists(biomass_file_path):
    #     with open(biomass_file_path, 'r') as f:
    #         biomass = float(f.read().strip())
    # else:
    #     biomass = 0.0

    #from biomass import get_status_and_recommendation
    # def load_monitoring_info(biomass, sick_spots, week):
    #     status, recommendation = get_status_and_recommendation(biomass, sick_spots)
    #     return status, recommendation


    return info_json


@app.route('/classification_report', methods=['POST'])
def classification_report():
    return report('classification_report')


@app.route('/developer_mode', methods=['POST'])
def developer_mode():
    return report('developer_mode')


@app.route('/images/<filepath>')
def send_image(filepath):
    return send_from_directory(filepath)


@app.route('/prediction_status')
def prediction_status_endpoint():
    global prediction_running, biomass_cal_running
    return jsonify({"prediction_running": prediction_running.is_set(), "biomass_cal_running": biomass_cal_running.is_set()})


if __name__ == '__main__':
    app.run(debug=True)
