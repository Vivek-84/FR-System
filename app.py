from flask import Flask, render_template, request, jsonify, redirect, url_for, Response
import face_recognition
import cv2
import numpy as np
import os
import sqlite3
from datetime import datetime
import io
import csv

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'

def load_images():
    known_face_encodings = []
    known_face_names = []
    image_paths = []

    for image_name in os.listdir(app.config['UPLOAD_FOLDER']):
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)
        if face_encoding:
            known_face_encodings.append(face_encoding[0])
            known_face_names.append(os.path.splitext(image_name)[0])  # Remove file extension
            image_paths.append(image_path)

    return known_face_encodings, known_face_names, image_paths

known_face_encodings, known_face_names, image_paths = load_images()

def log_to_database(name):
    conn = sqlite3.connect('face_recognition.db')
    c = conn.cursor()
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')
    c.execute("INSERT INTO face_recognition_logs (name, date, time) VALUES (?, ?, ?)",
              (name, date_str, time_str))
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    try:
        # Read and convert image
        img_arr = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Invalid image format"}), 400

        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Find face locations
        face_locations = face_recognition.face_locations(rgb_img)
        if not face_locations:
            return jsonify({"error": "No faces detected"}), 400

        # Get face encodings for all faces in the image
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
        if not face_encodings:
            return jsonify({"error": "Could not extract face encodings"}), 400

        matches = []
        for face_encoding in face_encodings:
            # Compare with known faces
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            
            # Use a threshold for face recognition (0.6 is typical)
            if face_distances[best_match_index] < 0.6:
                match_name = known_face_names[best_match_index]
                matches.append({
                    "name": match_name,
                    "image_path": image_paths[best_match_index],
                    "confidence": float(1 - face_distances[best_match_index])
                })
                log_to_database(match_name)
            else:
                matches.append({"name": "Unknown", "confidence": 0})

        return jsonify({"matches": matches})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and file.filename.lower().endswith(('jpg', 'jpeg', 'png')):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        global known_face_encodings, known_face_names, image_paths
        known_face_encodings, known_face_names, image_paths = load_images()
        return redirect(url_for('index'))
    return redirect(request.url)

@app.route('/view_logs')
def view_logs():
    conn = sqlite3.connect('face_recognition.db')
    c = conn.cursor()
    c.execute("SELECT * FROM face_recognition_logs ORDER BY date DESC, time DESC")
    logs = c.fetchall()
    conn.close()
    return render_template('view_logs.html', logs=logs)

@app.route('/clear_logs', methods=['POST'])
def clear_logs():
    try:
        conn = sqlite3.connect('face_recognition.db')
        c = conn.cursor()
        c.execute("DELETE FROM face_recognition_logs")
        conn.commit()
        conn.close()
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/export_logs')
def export_logs():
    conn = sqlite3.connect('face_recognition.db')
    c = conn.cursor()
    c.execute("SELECT * FROM face_recognition_logs ORDER BY date DESC, time DESC")
    logs = c.fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Name', 'Date', 'Time'])

    for log in logs:
        writer.writerow(log)

    output.seek(0)

    return Response(
        output,
        mimetype='text/csv',
        headers={"Content-Disposition": "attachment;filename=face_recognition_logs.csv"}
    )

if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # Initialize database if it doesn't exist
    if not os.path.exists('face_recognition.db'):
        conn = sqlite3.connect('face_recognition.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS face_recognition_logs
                     (name TEXT, date TEXT, time TEXT)''')
        conn.commit()
        conn.close()
    
    app.run(debug=True)