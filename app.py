import os
import cv2
import csv
import base64
import numpy as np
import face_recognition
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, 'templates'))
DATABASE = os.path.join(BASE_DIR, 'attendance.db')
CSV_FILE = os.path.join(BASE_DIR, 'known_faces.csv')


# ---------- UTILS ----------
def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (
                        name TEXT,
                        usn TEXT,
                        branch TEXT,
                        semester TEXT,
                        email TEXT,
                        address TEXT,
                        date TEXT,
                        attendance_in TEXT,
                        attendance_out TEXT
                    )''')
    conn.commit()

    # Ensure attendance_in and attendance_out columns exist
    for col in ['attendance_in', 'attendance_out']:
        try:
            cursor.execute(f"ALTER TABLE attendance ADD COLUMN {col} TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
    conn.commit()
    conn.close()


def load_known_faces():
    known_metadata = []  # [(name, usn, branch, semester, email, address)]
    known_encodings = []

    if not os.path.exists(CSV_FILE):
        return known_metadata, known_encodings

    with open(CSV_FILE, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 134:
                continue
            name, usn, branch, semester, email, address = row[:6]
            try:
                encoding_values = [float(val) for val in row[6:] if val.strip() != '']
                if len(encoding_values) != 128:
                    continue
                encoding = np.array(encoding_values, dtype='float64')
                known_metadata.append((name, usn, branch, semester, email, address))
                known_encodings.append(encoding)
            except ValueError:
                continue

    return known_metadata, known_encodings


def save_face_encoding(name, usn, branch, semester, email, address, encoding):
    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([name, usn, branch, semester, email, address] + list(encoding))


def mark_attendance(name, usn, branch, semester, email, address):
    now = datetime.now()
    date = now.strftime('%Y-%m-%d')
    current_time = now.strftime('%H:%M:%S')

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute("SELECT attendance_in, attendance_out FROM attendance WHERE name=? AND date=?", (name, date))
    record = cursor.fetchone()
    action = "Checked In"
    if record is None:
        # First entry of the day → mark attendance_in
        cursor.execute('''INSERT INTO attendance 
                          (name, usn, branch, semester, email, address, date, attendance_in, attendance_out) 
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                       (name, usn, branch, semester, email, address, date, current_time, None))
    elif record[0] is not None and record[1] is None:
        # Update attendance_out if already checked in but not checked out
        action = "Checked Out"
        cursor.execute('''UPDATE attendance 
                          SET attendance_out=? 
                          WHERE name=? AND date=?''',
                       (current_time, name, date))
    else:
        action = "Already Recorded"

    conn.commit()
    conn.close()
    return action, current_time


def decode_image_from_request(image_data_str=None, file_storage=None):
    """Decodes an image from base64 string or uploaded file into an RGB numpy array."""
    if file_storage and file_storage.filename:
        file_bytes = np.frombuffer(file_storage.read(), np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if image_data_str:
        if ',' in image_data_str:
            image_data_str = image_data_str.split(',', 1)[1]
        try:
            img_bytes = base64.b64decode(image_data_str)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if bgr is not None:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print("Error decoding base64 image:", e)

    return None


def capture_frame_from_system_camera():
    """Fallback capture using OpenCV or Pygame camera."""
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        pass

    try:
        import pygame.camera
        import pygame.surfarray
        pygame.camera.init()
        cams = pygame.camera.list_cameras()
        if cams:
            cam = pygame.camera.Camera(cams[0], (640, 480))
            cam.start()
            img = cam.get_image()
            cam.stop()
            arr = pygame.surfarray.array3d(img).swapaxes(0, 1)
            return arr
    except Exception:
        pass

    return None


# ---------- ROUTES ----------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/clear_attendance', methods=['POST'])
def clear_attendance():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM attendance")
    conn.commit()
    conn.close()
    return redirect(url_for('records'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        usn = request.form.get('usn', '').strip()
        branch = request.form.get('branch', '').strip()
        semester = request.form.get('semester', '').strip()
        email = request.form.get('email', '').strip()
        address = request.form.get('address', '').strip()

        image_data = request.form.get('image_data')
        image_file = request.files.get('image_file')

        rgb_img = decode_image_from_request(image_data_str=image_data, file_storage=image_file)

        # Fallback to system camera if no image came through web form
        if rgb_img is None:
            rgb_img = capture_frame_from_system_camera()

        if rgb_img is None:
            return render_template(
                'register.html',
                error_message="Camera image could not be captured. Please ensure your browser has camera permission allowed, or choose an image file below."
            )

        # Face detection and encoding
        faces = face_recognition.face_locations(rgb_img)
        if not faces:
            return render_template(
                'register.html',
                error_message="No face detected in the picture. Please ensure your face is well-lit, clearly visible, and facing the camera directly."
            )

        encodings = face_recognition.face_encodings(rgb_img, faces)
        if encodings:
            save_face_encoding(name, usn, branch, semester, email, address, encodings[0])
            return render_template(
                'register.html',
                success_message=f"Student '{name}' ({usn}) registered successfully!"
            )
        else:
            return render_template(
                'register.html',
                error_message="Could not extract face features. Please try again with better lighting."
            )

    return render_template('register.html')


@app.route('/attendance')
def attendance():
    """Interactive in-browser attendance page with live video scanning."""
    return render_template('take_attendance.html')


@app.route('/api/mark_attendance', methods=['POST'])
def api_mark_attendance():
    """Receives camera snapshots from browser and marks attendance."""
    data = request.get_json() or {}
    image_data = data.get('image_data')

    rgb_img = decode_image_from_request(image_data_str=image_data)
    if rgb_img is None:
        return jsonify({'recognized': False, 'error': 'Invalid image'}), 400

    known_metadata, known_encodings = load_known_faces()
    if not known_encodings:
        return jsonify({'recognized': False, 'message': 'No registered students found in database'}), 200

    # Downscale slightly for high-speed face recognition
    small_frame = cv2.resize(rgb_img, (0, 0), fx=0.5, fy=0.5)
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    matches_found = []
    for encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
        face_distances = face_recognition.face_distance(known_encodings, encoding)

        if any(matches):
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name, usn, branch, semester, email, address = known_metadata[best_match_index]
                action, current_time = mark_attendance(name, usn, branch, semester, email, address)
                matches_found.append({
                    'name': name,
                    'usn': usn,
                    'branch': branch,
                    'semester': semester,
                    'time': current_time,
                    'action': action
                })

    return jsonify({
        'recognized': len(matches_found) > 0,
        'matches': matches_found
    })


@app.route('/attendance_desktop')
def attendance_desktop():
    """Fallback desktop OpenCV attendance window."""
    known_metadata, known_encodings = load_known_faces()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for encoding, loc in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, encoding)
            face_distances = face_recognition.face_distance(known_encodings, encoding)

            if matches:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name, usn, branch, semester, email, address = known_metadata[best_match_index]
                    mark_attendance(name, usn, branch, semester, email, address)

                    top, right, bottom, left = [v * 4 for v in loc]
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return redirect(url_for('records'))


@app.route('/records', methods=['GET'])
def records():
    date = request.args.get('date')
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    if date:
        cursor.execute("SELECT * FROM attendance WHERE date=?", (date,))
    else:
        cursor.execute("SELECT * FROM attendance")
    rows = cursor.fetchall()
    conn.close()
    return render_template('attendance.html', records=rows)


if __name__ == '__main__':
    init_db()
    app.run(debug=True)
