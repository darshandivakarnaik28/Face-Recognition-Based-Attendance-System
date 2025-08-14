import os
import cv2
import csv
import numpy as np
import face_recognition
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
DATABASE = 'attendance.db'
CSV_FILE = 'known_faces.csv'


# ---------- UTILS ----------
def init_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    # Create the attendance table if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (
                        name TEXT,
                        usn TEXT,
                        branch TEXT,
                        semester TEXT,
                        email TEXT,
                        address TEXT,
                        date TEXT,
                        time_in TEXT,
                        time_out TEXT
                    )''')
    conn.commit()

    # Add missing columns if any
    expected_columns = ['time_in', 'time_out']
    for col in expected_columns:
        try:
            cursor.execute(f"ALTER TABLE attendance ADD COLUMN {col}TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
    conn.commit()
    conn.close()


def load_known_faces():
    known_metadata = []  # [(name, usn, branch, semester, email, address)]
    known_encodings = []

    if not os.path.exists(CSV_FILE):
        return known_metadata, known_encodings

    with open(CSV_FILE, 'r') as f:
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

    return known_metadata,known_encodings



def save_face_encoding(name, usn, branch, semester, email, address, encoding):
    with open(CSV_FILE, 'a', newline='') as f:
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
    if record is None:
        # First entry of the day → mark attendance_in
        cursor.execute('''INSERT INTO attendance 
                          (name, usn, branch, semester, email, address, date, attendance_in, attendance_out) 
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                       (name, usn, branch, semester, email, address, date, current_time, None))
    elif record[0] is not None and record[1] is None:
        # Update attendance_out if already checked in but not checked out
        cursor.execute('''UPDATE attendance 
                          SET attendance_out=? 
                          WHERE name=? AND date=?''',
                       (current_time, name, date))

    conn.commit()
    conn.close()


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
        name = request.form['name']
        usn = request.form['usn']
        branch = request.form['branch']
        semester = request.form['semester']
        email = request.form['email']
        address = request.form['address']
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Press SPACE to capture", frame)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = face_recognition.face_locations(rgb)
                encodings = face_recognition.face_encodings(rgb, faces)
                if encodings:
                    save_face_encoding(name, usn, branch, semester, email, address, encodings[0])
                    break
                else:
                    print("No face detected. Try again.")
        cap.release()
        cv2.destroyAllWindows()
        return redirect(url_for('index'))
    return render_template('register.html')


@app.route('/attendance')
def attendance():
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
