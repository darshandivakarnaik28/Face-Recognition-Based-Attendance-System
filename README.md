# Face-Recognition-Based-Attendance-System
It is a mini project 
he Face Recognition-Based Attendance System is an automated attendance management solution that uses computer vision and machine learning techniques to identify individuals and record their attendance without manual intervention.
The system captures the live image of a person through a webcam, detects their face, matches it with pre-registered face data, and stores their attendance in a database or file.

Unlike traditional attendance methods such as manual sign-in sheets or biometric fingerprint scanning, this system offers a contactless, fast, and accurate solution, making it ideal for use in educational institutions, workplaces, and events.

Key Working Principle

Face Registration – Each user’s facial features are captured and stored in a database along with their details such as USN, Name, Branch, and Semester.

Face Encoding – The system extracts unique facial feature vectors (numerical data) from registered images using machine learning models like dlib or FaceNet.

Real-time Detection – During attendance marking, the webcam detects faces in the live video feed.

Face Matching – The system compares detected faces with the stored encodings to find a match.

Attendance Recording – If a match is found, the system marks the person’s attendance with date and time, ensuring only one entry per person per day.

Data Storage – Attendance records are stored in CSV files or a database for easy retrieval and reporting.
