import pymysql.cursors
import pandas as pd
import os
import glob
import re
from dotenv import load_dotenv

# MySQL connection details
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST")
database = os.getenv("DB_NAME")

# Connect to the MySQL database
conn = pymysql.connect(
    host=host,
    user=user,
    password=password,
    db=database
)
cursor = conn.cursor()

# Create the clinical_notes table
cursor.execute('''
CREATE TABLE IF NOT EXISTS clinical_notes (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id CHAR(36),
    date DATETIME,
    chief_complaint TEXT,
    history_of_illness TEXT,
    social_history TEXT,
    allergies TEXT,
    medications TEXT,
    assessment TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(Id)
) ENGINE=InnoDB;
''')

# Function to parse the note file
def parse_note_file(file_path, patient_id):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    notes_data = []
    note_data = {
        "patient_id": patient_id,
        "date": None,
        "chief_complaint": None,
        "history_of_illness": None,
        "social_history": None,
        "allergies": None,
        "medications": None,
        "assessment": None
    }
    section = None

    for line in lines:
        line = line.strip()
        if line.startswith("## "):
            section = line[3:].strip().lower().replace(" ", "_")
            if section not in note_data:
                note_data[section] = None
        elif re.match(r'^\d{4}-\d{2}-\d{2}$', line):
            if note_data["date"]:
                notes_data.append(note_data.copy())
                note_data = {
                    "patient_id": patient_id,
                    "date": None,
                    "chief_complaint": None,
                    "history_of_illness": None,
                    "social_history": None,
                    "allergies": None,
                    "medications": None,
                    "assessment": None
                }
            note_data["date"] = line
        elif "Chief Complaint" in line:
            section = "chief_complaint"
        elif "History of Present Illness" in line:
            section = "history_of_illness"
        elif "Social History" in line:
            section = "social_history"
        elif "Allergies" in line:
            section = "allergies"
        elif "Medications" in line:
            section = "medications"
        elif "Assessment and Plan" in line:
            section = "assessment"
        else:
            if section:
                if section not in note_data:
                    note_data[section] = line.strip()
                else:
                    if note_data[section] is not None:
                        note_data[section] += " " + line.strip()
                    else:
                        note_data[section] = line.strip()

    notes_data.append(note_data.copy())
    return notes_data

# Path to the directory containing the note files
notes_dir = 'output/notes/'  # Update with your actual path

# Track inconsistencies
inconsistencies = []

# Loop through each text file and import into the clinical_notes table
for file_path in glob.glob(os.path.join(notes_dir, '*.txt')):
    # Extract patient ID from file name
    file_name = os.path.basename(file_path)
    match = re.search(r'_(\w{8}-\w{4}-\w{4}-\w{4}-\w{12})\.txt$', file_name)
    if match:
        patient_id = match.group(1)
        notes_data = parse_note_file(file_path, patient_id)
        for note_data in notes_data:
            cursor.execute('''
                INSERT INTO clinical_notes (patient_id, date, chief_complaint, history_of_illness, social_history, allergies, medications, assessment)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ''', (
                note_data["patient_id"],
                note_data["date"],
                note_data["chief_complaint"],
                note_data["history_of_illness"],
                note_data["social_history"],
                note_data["allergies"],
                note_data["medications"],
                note_data["assessment"]
            ))
    else:
        inconsistencies.append(f"File name {file_name} does not match the expected pattern")

# Commit the changes
conn.commit()

# Close the connection
conn.close()

# Print inconsistencies if any
if inconsistencies:
    print("Data Inconsistencies Found:")
    for inconsistency in inconsistencies:
        print(inconsistency)
else:
    print("No data inconsistencies found.")