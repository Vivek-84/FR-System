import sqlite3

# Connect to the SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('face_recognition.db')

# Create a cursor object
c = conn.cursor()

# Create the table with three columns: name, date, and time
c.execute('''
CREATE TABLE IF NOT EXISTS face_recognition_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    date TEXT NOT NULL,
    time TEXT NOT NULL
)
''')

# Commit the changes and close the connection
conn.commit()
conn.close()

print("Database and table created successfully.")
