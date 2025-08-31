import serial
import time
import csv
import pandas as pd
from datetime import datetime
import keyboard  # To detect key presses
import json
import xml.etree.ElementTree as ET

# Setup serial connection (ensure the correct COM port is used)
ser = serial.Serial('COM3', 115200)  # Replace COM3 with your actual port
time.sleep(2)  # Wait for the connection to establish

# Data collection variables
data = []  # To store the collected data
running = True  # Flag to control the recording state

# Function to collect data from the serial port
def collect_data():
    global running
    print("Recording started. Press 's' to stop.")

    while running:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            print(line)  # Print data from Arduino
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            data.append([timestamp] + line.split(','))  # Expecting: EMG,GSR
        
        # Check if 's' key is pressed to stop recording
        if keyboard.is_pressed('s'):
            running = False
            print("Recording stopped.")
            break

        time.sleep(0.2)

# Function to save the collected data to CSV, TXT, XLS, JSON, and XML
def save_data():
    if not data:
        print("No data to save.")
        return

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    try:
        # Save to CSV
        with open(f"data_{timestamp}.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'EMG', 'GSR'])
            writer.writerows(data)
        print(f"CSV saved as data_{timestamp}.csv")
    except Exception as e:
        print(f"Error saving CSV: {e}")
    
    try:
        # Save to TXT
        with open(f"data_{timestamp}.txt", 'w') as f:
            for row in data:
                f.write('\t'.join(row) + '\n')
        print(f"TXT saved as data_{timestamp}.txt")
    except Exception as e:
        print(f"Error saving TXT: {e}")
    
    try:
        # Save to Excel
        df = pd.DataFrame(data, columns=['Timestamp', 'EMG', 'GSR'])
        df.to_excel(f"data_{timestamp}.xlsx", index=False)
        print(f"Excel saved as data_{timestamp}.xlsx")
    except Exception as e:
        print(f"Error saving Excel: {e}")

    try:
        # Save to JSON
        json_data = [dict(zip(['Timestamp', 'EMG', 'GSR'], row)) for row in data]
        with open(f"data_{timestamp}.json", 'w') as f:
            json.dump(json_data, f, indent=4)
        print(f"JSON saved as data_{timestamp}.json")
    except Exception as e:
        print(f"Error saving JSON: {e}")

    try:
        # Save to XML
        root = ET.Element("Data")
        for row in data:
            entry = ET.SubElement(root, "Entry")
            ET.SubElement(entry, "Timestamp").text = row[0]
            ET.SubElement(entry, "EMG").text = row[1]
            ET.SubElement(entry, "GSR").text = row[2]
        tree = ET.ElementTree(root)
        tree.write(f"data_{timestamp}.xml")
        print(f"XML saved as data_{timestamp}.xml")
    except Exception as e:
        print(f"Error saving XML: {e}")

# Example usage
if __name__ == "__main__":
    collect_data()
    save_data()
