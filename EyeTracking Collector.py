import cv2
import pandas as pd
from datetime import datetime

def adjust_contrast(frame, alpha=1.7):
    """
    Adjusts the contrast of the frame. Alpha > 1 increases contrast, Alpha < 1 decreases contrast.
    """
    new_frame = cv2.convertScaleAbs(frame, alpha=alpha)
    return new_frame

def main():
    cap = cv2.VideoCapture(1)  # Adjusted to use camera index 2

    # Set resolution to 1280x720 (HD)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Reduce frame rate to 30 fps for testing
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Using MP4 codec
    out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (1280, 720))

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'r' to start/stop recording. Press 'q' to quit.")
    recording = False
    frame_counter = 0
    data_records = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Adjust the contrast of the frame
        frame = adjust_contrast(frame)

        # Display the resulting frame
        cv2.imshow('Webcam Stream with Contrast Adjustment', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):  # Toggle recording with 'r'
            recording = not recording
            if recording:
                print("Recording started.")
            else:
                out.release()
                print("Recording stopped. Video saved as 'output.mp4'.")
        
        if recording:
            out.write(frame)  # Write the frame to the file if recording
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            data_records.append({'Timestamp': timestamp, 'Frame': frame_counter})
            print(f"Timestamp: {timestamp}, Frame: {frame_counter}")  # Print timestamp and frame number
            frame_counter += 1

        if key == ord('q'):  # Press 'q' to quit
            break

    # Release everything when job is finished
    cap.release()
    if recording:
        out.release()
    cv2.destroyAllWindows()

    # Save data to CSV, TXT, and XLSX
    df = pd.DataFrame(data_records)
    df.to_csv('frame_data.csv', index=False)
    df.to_excel('frame_data.xlsx', index=False)
    df.to_csv('frame_data.txt', sep='\t', index=False)
    print("Data saved to files.")

if __name__ == '__main__':
    main()
