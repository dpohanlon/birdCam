import cv2
import time
import numpy as np
import os
import datetime

from pushbullet import Pushbullet

# Parameters
width, height = 640, 480
video_source = 0  # change this if you have multiple webcams
video_length = 5  # in seconds
video_fps = 30  # frames per second
motion_frac = 0.1

pb_api_key = ""  # your Pushbullet API key

motion_threshold = motion_frac * width * height  # motion threshold in pixels
cool_down_time = 1  # cool down time in seconds

# Initialize video capture
cap = cv2.VideoCapture(video_source)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 'mp4v' for .mp4

ret, frame1 = cap.read()
ret, frame2 = cap.read()
last_motion_time = (
    time.time() - cool_down_time
)  # initialize to enable immediate capture

pb = Pushbullet(pb_api_key)

try:
    while True:
        current_time = time.time()
        if (
            current_time - last_motion_time < cool_down_time
        ):  # skip if cool down has not passed
            continue

        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)

            if cv2.contourArea(contour) < motion_threshold:
                continue

            motion_detected = True

        if motion_detected:
            print("Motion detected!")
            last_motion_time = current_time
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join("cam_video", timestamp_str)
            os.makedirs(save_dir, exist_ok=True)

            video_file_name = f"motion_video_{timestamp_str}.mp4"
            video_file_path = os.path.join(save_dir, video_file_name)
            video_writer = cv2.VideoWriter(
                video_file_path, fourcc, video_fps, (width, height)
            )

            for _ in range(video_length * video_fps):
                ret, frame = cap.read()
                if ret:  # if frame read successfully
                    video_writer.write(frame)
                    time.sleep(1 / video_fps)

            video_writer.release()

            try:

                with open(video_file_path, "rb") as f:
                    file_data = pb.upload_file(f, video_file_name)

                pb.push_file(**file_data)
                pb.push_note(
                    "Motion detected", "Motion has been detected by the webcam."
                )

            except:

                continue

        frame1 = frame2
        ret, frame2 = cap.read()

        # Wait for FPS rate before next frame
        time.sleep(1 / video_fps)  # same as video FPS

except KeyboardInterrupt:
    print("Interrupt received, stopping...")
finally:
    cap.release()
