import cv2
import time
import numpy as np
import os
import datetime

from cam_interface import Camera

from birdCam.classifier.classify import predict_bird


def frame_has_bird(frame):

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    cv2.imwrite(f"/tmp/{timestamp_str}.png", frame)

    is_top5, probs = predict_bird(f"/tmp/{timestamp_str}.png")

    is_bird = is_top5 and np.sum(probs.ravel().numpy()) > 0.25

    return is_bird


def save_frames(camera, is_bird, frames_to_save, frame_save_interval):

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save to a directory for display on the web
    save_dir_display = "motion_images"

    # Save the rest of the frames somewhere else
    save_dir_rest = "motion_images_rest"

    # Save those that didn't pass the bird classifier, just in case
    save_dir_fail = "motion_images_fail"

    os.makedirs(save_dir_display, exist_ok=True)
    os.makedirs(save_dir_rest, exist_ok=True)
    os.makedirs(save_dir_fail, exist_ok=True)

    for i in range(frames_to_save):

        frame = camera.get_frame()

        if is_bird and i == 0:
            save_dir = save_dir_display
        elif is_bird:
            save_dir = save_dir_rest
        else:
            save_dir = save_dir_fail

        cv2.imwrite(
            os.path.join(save_dir, f"motion_frame_{timestamp_str}_{i}.png"),
            frame,
        )
        time.sleep(frame_save_interval)


def detect_motion(camera):

    # Parameters
    width, height = 640, 480
    video_source = 0  # change this if you have multiple webcams
    frame_save_interval = 1  # in seconds
    frames_to_save = 5  # number of frames to save

    motion_frac = 0.1

    motion_threshold = motion_frac * width * height  # motion threshold in pixels

    cool_down_time = 5  # cool down time in seconds

    last_motion_time = (
        time.time() - cool_down_time
    )  # initialize to enable immediate capture

    frame1 = camera.get_frame()
    frame2 = camera.get_frame()

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

            # Can take a while, would be nice not to block here
            is_bird = frame_has_bird(frame2)

            save_frames(camera, is_bird, frames_to_save, frame_save_interval)

        frame1 = frame2
        frame2 = camera.get_frame()

        # Wait for FPS rate before next frame
        time.sleep(1 / 30)  # 30 FPS


if __name__ == "__main__":

    camera = Camera.get_instance()

    detect_motion(camera)
