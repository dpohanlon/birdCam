import cv2
import queue
import threading
import atexit

# SPMC camera interface

class Camera:
    instance = None  # Default to no instance

    @classmethod
    def get_instance(cls):
        if cls.instance is None:
            cls.instance = cls()
        return cls.instance

    def __init__(self):
        self.frame_queue = queue.Queue()
        self.queue_lock = threading.Lock()

        # Add a condition for the capture loop
        self.running = True

        self.capture_thread = threading.Thread(target=self.capture_frames, args=(0,))
        self.capture_thread.daemon = True  # Set the thread as a daemon so it will exit when the main program does
        self.capture_thread.start()

        # Register the cleanup function to be called on exit
        atexit.register(self.cleanup)

    def capture_frames(self, camera_port):
        cap = cv2.VideoCapture(camera_port)
        while self.running:  # Only run while self.running is True
            ret, frame = cap.read()
            with self.queue_lock:
                if not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()   # discard previous frame
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame)

    def get_frame(self):
        # try to get a frame without blocking
        frame = None
        with self.queue_lock:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()

        # if no frame was available, wait for one
        if frame is None:
            frame = self.frame_queue.get()  # this will block until a frame is available

        return frame

    def cleanup(self):
        # This function will be called when the application exits
        # It stops the capture thread and waits for it to finish
        self.running = False
        if self.capture_thread.is_alive():
            self.capture_thread.join()


if __name__ == '__main__':

    camera = Camera.get_instance()

    frame1 = camera.get_frame()

    print(frame1)
