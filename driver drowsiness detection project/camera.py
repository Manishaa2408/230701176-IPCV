import cv2
import threading

class VideoCamera(object):
    def __init__(self, source=0):
        # source can be 0 for webcam, or a file path
        self.source = source
        if source == 0:
            self.video = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            self.video = cv2.VideoCapture(source)
            
        self.grabbed, self.frame = self.video.read()
        self.stopped = False
        
        # Start a thread to read frames from the video stream
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def __del__(self):
        self.stopped = True
        if self.video.isOpened():
            self.video.release()

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            if self.stopped:
                return

            grabbed, frame = self.video.read()
            if not grabbed:
                if self.source == 0:
                    # Webcam may have dropped a frame or not warmed up yet. Retry.
                    import time
                    time.sleep(0.1)
                    continue
                else:
                    # Video file reached the end. 
                    # Keep the thread alive and return the last frame to keep stream active
                    import time
                    while not self.stopped:
                        time.sleep(0.1)
                    return
                
            self.grabbed = grabbed
            self.frame = frame

    def read(self):
        # Return the most recent frame
        return self.grabbed, self.frame

    def release(self):
        self.stopped = True
        self.video.release()
