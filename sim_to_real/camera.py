import cv2


class Camera():
    def __init__(self, cam_index=0):
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # Set the framerate to 30 FPS
    def capture_img(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture image from camera.")

        # frame = cv2.resize(frame, (64, 64))
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        return frame     
    
if __name__=="__main__":
    camera = Camera()
    while True:
        img = camera.capture_img()
        cv2.imshow("Camera Feed", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.cap.release()
    cv2.destroyAllWindows()