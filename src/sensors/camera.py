try:
    import jetson.utils
    JETSON_AVAILABLE = True
except ImportError:
    JETSON_AVAILABLE = False

import cv2

cams = []

def create_camera(csi_port):
    if JETSON_AVAILABLE:
        cams.append(jetson.utils.videoSource("csi://" + str(csi_port)))
    else:
        print("Jetson utils not available, using OpenCV camera")
        cams.append(cv2.VideoCapture(csi_port if isinstance(csi_port, int) else 0))

def get_image_size(camera_id):
    if JETSON_AVAILABLE:
        return cams[camera_id].GetWidth(), cams[camera_id].GetHeight()
    else:
        ret, frame = cams[camera_id].read()
        if ret:
            return frame.shape[1], frame.shape[0]
        return 640, 480

def get_video(camera_id):
    if JETSON_AVAILABLE:
        return cv2.cvtColor(jetson.utils.cudaToNumpy(cams[camera_id].Capture()),cv2.COLOR_RGB2BGR)
    else:
        ret, frame = cams[camera_id].read()
        return frame if ret else None

def close_cameras():
    for cam in cams:
        if JETSON_AVAILABLE:
            cam.Close()
        else:
            cam.release()

if __name__ == "__main__":
    create_camera(0)

    while True:
        img = get_video(0)
        if img is not None:
            cv2.imshow("camera", img)
            cv2.waitKey(1)

    close_cameras()