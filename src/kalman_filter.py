"""
kalman_filter.py -- 2D Kalman filter for tracking multiple plates
"""
import torch
import cv2
import numpy as np
from collections import namedtuple

PRED_COLOR = (0, 255, 0)     # prediction marker color (green)
Z_COLOR = (255, 165, 0)      # measurment box color    (cyan)
PRED_THICKNES = 2            # thickness of marker line
PRED_SIZE = 10               # size of marker
PRED_TYPE = 0                # A crosshair marker shape
NUM_FRAMES_TO_DISCARD = 10
OD_MODEL = torch.hub.load("ultralytics/yolov5", "custom", path='best.pt', force_reload=True)

coordinate = namedtuple("Coordinate", ["x", "y"])         # int x / int y

def get_centroid(od_box:list):
    """
    Takes a bounding box from an object detection model and finds center of box
    :param od_box: list [x1, y1, x2, y2]
    :return: coordinate (x_coordinate, y_coordinate)
    """
    x_1 = od_box[0]
    y_1 = od_box[1]
    x_2 = od_box[2]
    y_2 = od_box[3]
    x_c = int((x_1 + x_2) / 2)
    y_c = int((y_1 + y_2) / 2)
    centroid = coordinate(x_c, y_c)
    return centroid

class Plate:
    def __init__(self, id_number:int, coord:coordinate):
        """
        Given the x,y coordinate of the center of a bounding box from
        the object detection model, creates a Plate object for tracking
        :param id_number: int unique to each plate
        :param coord: tuple (x,y) center of plate
        """
        self.ID = id_number      # unique object identifier
        self.prev = coord        # coordinate at previous time
        self.prior = coord       # prediction of current location
        self.velocity_x = 0      # estimated x velocity at current time
        self.velocity_y = 0      # estimated y velocity at current time
        self.not_detected = 0    # number of frames where plate has not been detected
        # self.confidence = TODO

    def predict(self):
        """
        Predict location at next timestep.
        :return: None
        """
        new_x = self.prior.x + self.velocity_x
        new_y = self.prior.y + self.velocity_y
        self.prior = coordinate(new_x, new_y)
        # TODO: Update self.confidence

    def update(self, z:coordinate):
        """
        using a measurement from object detection model update
        velocity, previous location, and current prediction.
        Current prediction is average of measurment and predicted prior.
        :param z: coordinate (x,y) center of bounding box
        :return: None
        """
        # velocity update
        self.velocity_x = z.x - self.prev.x
        self.velocity_y = z.y - self.prev.y
        # prev update
        self.prev = z
        # prior update
        new_x = int( (self.prior.x + z.x) / 2)
        new_y = int( (self.prior.y + z.y) / 2)
        self.prior = coordinate(new_x, new_y)
        # TODO: Update self.confidence

    def match_measurment(self, z:coordinate):
        """
        checks if z coordinate (center of bounding box) matches plate
        :param z: coordinate (x,y) center of bounding box
        :return: bool if True, outputs information to console
        """
        x_match = (self.prior.x  - 5 < z.x) and (z.x < self.prior.x + 5)
        y_match = (self.prior.y  - 5 < z.y) and (z.y < self.prior.y + 5)
        return x_match and y_match

class Kalman:
    def __init__(self, model, testing=False):
        """
        Multi object tracker that uses kalman filter
        to predict location of Plate objects.
        model should return detections as [x1, y1, x2, y2, confidence, category]
        :param model: <class 'models.common.AutoShape'>
        :param testing: bool if True, outputs information to console
        """
        self.plates = []
        self.new_ID = 1
        self.testing = testing
        self.model = model

    def process_frame(self, frame):
        """
        Use object detection model to detect plates.
        Compares detection to known plates and updates the matches
        using the detection as the measurment.
        Creates new plates for unmatched detections.
        Predicts location of undected known plates without doing the
        measurement update.

        :param frame: numpy.ndarray image for object detection
        :return: frame image with predictions and measurements drawn
        :return: detected list of detected plates
        """
        detected_plates = []
        undetected_plates = []
        detections = self.model(frame).pred[0]
        for det in detections:
            x1, y1, x2, y2, conf, cat = det.numpy()
            box = [x1, y1, x2, y2]
            match = False
            center = get_centroid(box)
            for plate in self.plates:
                if plate.match_measurment(center):
                    match = True
                    plate.not_detected = 0
                    detected_plates.append(plate)
                    self.plates.remove(plate)
                    # prediction followed by kalman update
                    plate.predict()
                    plate.update(center)
                    # draw prediction marker
                    cv2.drawMarker(frame, plate.prior, PRED_COLOR, PRED_TYPE, PRED_SIZE, PRED_THICKNES)
                    break

            if not match:  # create new plate with measurement
                new_plate = Plate(self.new_ID, center)
                self.new_ID += 1
                detected_plates.append(new_plate)

            # draw measurement box
            start = coordinate(int(x1), int(y1))
            end = coordinate(int(x2), int(y2))
            cv2.rectangle(frame, start, end, Z_COLOR, 2)

        for plate in self.plates:
            plate.predict()        # predict location, no kalman update
            # draw prediction marker
            cv2.drawMarker(frame, plate.prior, PRED_COLOR, PRED_TYPE, PRED_SIZE, PRED_THICKNES)
            plate.not_detected += 1
            self.plates.remove(plate)
            if plate.not_detected < NUM_FRAMES_TO_DISCARD:
                undetected_plates.append(plate)


        self.plates = detected_plates
        self.plates.extend(undetected_plates)

        return frame, detected_plates

def loop(filepath:str, od_model):
    """
    loops over frames in a video and applies
    kalman filter to each frame in sequence.
    Displays annotated frames in new window.
    od_model <class 'models.common.AutoShape'>
    should return detections as [x1, y1, x2, y2, confidence, category]
    :param filepath: str path to video
    :param od_model: object dection model
    :return: None
    """
    kalman_filter = Kalman(od_model)
    cap = cv2.VideoCapture(filepath)
    ret, frame = cap.read()
    while ret:
        output, detected = kalman_filter.process_frame(frame)
        cv2.imshow(f"Kalman filter", output)
        key = cv2.waitKey(1)
        ret, frame = cap.read()
        if key == ord('q'):
            ret = False
    cv2.destroyAllWindows()


if __name__ == "__main__":
    file = "vid.mp4"
    loop(file, OD_MODEL)
