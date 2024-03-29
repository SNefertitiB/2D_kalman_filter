"""
kalman_filter.py -- 2D Kalman filter for tracking multiple plates
"""
import torch
import cv2
import numpy as np
from collections import namedtuple

PRED_COLOR = (0, 255, 0)     # prediction marker color (green)
Z_COLOR = (255, 165, 0)      # measurement box color    (cyan)
PRED_THICKNESS = 2            # thickness of marker line
PRED_SIZE = 10               # size of marker
PRED_TYPE = 0                # A crosshair marker shape
PERCENT_TO_DISCARD = 0.02
PREDICTION_CONFIDENCE = 0.7   # note: 0.95 * 0.7^(10) = 0.026
OD_MODEL = torch.hub.load("ultralytics/yolov5", "custom", path='best.pt', force_reload=True)

Coordinate = namedtuple("Coordinate", ["x", "y"])         # int x / int y
BoundingBox = namedtuple("BoundingBox", ['x1', 'y1', 'x2', 'y2'])   # box = [x1, y1, x2, y2]

def get_centroid(od_box: BoundingBox) -> Coordinate:
    """
    Takes a bounding box from an object detection model and finds center of box
    :param od_box: BoundingBox (x1, y1, x2, y2)
    :return: Coordinate of centroid (x_coordinate, y_coordinate)
    """
    # x_1 = od_box[0]
    # y_1 = od_box[1]
    # x_2 = od_box[2]
    # y_2 = od_box[3]
    x_c = int((od_box.x1 + od_box.x2) / 2)
    y_c = int((od_box.y1 + od_box.y2) / 2)
    centroid = Coordinate(x_c, y_c)
    return centroid


class Plate:
    def __init__(self, id_number: int, coord: Coordinate, confidence: float):
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
        self.confidence = confidence  # model confidence in location

    def predict(self):
        """
        Predict location at next timestep.
        :return: None
        """
        new_x = self.prior.x + self.velocity_x
        new_y = self.prior.y + self.velocity_y
        self.prior = Coordinate(new_x, new_y)
        self.confidence *= PREDICTION_CONFIDENCE

    def update(self, z: Coordinate, confidence: float):
        """
        using a measurement from object detection model update
        velocity, previous location, and current prediction.
        Current prediction is average of measurment and predicted prior.
        :param z: Coordinate (x,y) center of bounding box
        :param confidence: float model's confidence in the detection
        :return: None
        """
        # velocity update
        self.velocity_x = z.x - self.prev.x
        self.velocity_y = z.y - self.prev.y
        # prev update
        self.prev = z
        # prior update
        new_x = int((self.prior.x + z.x) / 2)
        new_y = int((self.prior.y + z.y) / 2)
        self.prior = Coordinate(new_x, new_y)
        self.confidence = confidence          # model confidence in location

    def match_measurment(self, z: Coordinate):
        """
        checks if z coordinate (center of bounding box) matches plate
        :param z: coordinate (x,y) center of bounding box
        :return: bool if True, outputs information to console
        """
        x_match = (self.prior.x - 5 < z.x) and (z.x < self.prior.x + 5)
        y_match = (self.prior.y - 5 < z.y) and (z.y < self.prior.y + 5)
        return x_match and y_match

    def is_in_box(self, box:BoundingBox)->bool:
        """
        Checks to see if the current prediction is inside the passed bounding box
        :param box: BoundingBox [x1, y1, x2, y2]
        :return: True if prediction is inside box
        """
        x_test = box.x1 - 5 <= self.prior.x <= box.x2 + 5
        y_test = box.y1 - 5 <= self.prior.y <= box.y2 + 5
        if x_test and y_test:
            return True
        else:
            return False

    def get_distance(self, centroid:Coordinate) -> int:
        x_dif = self.prior.x - centroid.x
        y_dif = self.prior.y - centroid.y
        d = np.sqrt(x_dif**2 + y_dif**2)
        return int(d)



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
        detected_plates = []
        detections = self.model(frame).pred[0]
        for det in detections:
            x1, y1, x2, y2, conf, cat = det.numpy()
            # draw box measurement bounding box
            start = Coordinate(int(x1), int(y1))
            end = Coordinate(int(x2), int(y2))
            cv2.rectangle(frame, start, end, Z_COLOR, 2)
            # match to plates
            min_dist = 9999
            bounding_box = BoundingBox(x1, y1, x2, y2)
            center = get_centroid(bounding_box)
            match = None
            for plate in self.plates:
                if plate.is_in_box(bounding_box):
                    self.plates.remove(plate)
                    d = plate.get_distance(center)
                    if d < min_dist:
                        min_dist = d
                        match = plate
            if match is not None:
                # kalman update
                match.update(center, conf)
                # draw prediction
                cv2.drawMarker(frame, plate.prior, PRED_COLOR, PRED_TYPE, PRED_SIZE, PRED_THICKNESS)
                # predict for next frame
                plate.predict()
                detected_plates.append(plate)
            else: # no match then initialize
                new_plate = Plate(self.new_ID, center, conf)
                self.new_ID += 1
                detected_plates.append(new_plate)

        plates = detected_plates
        for plate in self.plates:
            self.plates.remove(plate)
            if plate.confidence >= PERCENT_TO_DISCARD:
                # draw prediction
                cv2.drawMarker(frame, plate.prior, PRED_COLOR, PRED_TYPE, PRED_SIZE, PRED_THICKNESS)
                # predict for next frame
                plate.predict()
                plates.append(plate)

        self.plates = plates
        print(len(self.plates))
        return frame, detected_plates

    # def process_frame(self, frame):
    #     """
    #     Use object detection model to detect plates.
    #     Compares detection to known plates and updates the matches
    #     using the detection as the measurment.
    #     Creates new plates for unmatched detections.
    #     Predicts location of undected known plates without doing the
    #     measurement update.
    #
    #     :param frame: numpy.ndarray image for object detection
    #     :return: frame image with predictions and measurements drawn
    #     :return: detected list of detected plates
    #     """
    #     detected_plates = []
    #     detections = self.model(frame).pred[0]
    #     for det in detections:
    #         x1, y1, x2, y2, conf, cat = det.numpy()
    #         box = [x1, y1, x2, y2]
    #         match = False
    #         center = get_centroid(box)
    #         for plate in self.plates:
    #             if plate.match_measurment(center):
    #                 match = True
    #                 detected_plates.append(plate)
    #                 self.plates.remove(plate)
    #                 # prediction followed by kalman update
    #                 plate.predict()
    #                 plate.update(center, conf)
    #                 # draw prediction marker
    #                 cv2.drawMarker(frame, plate.prior, PRED_COLOR, PRED_TYPE, PRED_SIZE, PRED_THICKNESS)
    #                 break
    #
    #         if not match:  # create new plate with measurement
    #             new_plate = Plate(self.new_ID, center, conf)
    #             self.new_ID += 1
    #             detected_plates.append(new_plate)
    #
    #         # draw measurement box
    #         start = Coordinate(int(x1), int(y1))
    #         end = Coordinate(int(x2), int(y2))
    #         cv2.rectangle(frame, start, end, Z_COLOR, 2)
    #
    #     plates = detected_plates
    #     for plate in self.plates:
    #         plate.predict()        # predict location, no kalman update
    #         if plate.confidence >= PERCENT_TO_DISCARD:
    #             # draw prediction marker
    #             cv2.drawMarker(frame, plate.prior, PRED_COLOR, PRED_TYPE, PRED_SIZE, PRED_THICKNESS)
    #             plates.append(plate)
    #         self.plates.remove(plate)
    #
    #     self.plates = plates
    #
    #     return frame, detected_plates


def loop(filepath: str, od_model):
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
