import cv2
import numpy as np

class ScoreBoardProcessor:
    BALL_MARKER_PATH = "WiiSportsGym/HomeRunEnv/ball_marker.png"
    EMPTY_MARKER_PATH = "WiiSportsGym/HomeRunEnv/empty_marker.png"
    HOME_RUN_MARKER_PATH = "WiiSportsGym/HomeRunEnv/home_run_marker.png"

    def __init__(self, score_board_rect):
        self._ball_marker = cv2.imread(ScoreBoardProcessor.BALL_MARKER_PATH)
        self._empty_marker = cv2.imread(ScoreBoardProcessor.EMPTY_MARKER_PATH)
        self._home_run_marker = cv2.imread(ScoreBoardProcessor.HOME_RUN_MARKER_PATH)
        self.score_board_rect = score_board_rect

    def step(self,observation):
        observation = observation[self.score_board_rect[1]:self.score_board_rect[3], self.score_board_rect[0]:self.score_board_rect[2]]
        result_ball = cv2.matchTemplate(observation[...,:3], self._ball_marker, cv2.TM_CCOEFF_NORMED)   
        result_ball[np.isnan(result_ball)] = 0
        result_ball[np.isinf(result_ball)] = 0
        locations_ball = np.array(np.nonzero(result_ball >= 0.90))

        result_empty = cv2.matchTemplate(observation[...,:3], self._empty_marker, cv2.TM_CCOEFF_NORMED)
        result_empty[np.isnan(result_empty)] = 0
        result_empty[np.isinf(result_empty)] = 0
        locations_empty = np.array(np.nonzero(result_empty >= 0.65))

        result_home_run = cv2.matchTemplate(observation[...,:3], self._home_run_marker, cv2.TM_CCOEFF_NORMED)
        result_home_run[np.isnan(result_home_run)] = 0
        result_home_run[np.isinf(result_home_run)] = 0
        locations_home_run = np.array(np.nonzero(result_home_run >= 0.90))

        if locations_ball.shape[1] > 0:
            w, h = self._ball_marker.shape[1], self._ball_marker.shape[0]
            bounding_boxes = locations_ball.T[:,[1,0,1,0]] + np.array([0,0,w,h])
            
            for box in bounding_boxes:
                cv2.rectangle(observation, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        if locations_empty.shape[1] > 0:
            w, h = self._empty_marker.shape[1], self._empty_marker.shape[0]
            bounding_boxes = locations_empty.T[:,[1,0,1,0]] + np.array([0,0,w,h])

            for box in bounding_boxes:
                cv2.rectangle(observation, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

        if locations_home_run.shape[1] > 0:
            w, h = self._home_run_marker.shape[1], self._home_run_marker.shape[0]
            bounding_boxes = locations_home_run.T[:,[1,0,1,0]] + np.array([0,0,w,h])

            for box in bounding_boxes:
                cv2.rectangle(observation, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

        return locations_home_run.shape[1] < 1 and locations_ball.shape[1] < 1 and locations_empty.shape[1] < 1