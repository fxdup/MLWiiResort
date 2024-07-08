from .DistanceProcessor import DistanceProcessor
from .ScoreBoardProcessor import ScoreBoardProcessor
from mss import mss
from win32gui import FindWindow, GetWindowRect
import numpy as np
from .KeyboardSim import mouse, click
from time import sleep

class ActionSpace:
    ACTION_SPACE = [
        0, # No action
        1 # Swing
    ]
    def sample(self):
        return np.random.choice(ActionSpace.ACTION_SPACE, p=[0.9, 0.1])
                

class HomeRunEnv:
    DISTANCE_RECT = (470, 95, 560, 125)
    PLAY_AREA_RECT = (170, 130, 450, 395)
    SCORE_BOARD_RECT = (0, 90, 165, 130)
    def __init__(self,):
        window_handle = FindWindow(None, "Dolphin 2407 | JIT64 SC | Direct3D 11 | HLE | Wii Sports (RSPE01)")
        window_rect   = GetWindowRect(window_handle)
        self.sct = mss()
        self.action_space = ActionSpace()

        self.window_bb = {'left': window_rect[0]+10, 'top': window_rect[1]+30, 'width': window_rect[2] - window_rect[0]-20, 'height': window_rect[3] - window_rect[1]-40}
        self.distance_processor = DistanceProcessor(HomeRunEnv.DISTANCE_RECT)
        self.score_board_processor = ScoreBoardProcessor(HomeRunEnv.SCORE_BOARD_RECT)

    def step(self, action):
        self.do_action(action)

        observation = self.get_observation()
        reward = self.distance_processor.step(observation)

        done = self.score_board_processor.step(observation)

        return reward, done
    
    def do_action(self, action):
        if action == 1:
            click(mouse.middle)

    def get_observation(self):
        screenShot = self.sct.grab(self.window_bb)
        screenShot = np.array(screenShot)
        return screenShot
    

    def reset(self):
        while True:
            observation = self.get_observation()
            if not self.score_board_processor.step(observation):
                break
            click(mouse.right)
            sleep(1)