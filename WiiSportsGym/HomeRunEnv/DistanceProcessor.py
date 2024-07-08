import numpy as np
import cv2
import pytesseract

class DistanceProcessor:
    def __init__(self, distance_bbox):
        self.distance = None
        self.custom_oem_psm_config = r'--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789'

        self.last_mean = 0
        self.last_dist_imgs = np.zeros((60, 180, 3), dtype=np.uint8)
        self.distance_bbox = distance_bbox

    def step(self, observation):
        observation = observation[self.distance_bbox[1]:self.distance_bbox[3], self.distance_bbox[0]:self.distance_bbox[2]]
        new_dist = self.get_distance(observation)
        self.distance = new_dist if new_dist else self.distance
        return new_dist if new_dist else 0
    

    def get_distance(self,dist_img):
        dist_img = cv2.resize(dist_img[:,:,:3], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        lower = (22, 93, 0)
        upper = (55, 255, 255)
        mask = cv2.inRange(dist_img, lower, upper)
        dist_img = cv2.bitwise_and(dist_img, dist_img, mask=mask)
        dist_img = cv2.cvtColor(dist_img, cv2.COLOR_BGR2GRAY)
        new_dist = None
        if dist_img.mean() < 5 and self.last_mean > 5:
            text = pytesseract.image_to_string(self.last_dist_imgs.mean(axis=-1, dtype=np.uint8), config=self.custom_oem_psm_config)
            if text:
                new_dist = int(text)
        self.last_mean = dist_img.mean()
        self.last_dist_imgs = np.roll(self.last_dist_imgs, 1, axis=2)
        self.last_dist_imgs[:,:,0] = dist_img
        return new_dist