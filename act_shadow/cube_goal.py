"""Completion evidence for one orange cube and the fixed overhead SO101 camera."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


# Coordinates at half the capture resolution. The reference is a crop of the
# empty tray. Locate independent sections of the rim again in each observation.
ORIGIN = (39, 6)
RIM = np.array([(83, 21), (155, 28), (139, 89), (67, 74)], np.int32)
PATCHES = ((78, 14, 117, 39), (125, 25, 164, 91), (65, 51, 102, 86))
MAX_GAP = .32
SETTLE_SECONDS = .60


class CubeInBox:
    def __init__(self):
        self.reference = cv2.imread(str(Path(__file__).with_name("empty_tray.png")), cv2.IMREAD_GRAYSCALE)
        if self.reference is None or self.reference.shape != (105, 134):
            raise RuntimeError("缺少顶部相机的空盒参考图")
        self.kernel = np.ones((3, 3), np.uint8)
        self.low = cv2.erode(self.reference, self.kernel, iterations=2).astype(np.int16)
        self.high = cv2.dilate(self.reference, self.kernel, iterations=2).astype(np.int16)
        self.last = None
        self.armed = False
        self.outside_since = None
        self.last_inside = None
        self.entered = None
        self.minimum_grip = None
        self.release_time = None
        self.stable = deque()

    def _tray(self, gray):
        offsets = []
        for x, y, x2, y2 in PATCHES:
            patch = self.reference[y-ORIGIN[1]:y2-ORIGIN[1], x-ORIGIN[0]:x2-ORIGIN[0]]
            left, top = x-31, max(0, y-23)
            crop = gray[top:y2+24, left:x2+32]
            _, score, _, (cx, cy) = cv2.minMaxLoc(cv2.matchTemplate(crop, patch, cv2.TM_CCOEFF_NORMED))
            if score >= .63:
                offsets.append(np.array([cx+left-x, cy+top-y]))
        for offset in offsets:
            agreeing = [other for other in offsets if max(abs(other-offset)) <= 7]
            if len(agreeing) >= 2:
                return np.rint(np.mean(agreeing, axis=0)).astype(int)
        return None

    def _cube(self, small):
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, (0, 150, 145), (7, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        _, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
        candidates = []
        for label, ((x, y, w, h, area), center) in enumerate(zip(stats[1:], centers[1:]), 1):
            if (170 <= area <= 900 and 13 <= w <= 39 and 13 <= h <= 39 and
                    area >= .48*w*h and 37 < center[0] < 280 and 14 < center[1] < 200):
                candidates.append((center, label))
        if len(candidates) == 1:
            center, label = candidates[0]
            return center, (labels == label).astype(np.uint8)
        return None

    def _clear(self, gray, center, cube_mask, offset):
        # Compare the area around the cube (including the approach side of the
        # tray) to the empty scene. A gripper over the cube obscures this ring.
        x, y = ORIGIN + offset
        h, w = self.reference.shape
        if x < 0 or y < 0 or gray.shape[1] < x+w or gray.shape[0] < y+h:
            return False
        cx, cy = np.rint(center).astype(int)
        yy, xx = np.ogrid[y:y+h, x:x+w]
        ring = ((xx-cx)**2 + (yy-cy)**2 <= 34**2)
        ring &= cv2.dilate(cube_mask[y:y+h, x:x+w], self.kernel, iterations=4) == 0
        if np.count_nonzero(ring) < 1150:
            return False
        difference = gray[y:y+h, x:x+w].astype(np.int16) - self.reference
        values = difference[ring]
        shift = np.median(values)
        aligned = gray[y:y+h, x:x+w].astype(np.int16) - int(shift)
        changed = (aligned < self.low-16) | (aligned > self.high+16)
        return abs(shift) < 23 and np.mean(changed[ring]) < .085

    def _reset_deposit(self):
        self.last_inside = self.entered = self.minimum_grip = self.release_time = None
        self.stable.clear()

    def update(self, stamp, front, gripper):
        if self.last is not None:
            if stamp <= self.last:
                return None
            if stamp-self.last > MAX_GAP:
                self._reset_deposit()
                self.outside_since = None
        self.last = stamp
        if self.last_inside is not None and stamp-self.last_inside > 2.:
            self._reset_deposit()
        small = cv2.resize(front, (320, 240), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        offset = self._tray(gray)
        cube = self._cube(small)
        if cube is None or offset is None:
            self.stable.clear()
            self.outside_since = None
            return None
        center, mask = cube
        polygon = (RIM + offset).astype(np.int32)
        distance = cv2.pointPolygonTest(polygon, tuple(center), True)
        if distance < -9:
            self._reset_deposit()
            if self.outside_since is None:
                self.outside_since = stamp
            if stamp - self.outside_since >= .15:
                self.armed = True
            return None
        self.outside_since = None
        if not self.armed or distance < 1:
            self.stable.clear()
            return None
        self.last_inside = stamp
        if self.minimum_grip is None:
            self.entered, self.minimum_grip = stamp, gripper
        self.minimum_grip = min(gripper, self.minimum_grip)
        if self.release_time is None and gripper-self.minimum_grip >= 5:
            self.release_time = stamp
        tray_mask = np.zeros_like(mask)
        cv2.fillConvexPoly(tray_mask, polygon, 1)
        if (self.release_time is None or cv2.countNonZero(mask & tray_mask) < .82*cv2.countNonZero(mask)
                or not self._clear(gray, center, mask, offset)):
            self.stable.clear()
            return None
        if self.stable and any(np.linalg.norm(center-prev) > 3.5 for _, prev in self.stable):
            self.stable.clear()
        self.stable.append((stamp, center))
        if len(self.stable) >= 4 and stamp-self.stable[0][0] >= SETTLE_SECONDS:
            return dict(reason="cube_in_box", frame_s=stamp, entered_s=self.entered,
                        release_s=self.release_time, still_s=round(stamp-self.stable[0][0], 3),
                        top_center_px=np.rint(center*2).astype(int).tolist())


class GoalMonitor:
    """Process at most one recent observation in the background; never queue video."""
    def __init__(self):
        self.detector = CubeInBox()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.submitted = -float("inf")

    def check(self, now, observation):
        if self.pending is not None and self.pending.done():
            result = self.pending.result()
            self.pending = None
            if result and 0 <= now-result["frame_s"] <= .25:
                return result
        if observation is not None and self.pending is None:
            header, front, _ = observation
            stamp = header["frame_time"][0]
            if stamp-self.submitted >= .09 and 0 <= now-stamp <= .15:
                self.submitted = stamp
                self.pending = self.pool.submit(self.detector.update, stamp, front, float(header["state"][-1]))
        return None

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
