import numpy as np
from sklearn.cluster import DBSCAN
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import cv2

# --------- DBSCAN params (can be overridden) ----------
DBSCAN_EPS = 2
DBSCAN_MIN_SAMPLES = 15
MIN_TARGET_SIZE = 3
MAX_TARGET_SIZE = 260
MIN_DENSITY = 0.7

# --------- Tracker params ----------
IOU_THRESHOLD = 0.3
MAX_DISAPPEARED = 5
TIME_WEIGHT_DECAY = 2.0


@dataclass
class Target:
    id: int
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    event_count: int
    avg_timestamp: float = 0.0


@dataclass
class Track:
    id: int
    color: Tuple[int, int, int]
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    history: List[Tuple[int, float, float, float, int]] = field(default_factory=list)
    disappeared: int = 0
    active: bool = True
    total_events: int = 0

    def update(self, target: Target, frame_idx: int):
        self.bbox = target.bbox
        self.centroid = target.centroid
        self.disappeared = 0
        self.total_events += target.event_count
        self.history.append((frame_idx, target.centroid[0], target.centroid[1],
                            target.avg_timestamp, target.event_count))


def compute_iou(box1, box2) -> float:
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = max(1, (x2_1 - x1_1) * (y2_1 - y1_1))
    box2_area = max(1, (x2_2 - x1_2) * (y2_2 - y1_2))
    union_area = box1_area + box2_area - inter_area

    return inter_area / max(union_area, 1e-6)


def generate_colors(n: int):
    colors = []
    for i in range(n):
        hue = int(180 * i / n)
        color = cv2.cvtColor(np.uint8([[[hue, 255, 200]]]), cv2.COLOR_HSV2BGR)[0, 0]
        colors.append(tuple(map(int, color)))
    return colors


def compute_time_weighted_centroid(pixels: np.ndarray, timestamps: np.ndarray,
                                   decay: float = TIME_WEIGHT_DECAY):
    if len(pixels) == 0:
        return (0.0, 0.0, 0.0)

    timestamps = timestamps.astype(np.float64)
    t_min, t_max = timestamps.min(), timestamps.max()

    if t_max > t_min:
        normalized_t = (timestamps - t_min) / (t_max - t_min)
        weights = np.exp(normalized_t * decay)
        weights = weights / weights.sum()

        centroid_x = np.sum(pixels[:, 0] * weights)
        centroid_y = np.sum(pixels[:, 1] * weights)
        avg_timestamp = np.sum(timestamps * weights)
    else:
        centroid_x = np.mean(pixels[:, 0])
        centroid_y = np.mean(pixels[:, 1])
        avg_timestamp = t_min

    return (float(centroid_x), float(centroid_y), float(avg_timestamp))


def detect_targets(events: np.ndarray,
                   eps: int = DBSCAN_EPS,
                   min_samples: int = DBSCAN_MIN_SAMPLES,
                   use_time_weight: bool = True) -> List[Target]:
    if len(events) < min_samples:
        return []

    X = np.column_stack((events['x'], events['y']))
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(X)

    targets: List[Target] = []
    for label in set(db.labels_):
        if label == -1:
            continue

        mask = db.labels_ == label
        cluster_pixels = X[mask]
        cluster_events = events[mask]

        x_min, y_min = np.min(cluster_pixels, axis=0)
        x_max, y_max = np.max(cluster_pixels, axis=0)
        w, h = x_max - x_min, y_max - y_min

        if w < MIN_TARGET_SIZE or h < MIN_TARGET_SIZE:
            continue
        if w > MAX_TARGET_SIZE or h > MAX_TARGET_SIZE:
            continue

        density = len(cluster_pixels) / max(1, w * h)
        if density < MIN_DENSITY:
            continue

        if use_time_weight and 't' in cluster_events.dtype.names:
            timestamps = cluster_events['t']
            cx, cy, avg_t = compute_time_weighted_centroid(cluster_pixels, timestamps, TIME_WEIGHT_DECAY)
        else:
            cx, cy = float(np.mean(cluster_pixels[:, 0])), float(np.mean(cluster_pixels[:, 1]))
            avg_t = 0.0

        targets.append(Target(
            id=int(label),
            bbox=(int(x_min), int(y_min), int(x_max), int(y_max)),
            centroid=(cx, cy),
            event_count=int(len(cluster_pixels)),
            avg_timestamp=float(avg_t)
        ))

    return targets


def events_to_color_image(events: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    if len(events) == 0:
        return img

    x = np.clip(events['x'], 0, shape[1] - 1).astype(int)
    y = np.clip(events['y'], 0, shape[0] - 1).astype(int)
    p = events['p']

    img[y[p == 0], x[p == 0]] = [255, 100, 100]
    img[y[p == 1], x[p == 1]] = [255, 255, 255]
    return img


class IOUTracker:
    def __init__(self, start_id: int = 1, color_count: int = 500):
        self.tracks: List[Track] = []
        self.next_id = start_id
        self.colors = generate_colors(color_count)

    def update(self, targets: List[Target], frame_idx: int) -> List[Track]:
        active_tracks = [t for t in self.tracks if t.active]

        if len(active_tracks) == 0:
            for target in targets:
                tr = Track(
                    id=self.next_id,
                    color=self.colors[self.next_id % len(self.colors)],
                    bbox=target.bbox,
                    centroid=target.centroid,
                    total_events=target.event_count
                )
                tr.history.append((frame_idx, target.centroid[0], target.centroid[1],
                                   target.avg_timestamp, target.event_count))
                self.tracks.append(tr)
                self.next_id += 1
            return [t for t in self.tracks if t.active]

        if len(targets) == 0:
            for tr in active_tracks:
                tr.disappeared += 1
                if tr.disappeared > MAX_DISAPPEARED:
                    tr.active = False
            return [t for t in self.tracks if t.active]

        iou_matrix = np.zeros((len(active_tracks), len(targets)), dtype=np.float64)
        for i, tr in enumerate(active_tracks):
            for j, tgt in enumerate(targets):
                iou_matrix[i, j] = compute_iou(tr.bbox, tgt.bbox)

        matched_tracks = set()
        matched_targets = set()

        while iou_matrix.size > 0:
            max_iou = float(iou_matrix.max())
            if max_iou < IOU_THRESHOLD:
                break
            i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            active_tracks[i].update(targets[j], frame_idx)
            matched_tracks.add(i)
            matched_targets.add(j)
            iou_matrix[i, :] = 0.0
            iou_matrix[:, j] = 0.0

        for i, tr in enumerate(active_tracks):
            if i not in matched_tracks:
                tr.disappeared += 1
                if tr.disappeared > MAX_DISAPPEARED:
                    tr.active = False

        for j, tgt in enumerate(targets):
            if j not in matched_targets:
                tr = Track(
                    id=self.next_id,
                    color=self.colors[self.next_id % len(self.colors)],
                    bbox=tgt.bbox,
                    centroid=tgt.centroid,
                    total_events=tgt.event_count
                )
                tr.history.append((frame_idx, tgt.centroid[0], tgt.centroid[1],
                                   tgt.avg_timestamp, tgt.event_count))
                self.tracks.append(tr)
                self.next_id += 1

        return [t for t in self.tracks if t.active]