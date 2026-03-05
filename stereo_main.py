import os
import json
import numpy as np
import cv2
from collections import deque
from typing import List, Tuple, Dict

from stereo_tracking import (
    IOUTracker, detect_targets, events_to_color_image, Track
)
from stereo_matching import (
    fundamental_from_krt, Observation,
    EpipolarGatedMatcher, PairAccumulator
)

# ================= 参数配置 =================
INPUT_RAW_LEFT = r"E:\EVS\Date\1.18chuanjia\L\toreconL.raw"
INPUT_RAW_RIGHT = r"E:\EVS\Date\1.18chuanjia\L\toreconR.raw"

OUTPUT_JSON = "auto_locked_pairs_and_3d.json"

STEP_US = 200
WINDOW_US = 1000
BUFFER_LEN = WINDOW_US // STEP_US

# 你原来的偏移：右相机早 960us => 左需要跳过 960us 或右延后 960us
LEFT_START_OFFSET_US = 960
RIGHT_START_OFFSET_US = 0

IMG_WIDTH = 1280
IMG_HEIGHT = 720

# 极线 gating（你说大概 10 px 内）
EPIPOLAR_THRESH_PX = 10.0

# 匹配确认（200 目标/帧，建议锁定阈值略高一点）
LOCK_SCORE_THRESH = 10.0         # 累积多少“命中”才锁定
UNLOCK_SCORE_THRESH = 2.5        # 低于则解锁
PAIR_DECAY = 0.985               # 每帧衰减（太快会锁不住，太慢会粘住）
INC_PER_HIT = 1.0
DEC_PER_MISS = 0.25

# 匹配代价权重
W_EPI = 0.80
W_SIZE = 0.08
W_EVT = 0.12
MAX_ASSIGN_COST = 0.90           # 越小越严格（推荐 0.8~0.95 试）

# 在参数区新增
OUTPUT_JSONL = "stereo_3d_tracks.jsonl"
AUTO_EXPORT_JSONL = True   # 想关就 False

# ================= 相机参数 =================
K_LEFT = np.array([
    [1.70372598e+03, 0.00000000e+00, 6.40357597e+02],
    [0.00000000e+00, 1.71382295e+03, 3.60946936e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

K_RIGHT = np.array([
    [1.76476538e+03, 0.00000000e+00, 6.40788642e+02],
    [0.00000000e+00, 1.76258086e+03, 3.60092239e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])

R = np.array([
    [0.99687021, 0.00301957, 0.07899791],
    [-0.00409707, 0.99990073, 0.01348117],
    [-0.07894936, -0.01376264, 0.99678362]
])

T = np.array([-148.35148614, -2.12208858, 22.07562686])  # mm


# ================= 三角化（可选） =================
class StereoTriangulator:
    def __init__(self, K_left, K_right, R, T):
        self.P_left = K_left @ np.hstack([np.eye(3), np.zeros((3, 1))])
        self.P_right = K_right @ np.hstack([R, T.reshape(3, 1)])

    def triangulate_point(self, pt_left: Tuple[float, float], pt_right: Tuple[float, float]) -> np.ndarray:
        pts_left = np.array([[pt_left[0], pt_left[1]]], dtype=np.float64)
        pts_right = np.array([[pt_right[0], pt_right[1]]], dtype=np.float64)
        points_4d = cv2.triangulatePoints(self.P_left, self.P_right, pts_left.T, pts_right.T)
        point_3d = points_4d[:3, 0] / points_4d[3, 0]
        return point_3d


def tracks_to_observations(tracks: List[Track]) -> List[Observation]:
    obs = []
    for t in tracks:
        if t.disappeared > 0:
            continue
        # last history has avg_timestamp, event_count
        if t.history:
            _, _, _, avg_ts, evt_cnt = t.history[-1]
        else:
            avg_ts, evt_cnt = 0.0, 0
        obs.append(Observation(
            track_id=t.id,
            centroid=(float(t.centroid[0]), float(t.centroid[1])),
            bbox=t.bbox,
            event_count=int(evt_cnt),
            avg_timestamp=float(avg_ts)
        ))
    return obs


def draw_tracks(img, tracks: List[Track], prefix: str):
    for t in tracks:
        if t.disappeared > 0:
            continue
        x1, y1, x2, y2 = t.bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), t.color, 2)
        cv2.putText(img, f"{prefix}{t.id}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, t.color, 2)
        cx, cy = int(t.centroid[0]), int(t.centroid[1])
        cv2.circle(img, (cx, cy), 3, (0, 255, 255), -1)


def main():
    try:
        from metavision_core.event_io import EventsIterator
    except ImportError:
        print("需要安装 metavision_core (Prophesee SDK)")
        return

    if not os.path.exists(INPUT_RAW_LEFT) or not os.path.exists(INPUT_RAW_RIGHT):
        print("raw 文件不存在")
        return

    # Build F for epipolar gating (unrectified)
    F = fundamental_from_krt(K_LEFT, K_RIGHT, R, T)
    print("[匹配] Fundamental matrix F computed.")

    matcher = EpipolarGatedMatcher(
        F=F,
        epipolar_thresh_px=EPIPOLAR_THRESH_PX,
        w_epi=W_EPI, w_size=W_SIZE, w_evt=W_EVT,
        max_cost=MAX_ASSIGN_COST
    )

    accumulator = PairAccumulator(
        lock_score_thresh=LOCK_SCORE_THRESH,
        unlock_score_thresh=UNLOCK_SCORE_THRESH,
        inc_per_hit=INC_PER_HIT,
        dec_per_miss=DEC_PER_MISS,
        decay=PAIR_DECAY
    )

    tracker_left = IOUTracker(start_id=1)
    tracker_right = IOUTracker(start_id=1)

    triangulator = StereoTriangulator(K_LEFT, K_RIGHT, R, T)

    mv_left = EventsIterator(input_path=INPUT_RAW_LEFT, delta_t=STEP_US)
    mv_right = EventsIterator(input_path=INPUT_RAW_RIGHT, delta_t=STEP_US)

    buffer_left = deque()
    buffer_right = deque()

    skip_frames = max(LEFT_START_OFFSET_US, RIGHT_START_OFFSET_US) // STEP_US

    paused = False
    frame_idx = 0

    win = "Stereo Auto Cluster Matching (Unrectified)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    locked_history_out: List[Dict] = []
    jsonl_fp = None
    if AUTO_EXPORT_JSONL:
        jsonl_fp = open(OUTPUT_JSONL, "a", encoding="utf-8")
        print(f"[导出] 将持续写入3D点到: {OUTPUT_JSONL}")

    print("\n键盘操作:")
    print("  SPACE: 暂停/继续")
    print("  S    : 保存当前锁定配对+3D样例到JSON")
    print("  Q    : 退出\n")

    for chunk_left, chunk_right in zip(mv_left, mv_right):
        if frame_idx < skip_frames:
            frame_idx += 1
            continue

        buffer_left.append(chunk_left if len(chunk_left) > 0 else np.array([], dtype=chunk_left.dtype))
        buffer_right.append(chunk_right if len(chunk_right) > 0 else np.array([], dtype=chunk_right.dtype))

        if len(buffer_left) > BUFFER_LEN:
            buffer_left.popleft()
        if len(buffer_right) > BUFFER_LEN:
            buffer_right.popleft()

        if len(buffer_left) < BUFFER_LEN:
            frame_idx += 1
            continue

        valid_l = [c for c in buffer_left if len(c) > 0]
        valid_r = [c for c in buffer_right if len(c) > 0]

        events_left = np.concatenate(valid_l) if valid_l else np.array([], dtype=chunk_left.dtype)
        events_right = np.concatenate(valid_r) if valid_r else np.array([], dtype=chunk_right.dtype)

        # Detect clusters
        targets_left = detect_targets(events_left, use_time_weight=True)
        targets_right = detect_targets(events_right, use_time_weight=True)

        # Update trackers
        tracks_left = tracker_left.update(targets_left, frame_idx)
        tracks_right = tracker_right.update(targets_right, frame_idx)

        # Build observations
        obs_left = tracks_to_observations(tracks_left)
        obs_right = tracks_to_observations(tracks_right)

        # Frame matching with epipolar gating (+ respect locks)
        frame_pairs = matcher.match_frame(obs_left, obs_right, locked_l2r=accumulator.locked_l2r)

        # Update accumulator / lock pairs over time
        accumulator.update_with_frame_matches(
            matched_pairs=frame_pairs,
            active_left_ids=[o.track_id for o in obs_left],
            active_right_ids=[o.track_id for o in obs_right]
        )
        locked_pairs = accumulator.get_locked_pairs()

        # Visualization
        img_left = events_to_color_image(events_left, (IMG_HEIGHT, IMG_WIDTH))
        img_right = events_to_color_image(events_right, (IMG_HEIGHT, IMG_WIDTH))

        draw_tracks(img_left, tracks_left, "L")
        draw_tracks(img_right, tracks_right, "R")

        canvas = np.hstack([img_left, img_right])
        cv2.line(canvas, (IMG_WIDTH, 0), (IMG_WIDTH, IMG_HEIGHT), (0, 255, 255), 2)

        cv2.rectangle(canvas, (0, 0), (2 * IMG_WIDTH, 70), (30, 30, 30), -1)
        cv2.putText(canvas, f"Frame: {frame_idx} | L_obs={len(obs_left)} R_obs={len(obs_right)} | locked={len(locked_pairs)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(canvas, "SPACE:Pause  S:Save JSON  Q:Quit",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        # Show top locked pairs
        y0 = 95
        for k, (l_id, r_id, sc) in enumerate(locked_pairs[:10]):
            cv2.putText(canvas, f"LOCK {k}: L{l_id} <-> R{r_id}  score={sc:.1f}",
                        (10, y0 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

        cv2.imshow(win, canvas)

        while True:
            key = cv2.waitKey(1 if not paused else 60) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                # autosave on quit
                if locked_history_out:
                    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                        json.dump(locked_history_out, f, indent=2, ensure_ascii=False)
                    print(f"已保存: {OUTPUT_JSON}")
                return
            elif key == ord(' '):
                paused = not paused
            elif key == ord('s'):
                # Save snapshot: locked pairs + one 3D sample per locked pair using current centroids
                snap = {"frame": frame_idx, "locked_pairs": [], "triangulated_samples": []}
                # Map id->obs for current frame
                mapL = {o.track_id: o for o in obs_left}
                mapR = {o.track_id: o for o in obs_right}

                for l_id, r_id, sc in locked_pairs:
                    snap["locked_pairs"].append({"left": l_id, "right": r_id, "score": sc})
                    if l_id in mapL and r_id in mapR:
                        X = triangulator.triangulate_point(mapL[l_id].centroid, mapR[r_id].centroid)
                        snap["triangulated_samples"].append({
                            "left": l_id, "right": r_id, "score": sc,
                            "X_mm": [float(X[0]), float(X[1]), float(X[2])]
                        })

                locked_history_out.append(snap)
                with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                    json.dump(locked_history_out, f, indent=2, ensure_ascii=False)
                print(f"[保存] frame={frame_idx}, locked={len(locked_pairs)} -> {OUTPUT_JSON}")

            if not paused:
                break

        frame_idx += 1

    cv2.destroyAllWindows()
    if locked_history_out:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(locked_history_out, f, indent=2, ensure_ascii=False)
        print(f"已保存: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()