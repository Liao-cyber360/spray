import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Tuple, List, Optional


def skew(t: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix [t]_x for t shape (3,)."""
    tx, ty, tz = float(t[0]), float(t[1]), float(t[2])
    return np.array([
        [0.0, -tz,  ty],
        [tz,  0.0, -tx],
        [-ty, tx,  0.0]
    ], dtype=np.float64)


def fundamental_from_krt(K_left: np.ndarray, K_right: np.ndarray,
                         R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Compute fundamental matrix F from intrinsics and extrinsics.
    Assumes X_R = R X_L + T (right relative to left).
    """
    K_left = K_left.astype(np.float64)
    K_right = K_right.astype(np.float64)
    R = R.astype(np.float64)
    T = T.astype(np.float64).reshape(3)

    E = skew(T) @ R
    F = np.linalg.inv(K_right).T @ E @ np.linalg.inv(K_left)

    # Normalize for numerical stability (not strictly required)
    norm = np.linalg.norm(F)
    if norm > 0:
        F = F / norm
    return F


def point_line_distance_px(x: float, y: float, line_abc: np.ndarray) -> float:
    """Distance from point (x,y) to line ax+by+c=0 in pixels."""
    a, b, c = float(line_abc[0]), float(line_abc[1]), float(line_abc[2])
    denom = np.hypot(a, b)
    if denom < 1e-12:
        return float("inf")
    return abs(a * x + b * y + c) / denom


def epipolar_distance_lr(F: np.ndarray,
                         pt_left: Tuple[float, float],
                         pt_right: Tuple[float, float]) -> float:
    """
    Compute distance of right point to epipolar line induced by left point.
    l_r = F x_l
    d = |x_r^T l_r| / sqrt(l0^2 + l1^2)
    """
    xl = np.array([pt_left[0], pt_left[1], 1.0], dtype=np.float64)
    xr = np.array([pt_right[0], pt_right[1], 1.0], dtype=np.float64)
    line_r = F @ xl
    return point_line_distance_px(xr[0], xr[1], line_r)


def greedy_assignment(cost: np.ndarray, max_cost: float) -> List[Tuple[int, int]]:
    """Greedy min-cost matching for rectangular cost matrix."""
    matches: List[Tuple[int, int]] = []
    if cost.size == 0:
        return matches

    used_r = set()
    # flatten indices sorted by cost
    flat = [(cost[i, j], i, j) for i in range(cost.shape[0]) for j in range(cost.shape[1])]
    flat.sort(key=lambda x: x[0])

    used_l = set()
    for c, i, j in flat:
        if c > max_cost:
            break
        if i in used_l or j in used_r:
            continue
        used_l.add(i)
        used_r.add(j)
        matches.append((i, j))
    return matches


def hungarian_or_greedy(cost: np.ndarray, max_cost: float) -> List[Tuple[int, int]]:
    """
    Try Hungarian via SciPy if available; otherwise greedy.
    Returns list of (i_left, j_right) assignments with cost <= max_cost.
    """
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
        row_ind, col_ind = linear_sum_assignment(cost)
        pairs = []
        for i, j in zip(row_ind.tolist(), col_ind.tolist()):
            if i < cost.shape[0] and j < cost.shape[1] and cost[i, j] <= max_cost:
                pairs.append((i, j))
        return pairs
    except Exception:
        return greedy_assignment(cost, max_cost)


@dataclass
class Observation:
    track_id: int
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    event_count: int
    avg_timestamp: float


class PairAccumulator:
    """
    Accumulate evidence over time that (L_track_id, R_track_id) correspond.
    Locks pairs after enough support.
    """
    def __init__(self,
                 lock_score_thresh: float = 8.0,
                 inc_per_hit: float = 1.0,
                 dec_per_miss: float = 0.3,
                 decay: float = 0.98,
                 unlock_score_thresh: float = 2.0):
        self.lock_score_thresh = lock_score_thresh
        self.inc_per_hit = inc_per_hit
        self.dec_per_miss = dec_per_miss
        self.decay = decay
        self.unlock_score_thresh = unlock_score_thresh

        self.scores: Dict[Tuple[int, int], float] = defaultdict(float)
        self.locked_l2r: Dict[int, int] = {}
        self.locked_r2l: Dict[int, int] = {}

    def step_decay(self):
        # global decay to forget old evidence
        for k in list(self.scores.keys()):
            self.scores[k] *= self.decay
            if self.scores[k] < 1e-3:
                del self.scores[k]

    def update_with_frame_matches(self,
                                 matched_pairs: List[Tuple[int, int]],
                                 active_left_ids: List[int],
                                 active_right_ids: List[int]):
        """
        matched_pairs: list of (L_track_id, R_track_id) for this frame.
        Also penalize conflicting associations for active tracks.
        """
        self.step_decay()

        matched_set = set(matched_pairs)

        # Reward hits
        for l_id, r_id in matched_pairs:
            self.scores[(l_id, r_id)] += self.inc_per_hit

        # Penalize alternatives for active tracks (softly)
        # For each active left track, if it matched to some r, penalize other r with existing scores
        left_to_r = {l: r for l, r in matched_pairs}
        right_to_l = {r: l for l, r in matched_pairs}

        for (l_id, r_id), sc in list(self.scores.items()):
            if l_id in left_to_r and left_to_r[l_id] != r_id:
                self.scores[(l_id, r_id)] = max(0.0, sc - self.dec_per_miss)
            if r_id in right_to_l and right_to_l[r_id] != l_id:
                self.scores[(l_id, r_id)] = max(0.0, self.scores[(l_id, r_id)] - self.dec_per_miss)

        # Locking logic: choose best for each left, require one-to-one
        # Build candidate list sorted by score desc
        candidates = [(sc, l, r) for (l, r), sc in self.scores.items()]
        candidates.sort(reverse=True, key=lambda x: x[0])

        new_l2r: Dict[int, int] = {}
        new_r2l: Dict[int, int] = {}

        # Keep existing locks if still supported
        for l, r in list(self.locked_l2r.items()):
            sc = self.scores.get((l, r), 0.0)
            if sc >= self.unlock_score_thresh and l in active_left_ids and r in active_right_ids:
                new_l2r[l] = r
                new_r2l[r] = l

        # Add new locks
        for sc, l, r in candidates:
            if sc < self.lock_score_thresh:
                break
            if l in new_l2r or r in new_r2l:
                continue
            new_l2r[l] = r
            new_r2l[r] = l

        self.locked_l2r = new_l2r
        self.locked_r2l = new_r2l

    def get_locked_pairs(self) -> List[Tuple[int, int, float]]:
        out = []
        for l, r in self.locked_l2r.items():
            out.append((l, r, float(self.scores.get((l, r), 0.0))))
        out.sort(key=lambda x: -x[2])
        return out


class EpipolarGatedMatcher:
    def __init__(self,
                 F: np.ndarray,
                 epipolar_thresh_px: float = 10.0,
                 w_epi: float = 0.75,
                 w_size: float = 0.10,
                 w_evt: float = 0.15,
                 max_cost: float = 0.95):
        self.F = F.astype(np.float64)
        self.epipolar_thresh_px = float(epipolar_thresh_px)
        self.w_epi = float(w_epi)
        self.w_size = float(w_size)
        self.w_evt = float(w_evt)
        self.max_cost = float(max_cost)

    @staticmethod
    def _bbox_area(b: Tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = b
        return max(1.0, float((x2 - x1) * (y2 - y1)))

    def compute_pair_cost(self, left: Observation, right: Observation) -> Tuple[float, float]:
        """
        Returns (total_cost, epi_dist_px). total_cost in [0, +inf), lower better.
        """
        d_epi = epipolar_distance_lr(self.F, left.centroid, right.centroid)
        c_epi = min(d_epi / self.epipolar_thresh_px, 1.0)

        # size/area similarity
        aL = self._bbox_area(left.bbox)
        aR = self._bbox_area(right.bbox)
        c_size = min(abs(np.log(aL / aR)) / 1.5, 1.0)  # scale factor 1.5 is empirical

        # event count similarity
        eL = max(1.0, float(left.event_count))
        eR = max(1.0, float(right.event_count))
        c_evt = abs(eL - eR) / max(eL, eR)

        total = self.w_epi * c_epi + self.w_size * c_size + self.w_evt * c_evt
        return float(total), float(d_epi)

    def match_frame(self,
                    left_obs: List[Observation],
                    right_obs: List[Observation],
                    locked_l2r: Optional[Dict[int, int]] = None) -> List[Tuple[int, int]]:
        """
        Return list of (left_track_id, right_track_id) for this frame.
        Uses one-to-one assignment with epipolar gating.
        If a left track is locked, it will only consider its locked right (if present) unless that violates gating badly.
        """
        if not left_obs or not right_obs:
            return []

        locked_l2r = locked_l2r or {}

        # Build cost matrix
        nL, nR = len(left_obs), len(right_obs)
        cost = np.full((nL, nR), fill_value=1e6, dtype=np.float64)

        for i, lo in enumerate(left_obs):
            for j, ro in enumerate(right_obs):
                # Respect locks: if lo is locked, only allow that ro
                if lo.track_id in locked_l2r and locked_l2r[lo.track_id] != ro.track_id:
                    continue

                total, d_epi = self.compute_pair_cost(lo, ro)
                # hard gate by epipolar distance
                if d_epi <= self.epipolar_thresh_px:
                    cost[i, j] = total

        pairs_idx = hungarian_or_greedy(cost, max_cost=self.max_cost)

        matched_pairs: List[Tuple[int, int]] = []
        for i, j in pairs_idx:
            if cost[i, j] >= 1e5:
                continue
            matched_pairs.append((left_obs[i].track_id, right_obs[j].track_id))
        return matched_pairs