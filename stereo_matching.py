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
                 unlock_score_thresh: float = 2.0,
                 max_score: float = 20.0,
                 mutual_exclusion_penalty: float = 2.0,
                 mutual_exclusion_min_ratio: float = 0.5):
        self.lock_score_thresh = lock_score_thresh
        self.inc_per_hit = inc_per_hit
        self.dec_per_miss = dec_per_miss
        self.decay = decay
        self.unlock_score_thresh = unlock_score_thresh
        # Upper bound for a single pair's accumulated score
        self.max_score = max_score
        # Extra penalty applied to lower-ranked alternatives when mutual exclusion is enforced
        self.mutual_exclusion_penalty = mutual_exclusion_penalty
        # Only enforce mutual exclusion for pairs whose score exceeds this fraction of lock_score_thresh
        self.mutual_exclusion_min_ratio = mutual_exclusion_min_ratio

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

        # Reward hits (capped at max_score to prevent unbounded growth)
        for l_id, r_id in matched_pairs:
            self.scores[(l_id, r_id)] = min(
                self.scores[(l_id, r_id)] + self.inc_per_hit,
                self.max_score
            )

        # Penalize alternatives for active tracks (softly)
        left_to_r = {l: r for l, r in matched_pairs}
        right_to_l = {r: l for l, r in matched_pairs}

        for (l_id, r_id), sc in list(self.scores.items()):
            if l_id in left_to_r and left_to_r[l_id] != r_id:
                self.scores[(l_id, r_id)] = max(0.0, sc - self.dec_per_miss)
            if r_id in right_to_l and right_to_l[r_id] != l_id:
                self.scores[(l_id, r_id)] = max(0.0, self.scores[(l_id, r_id)] - self.dec_per_miss)

        # Enforce mutual exclusion: when multiple right IDs compete for the same left ID
        # (or vice versa), penalise the lower-scoring alternatives.
        min_score_for_exclusion = self.lock_score_thresh * self.mutual_exclusion_min_ratio
        left_groups: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        right_groups: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        for (l_id, r_id), sc in self.scores.items():
            if sc > min_score_for_exclusion:
                left_groups[l_id].append((r_id, sc))
                right_groups[r_id].append((l_id, sc))

        for l_id, candidates in left_groups.items():
            if len(candidates) > 1:
                candidates.sort(key=lambda x: -x[1])
                for r_id, _ in candidates[1:]:
                    self.scores[(l_id, r_id)] = max(
                        0.0, self.scores[(l_id, r_id)] - self.mutual_exclusion_penalty
                    )

        for r_id, candidates in right_groups.items():
            if len(candidates) > 1:
                candidates.sort(key=lambda x: -x[1])
                for l_id, _ in candidates[1:]:
                    self.scores[(l_id, r_id)] = max(
                        0.0, self.scores[(l_id, r_id)] - self.mutual_exclusion_penalty
                    )

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
                 max_cost: float = 0.95,
                 motion_penalty_thresh: float = 50.0,
                 layer1_thresh_multiplier: float = 1.5,
                 layer2_max_cost: float = 0.75,
                 layer3_thresh_multiplier: float = 2.0):
        self.F = F.astype(np.float64)
        self.epipolar_thresh_px = float(epipolar_thresh_px)
        self.w_epi = float(w_epi)
        self.w_size = float(w_size)
        self.w_evt = float(w_evt)
        self.max_cost = float(max_cost)
        # Pixel distance at which motion prediction penalty reaches maximum
        self.motion_penalty_thresh = float(motion_penalty_thresh)
        # Layer 1: retain locked pairs at this multiple of the base epipolar threshold
        self.layer1_thresh_multiplier = float(layer1_thresh_multiplier)
        # Layer 2: strict max assignment cost for new (non-locked) pairs
        self.layer2_max_cost = float(layer2_max_cost)
        # Layer 3: relaxed epipolar threshold multiplier for the fallback pass
        self.layer3_thresh_multiplier = float(layer3_thresh_multiplier)

    @staticmethod
    def _bbox_area(b: Tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = b
        return max(1.0, float((x2 - x1) * (y2 - y1)))

    def _compute_match_cost(self,
                            left: Observation,
                            right: Observation,
                            pred_left: Optional[Tuple[float, float]],
                            pred_right: Optional[Tuple[float, float]],
                            epipolar_thresh: float) -> Tuple[float, float]:
        """
        Compute match cost between left and right observations.

        When motion predictions are available the weights shift to:
            0.50 × epipolar + 0.15 × size + 0.15 × event + 0.20 × motion
        Otherwise falls back to the instance weights (w_epi / w_size / w_evt).

        Returns:
            (total_cost, epi_dist_px)
        """
        d_epi = epipolar_distance_lr(self.F, left.centroid, right.centroid)
        c_epi = min(d_epi / epipolar_thresh, 1.0)

        # size/area similarity
        aL = self._bbox_area(left.bbox)
        aR = self._bbox_area(right.bbox)
        c_size = min(abs(np.log(aL / aR)) / 1.5, 1.0)

        # event count similarity
        eL = max(1.0, float(left.event_count))
        eR = max(1.0, float(right.event_count))
        c_evt = abs(eL - eR) / max(eL, eR)

        # motion prediction penalty
        c_motion = 0.0
        motion_count = 0
        if pred_left is not None:
            dist_l = np.hypot(left.centroid[0] - pred_left[0],
                              left.centroid[1] - pred_left[1])
            c_motion += min(dist_l / self.motion_penalty_thresh, 1.0)
            motion_count += 1
        if pred_right is not None:
            dist_r = np.hypot(right.centroid[0] - pred_right[0],
                              right.centroid[1] - pred_right[1])
            c_motion += min(dist_r / self.motion_penalty_thresh, 1.0)
            motion_count += 1

        if motion_count > 0:
            c_motion /= motion_count
            total = 0.50 * c_epi + 0.15 * c_size + 0.15 * c_evt + 0.20 * c_motion
        else:
            total = self.w_epi * c_epi + self.w_size * c_size + self.w_evt * c_evt

        return float(total), float(d_epi)

    def compute_pair_cost(self, left: Observation, right: Observation) -> Tuple[float, float]:
        """
        Returns (total_cost, epi_dist_px). total_cost in [0, +inf), lower is better.
        Uses the instance epipolar threshold and no motion prediction (backward-compatible).
        """
        return self._compute_match_cost(left, right, None, None, self.epipolar_thresh_px)

    def _build_and_solve(self,
                         left_obs: List[Observation],
                         right_obs: List[Observation],
                         left_predictions: Dict[int, Tuple[float, float]],
                         right_predictions: Dict[int, Tuple[float, float]],
                         epipolar_thresh: float,
                         max_cost: float) -> List[Tuple[int, int]]:
        """Build a cost matrix for the given subsets and solve the assignment problem."""
        nL, nR = len(left_obs), len(right_obs)
        cost = np.full((nL, nR), fill_value=1e6, dtype=np.float64)

        for i, lo in enumerate(left_obs):
            pred_l = left_predictions.get(lo.track_id)
            for j, ro in enumerate(right_obs):
                pred_r = right_predictions.get(ro.track_id)
                total, d_epi = self._compute_match_cost(lo, ro, pred_l, pred_r,
                                                        epipolar_thresh)
                if d_epi <= epipolar_thresh:
                    cost[i, j] = total

        pairs_idx = hungarian_or_greedy(cost, max_cost=max_cost)
        result: List[Tuple[int, int]] = []
        for i, j in pairs_idx:
            if cost[i, j] < 1e5:
                result.append((left_obs[i].track_id, right_obs[j].track_id))
        return result

    def match_frame(self,
                    left_obs: List[Observation],
                    right_obs: List[Observation],
                    locked_l2r: Optional[Dict[int, int]] = None,
                    left_predictions: Optional[Dict[int, Tuple[float, float]]] = None,
                    right_predictions: Optional[Dict[int, Tuple[float, float]]] = None
                    ) -> List[Tuple[int, int]]:
        """
        Return list of (left_track_id, right_track_id) matched pairs for this frame.

        Implements a three-layer matching strategy:
          Layer 1 – Retain existing locked pairs (epipolar threshold relaxed ×1.5).
          Layer 2 – Strict new matching for unmatched observations (base threshold).
          Layer 3 – Relaxed fallback for still-unmatched observations (threshold ×2.0).

        Args:
            left_obs: Active left-camera observations.
            right_obs: Active right-camera observations.
            locked_l2r: Existing locked left→right ID mapping from PairAccumulator.
            left_predictions: Dict mapping left track_id → predicted (x, y) from Kalman filter.
            right_predictions: Dict mapping right track_id → predicted (x, y) from Kalman filter.
        """
        if not left_obs or not right_obs:
            return []

        locked_l2r = locked_l2r or {}
        left_predictions = left_predictions or {}
        right_predictions = right_predictions or {}

        lo_by_id = {o.track_id: o for o in left_obs}
        ro_by_id = {o.track_id: o for o in right_obs}

        matched_pairs: List[Tuple[int, int]] = []
        matched_l_ids: set = set()
        matched_r_ids: set = set()

        # ── Layer 1: maintain existing locked pairs with relaxed epipolar threshold ──
        layer1_thresh = self.epipolar_thresh_px * self.layer1_thresh_multiplier
        for l_id, r_id in locked_l2r.items():
            lo = lo_by_id.get(l_id)
            ro = ro_by_id.get(r_id)
            if lo is None or ro is None:
                continue
            d_epi = epipolar_distance_lr(self.F, lo.centroid, ro.centroid)
            if d_epi <= layer1_thresh:
                matched_pairs.append((l_id, r_id))
                matched_l_ids.add(l_id)
                matched_r_ids.add(r_id)

        # ── Layer 2: strict matching for unmatched observations ──
        unmatched_l = [o for o in left_obs if o.track_id not in matched_l_ids]
        unmatched_r = [o for o in right_obs if o.track_id not in matched_r_ids]
        if unmatched_l and unmatched_r:
            new_pairs = self._build_and_solve(
                unmatched_l, unmatched_r,
                left_predictions, right_predictions,
                epipolar_thresh=self.epipolar_thresh_px,
                max_cost=self.layer2_max_cost,
            )
            for l_id, r_id in new_pairs:
                matched_pairs.append((l_id, r_id))
                matched_l_ids.add(l_id)
                matched_r_ids.add(r_id)

        # ── Layer 3: relaxed fallback for remaining unmatched observations ──
        unmatched_l = [o for o in left_obs if o.track_id not in matched_l_ids]
        unmatched_r = [o for o in right_obs if o.track_id not in matched_r_ids]
        if unmatched_l and unmatched_r:
            new_pairs = self._build_and_solve(
                unmatched_l, unmatched_r,
                left_predictions, right_predictions,
                epipolar_thresh=self.epipolar_thresh_px * self.layer3_thresh_multiplier,
                max_cost=self.max_cost,
            )
            for l_id, r_id in new_pairs:
                matched_pairs.append((l_id, r_id))

        return matched_pairs