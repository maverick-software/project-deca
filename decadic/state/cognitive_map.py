"""WS-EXPAND E1 — cognitive map: spatial state estimation + experiential routing.

Three capabilities, all pure python (no torch), all built from the agent's own
lived signals (pathway-not-policy: nothing is downloaded, the map is earned):

1. **Pose estimation (E1.1/E1.2).** A running world pose (x, y, yaw). The body
   observation usually carries a proprioceptive position/orientation — treated
   as a high-weight correction into a complementary filter — and when it is
   absent the pose dead-reckons by integrating planar velocity. Landmark
   re-sighting (`correct_from_landmark`) blends the pose toward the position a
   remembered landmark implies, bounding drift.

2. **Experiential adjacency graph (E1.4).** As the agent travels it drops
   breadcrumb nodes every ``breadcrumb_m`` meters of MEASURED path (not
   straight-line displacement) and records the measured travel cost of each
   hop — so a hop that wound around an obstacle costs what it actually cost.
   Remembered resource targets are registered as landmark nodes. Node count is
   bounded (oldest breadcrumbs evicted with their edges).

3. **Stall-gated waypoint planning (E1.5).** The planner deliberately does
   NOT reroute by default: with no evidence, the straight-line bearing is the
   best guess and byte-parity with pre-E1 behavior. Only when pursuit of a
   target has STALLED (``stall_cycles`` without ``min_progress_m`` of approach,
   repeated ``block_threshold`` times) does `plan_next_waypoint` return the
   first hop of an A* route through the experiential graph — "the direct way
   failed; go the way I have actually walked." Evidence-review guardrails:
   measured (not raw-connectivity) edge weights, pruning by eviction, and a
   conservative fallback to the straight line whenever the graph has no answer.
"""

from __future__ import annotations

import heapq
import math
from typing import Any

from decadic import config as C


def _finite(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class CognitiveMap:
    """Per-agent spatial state. Reset with the mind (a new life maps anew)."""

    def __init__(
        self,
        *,
        breadcrumb_m: float | None = None,
        connect_radius_m: float | None = None,
        max_nodes: int | None = None,
        stall_cycles: int | None = None,
        min_progress_m: float | None = None,
        block_threshold: int | None = None,
        pose_blend: float | None = None,
    ) -> None:
        self.breadcrumb_m = float(breadcrumb_m if breadcrumb_m is not None else C.cmap_breadcrumb_m())
        self.connect_radius_m = float(
            connect_radius_m if connect_radius_m is not None else C.cmap_connect_radius_m()
        )
        self.max_nodes = int(max_nodes if max_nodes is not None else C.cmap_max_nodes())
        self.stall_cycles = int(stall_cycles if stall_cycles is not None else C.cmap_stall_cycles())
        self.min_progress_m = float(
            min_progress_m if min_progress_m is not None else C.cmap_min_progress_m()
        )
        self.block_threshold = int(
            block_threshold if block_threshold is not None else C.cmap_block_threshold()
        )
        self.pose_blend = float(pose_blend if pose_blend is not None else C.cmap_pose_blend())

        # -- pose state --
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._has_pose = False
        # -- graph state --
        self._nodes: dict[str, tuple[float, float]] = {}
        self._node_order: list[str] = []  # breadcrumb eviction order (FIFO)
        self._edges: dict[tuple[str, str], float] = {}  # key = sorted pair
        self._crumb_seq = 0
        self._last_node: str | None = None
        self._travel_since_node = 0.0
        self._prev_xy: tuple[float, float] | None = None
        # -- pursuit / blockage state --
        self._pursuit: dict[str, tuple[float, int]] = {}  # target -> (best_dist, stalled_cycles)
        self._blocked: dict[str, int] = {}  # target -> stall strikes
        # -- telemetry --
        self.pose_updates = 0
        self.dead_reckon_cycles = 0
        self.landmark_corrections = 0
        self.stall_events = 0
        self.reroutes = 0

    # ------------------------------------------------------------- pose (E1.1/2)

    def update_pose(self, proprio: Any, *, dt: float | None = None) -> bool:
        """Advance the pose from this cycle's proprioception dict.

        Observed position/orientation -> complementary blend (snap on first
        fix). No position but a planar velocity -> dead-reckoning integration
        (requires ``dt`` seconds). Anything malformed -> no-op. Returns whether
        a pose estimate exists. Travel is measured here (breadcrumb source).
        """
        if not isinstance(proprio, dict):
            return self._has_pose
        pos = proprio.get("position")
        ori = proprio.get("orientation")
        moved = False
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            ox, oy = _finite(pos[0]), _finite(pos[1])
            if not self._has_pose:
                self._x, self._y = ox, oy
            else:
                a = self.pose_blend
                self._x += a * (ox - self._x)
                self._y += a * (oy - self._y)
            if isinstance(ori, (list, tuple)) and len(ori) >= 3:
                oyaw = _finite(ori[2])
                if not self._has_pose:
                    self._yaw = oyaw
                else:
                    self._yaw = _wrap_pi(self._yaw + self.pose_blend * _wrap_pi(oyaw - self._yaw))
            self._has_pose = True
            self.pose_updates += 1
            moved = True
        elif self._has_pose and dt is not None:
            vel = proprio.get("velocity")
            if isinstance(vel, (list, tuple)) and len(vel) >= 2:
                self._x += _finite(vel[0]) * float(dt)
                self._y += _finite(vel[1]) * float(dt)
                if isinstance(ori, (list, tuple)) and len(ori) >= 3:
                    self._yaw = _finite(ori[2], self._yaw)
                self.dead_reckon_cycles += 1
                moved = True
        if moved:
            self._note_travel()
        return self._has_pose

    def correct_from_landmark(
        self, landmark_world: "tuple[float, float]", observed_rel: "tuple[float, float]"
    ) -> None:
        """E1.2: blend the pose toward the position a re-sighted landmark implies.

        ``observed_rel`` is the landmark's body-frame (forward, left) offset as
        currently perceived; the implied self position is the landmark's known
        world position minus that offset rotated into the world frame.
        """
        if not self._has_pose:
            return
        fwd, left = _finite(observed_rel[0]), _finite(observed_rel[1])
        cy, sy = math.cos(self._yaw), math.sin(self._yaw)
        implied_x = _finite(landmark_world[0]) - (fwd * cy - left * sy)
        implied_y = _finite(landmark_world[1]) - (fwd * sy + left * cy)
        a = self.pose_blend
        self._x += a * (implied_x - self._x)
        self._y += a * (implied_y - self._y)
        self.landmark_corrections += 1

    def pose(self) -> "tuple[float, float, float] | None":
        """(x, y, yaw) or None before the first fix."""
        if not self._has_pose:
            return None
        return self._x, self._y, self._yaw

    # ------------------------------------------------- experiential graph (E1.4)

    def _note_travel(self) -> None:
        xy = (self._x, self._y)
        if self._prev_xy is not None:
            self._travel_since_node += math.hypot(xy[0] - self._prev_xy[0], xy[1] - self._prev_xy[1])
        self._prev_xy = xy
        if self._travel_since_node >= self.breadcrumb_m:
            self._drop_breadcrumb()

    def _drop_breadcrumb(self) -> None:
        nid = f"b{self._crumb_seq}"
        self._crumb_seq += 1
        self._nodes[nid] = (self._x, self._y)
        self._node_order.append(nid)
        if self._last_node is not None and self._last_node in self._nodes:
            # Measured path cost of the hop actually walked (>= euclid when the
            # route wound around something) — the E1.4 edge-weight guardrail.
            self._add_edge(self._last_node, nid, self._travel_since_node)
        self._last_node = nid
        self._travel_since_node = 0.0
        while len(self._nodes) > self.max_nodes and self._node_order:
            victim = self._node_order.pop(0)
            self._nodes.pop(victim, None)
            for k in [k for k in self._edges if victim in k]:
                self._edges.pop(k, None)
            if self._last_node == victim:
                self._last_node = None

    def note_landmark(self, key: str, pos: Any) -> None:
        """Register/refresh a remembered target as a landmark node, connected to
        the nearest existing node by euclidean cost (it was co-experienced)."""
        if not isinstance(pos, (list, tuple)) or len(pos) < 2:
            return
        nid = f"e:{key}"
        p = (_finite(pos[0]), _finite(pos[1]))
        fresh = nid not in self._nodes
        self._nodes[nid] = p
        if fresh:
            near = self._nearest_node(p, exclude=nid)
            if near is not None:
                nx, ny = self._nodes[near]
                self._add_edge(nid, near, math.hypot(p[0] - nx, p[1] - ny))

    def _add_edge(self, a: str, b: str, cost: float) -> None:
        if a == b:
            return
        key = (a, b) if a < b else (b, a)
        cost = max(1e-6, float(cost))
        prev = self._edges.get(key)
        # Last-wins with optimism: keep the CHEAPEST measured traversal (a slow
        # wander must not poison a hop that was once walked directly).
        self._edges[key] = cost if prev is None else min(prev, cost)

    def _nearest_node(
        self, p: "tuple[float, float]", *, exclude: str | None = None, max_dist: float | None = None
    ) -> "str | None":
        best, best_d = None, float("inf")
        for nid, (nx, ny) in self._nodes.items():
            if nid == exclude:
                continue
            d = math.hypot(p[0] - nx, p[1] - ny)
            if d < best_d:
                best, best_d = nid, d
        if best is None:
            return None
        if max_dist is not None and best_d > max_dist:
            return None
        return best

    # ------------------------------------------------ pursuit / blockage (E1.5)

    def note_pursuit(self, target_key: str, dist_m: float) -> None:
        """Track approach progress toward the active remembered target.

        No ``min_progress_m`` of net approach for ``stall_cycles`` consecutive
        notes -> one stall strike against the target (the direct route is
        suspect). Reaching the target clears its record entirely.
        """
        d = _finite(dist_m, float("inf"))
        if d <= self.connect_radius_m * 0.5:  # arrived: the route works
            self._pursuit.pop(target_key, None)
            self._blocked.pop(target_key, None)
            return
        best, stalled = self._pursuit.get(target_key, (d, 0))
        if d < best - self.min_progress_m:
            self._pursuit[target_key] = (d, 0)  # real progress: reset the clock
            return
        stalled += 1
        if stalled >= self.stall_cycles:
            self._blocked[target_key] = min(self._blocked.get(target_key, 0) + 1, 1000)
            self.stall_events += 1
            stalled = 0
        self._pursuit[target_key] = (min(best, d), stalled)

    def is_blocked(self, target_key: str) -> bool:
        return self._blocked.get(target_key, 0) >= self.block_threshold

    # --------------------------------------------------------------- plan (E1.5)

    def plan_next_waypoint(
        self,
        self_xy: "tuple[float, float]",
        target_xy: "tuple[float, float]",
        target_key: str,
    ) -> "tuple[float, float] | None":
        """First hop of an experiential route to the target, or None.

        None means "use the straight-line bearing" — the default, and the
        answer whenever the target is not evidenced-blocked, the graph cannot
        anchor both endpoints, no path exists, or anything at all fails.
        """
        try:
            if not self.is_blocked(target_key):
                return None
            start = self._nearest_node(self_xy, max_dist=self.connect_radius_m)
            goal = self._nearest_node(target_xy, max_dist=self.connect_radius_m)
            if start is None or goal is None or start == goal:
                return None
            path = self._astar(start, goal)
            if not path:
                return None
            # First hop far enough away to steer by (skip the node underfoot).
            for nid in path:
                nx, ny = self._nodes[nid]
                if math.hypot(nx - self_xy[0], ny - self_xy[1]) > self.breadcrumb_m * 0.5:
                    self.reroutes += 1
                    return nx, ny
            return None
        except Exception:
            return None  # planner failure can never break the bearing

    def _astar(self, start: str, goal: str) -> "list[str]":
        adj: dict[str, list[tuple[str, float]]] = {}
        for (a, b), cost in self._edges.items():
            if a in self._nodes and b in self._nodes:
                adj.setdefault(a, []).append((b, cost))
                adj.setdefault(b, []).append((a, cost))
        gx, gy = self._nodes[goal]

        def h(nid: str) -> float:
            x, y = self._nodes[nid]
            return math.hypot(x - gx, y - gy)

        open_q: list[tuple[float, str]] = [(h(start), start)]
        g_score: dict[str, float] = {start: 0.0}
        came: dict[str, str] = {}
        seen: set[str] = set()
        while open_q:
            _f, cur = heapq.heappop(open_q)
            if cur == goal:
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                path.reverse()
                return path[1:]  # exclude the start node (we are standing there)
            if cur in seen:
                continue
            seen.add(cur)
            for nb, cost in adj.get(cur, []):
                ng = g_score[cur] + cost
                if ng < g_score.get(nb, float("inf")):
                    g_score[nb] = ng
                    came[nb] = cur
                    heapq.heappush(open_q, (ng + h(nb), nb))
        return []

    # ------------------------------------------------------------------ telemetry

    def telemetry(self) -> dict[str, float | int]:
        return {
            "cmap_nodes": len(self._nodes),
            "cmap_edges": len(self._edges),
            "cmap_pose_updates": self.pose_updates,
            "cmap_dead_reckon_cycles": self.dead_reckon_cycles,
            "cmap_landmark_corrections": self.landmark_corrections,
            "cmap_stall_events": self.stall_events,
            "cmap_reroutes": self.reroutes,
            "cmap_blocked_targets": sum(1 for v in self._blocked.values() if v >= self.block_threshold),
        }
