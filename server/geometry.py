"""Camera -> torso transform and simple shape fitting from aligned depth.

Everything is expressed in the torso frame (x forward, y left, z up), which is also the planner's
base frame. The camera is fixed on the torso, so the transform is constant (from the URDF).
"""
import numpy as np

from common import config
from common.fk import FK

# columns = optical x (right), y (down), z (forward) expressed in the camera link frame (x fwd, y left, z up)
R_LINK_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


class Geometry:
    def __init__(self):
        cfg = config.load()
        self.cfg = cfg
        T_link = FK(config.path(cfg["robot"]["urdf"])).pose(cfg["camera_mount"]["link"])
        self.R = T_link[:3, :3] @ R_LINK_OPTICAL
        self.t = T_link[:3, 3] + T_link[:3, :3] @ np.array(cfg["camera_mount"]["rgb_offset_m"])

    # ---------------------------------------------------------------- points
    def points(self, frame, rect=None, step=2, exclude=None):
        """Torso-frame points (N,3) for depth pixels inside rect (x, y, w, h); optionally minus `exclude` rect."""
        h, w = frame.depth.shape
        x0, y0, x1, y1 = (0, 0, w, h) if rect is None else (
            max(0, int(rect[0])), max(0, int(rect[1])), min(w, int(rect[0] + rect[2])), min(h, int(rect[1] + rect[3])))
        vs, us = np.mgrid[y0:y1:step, x0:x1:step]
        z = frame.depth[vs, us].astype(np.float32) * frame.depth_scale
        keep = z > 0.05
        if exclude is not None:
            ex, ey, ew, eh = exclude
            keep &= ~((us >= ex) & (us < ex + ew) & (vs >= ey) & (vs < ey + eh))
        us, vs, z = us[keep], vs[keep], z[keep]
        k = frame.intrinsics
        cam = np.stack([(us - k["ppx"]) * z / k["fx"], (vs - k["ppy"]) * z / k["fy"], z], axis=1)
        return cam @ self.R.T + self.t

    @staticmethod
    def plane_height(z, bin_m=0.01):
        """Dominant horizontal surface height among z values (histogram mode), or None."""
        if z.size < 30:
            return None
        hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + bin_m, bin_m))
        if hist.size == 0:
            return None
        i = int(np.argmax(hist))
        sel = z[(z >= edges[i] - bin_m) & (z <= edges[i + 1] + bin_m)]
        return float(np.median(sel))

    # ---------------------------------------------------------------- shapes
    def fit_box(self, frame, rect):
        """Cuboid of an object standing on a surface: OWLv2 rect + depth.
        Returns {center, dims, top_z, support_z, front_x} in metres (torso frame) or None."""
        x, y, w, h = rect
        ring = (x - 0.4 * w, y - 0.4 * h, 1.8 * w, 1.8 * h)
        inner = (x + 0.08 * w, y + 0.08 * h, 0.84 * w, 0.84 * h)
        obj = self.points(frame, inner)
        around = self.points(frame, ring, exclude=rect)
        if obj.shape[0] < 50:
            return None
        top_z = float(np.percentile(obj[:, 2], 97))
        support = self.plane_height(around[around[:, 2] < top_z - 0.02, 2]) if around.size else None
        if support is None:
            support = float(np.percentile(obj[:, 2], 3))
        body = obj[obj[:, 2] > support + 0.01]
        if body.shape[0] < 30:
            return None
        x0, x1 = np.percentile(body[:, 0], [2, 98])
        y0, y1 = np.percentile(body[:, 1], [2, 98])
        dims = [float(x1 - x0), float(y1 - y0), float(top_z - support)]
        center = [float((x0 + x1) / 2), float((y0 + y1) / 2), float((top_z + support) / 2)]
        return {"center": center, "dims": dims, "top_z": top_z, "support_z": support, "front_x": float(x0)}

    def fit_surface(self, frame, rect):
        """Table-like surface inside rect: top height, front edge distance and extent (torso frame)."""
        pts = self.points(frame, rect)
        pts = pts[pts[:, 2] > -0.6]          # drop the floor (torso is ~0.8 m above it)
        top = self.plane_height(pts[:, 2]) if pts.size else None
        if top is None:
            return None
        s = pts[np.abs(pts[:, 2] - top) < 0.02]
        if s.shape[0] < 30:
            return None
        x0, x1 = np.percentile(s[:, 0], [2, 98])
        y0, y1 = np.percentile(s[:, 1], [2, 98])
        return {"top_z": top, "front_x": float(x0), "back_x": float(x1), "y_range": [float(y0), float(y1)]}
