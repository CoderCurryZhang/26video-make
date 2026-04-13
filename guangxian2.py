from manim import *
import numpy as np
import math
import random
from typing import List, Tuple, Dict, Optional


class Vector3D:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @staticmethod
    def from_array(arr: np.ndarray) -> 'Vector3D':
        return Vector3D(float(arr[0]), float(arr[1]), float(arr[2]))

    def clone(self) -> 'Vector3D':
        return Vector3D(self.x, self.y, self.z)

    def add(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def sub(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def mul(self, scalar: float) -> 'Vector3D':
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def div(self, scalar: float) -> 'Vector3D':
        if abs(scalar) < 1e-12:
            return Vector3D(0.0, 0.0, 0.0)
        return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)

    def dot(self, other: 'Vector3D') -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: 'Vector3D') -> 'Vector3D':
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def cross_z(self, other: 'Vector3D') -> float:
        return self.x * other.y - self.y * other.x

    def mag_sq(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z

    def mag(self) -> float:
        return math.sqrt(self.mag_sq())

    def norm(self) -> 'Vector3D':
        m = self.mag()
        if m < 1e-12:
            return Vector3D(0.0, 0.0, 0.0)
        return Vector3D(self.x / m, self.y / m, self.z / m)

    def rotate_x(self, angle: float) -> 'Vector3D':
        c = math.cos(angle)
        s = math.sin(angle)
        ny = self.y * c - self.z * s
        nz = self.y * s + self.z * c
        return Vector3D(self.x, ny, nz)

    def rotate_y(self, angle: float) -> 'Vector3D':
        c = math.cos(angle)
        s = math.sin(angle)
        nx = self.x * c + self.z * s
        nz = -self.x * s + self.z * c
        return Vector3D(nx, self.y, nz)

    def rotate_z(self, angle: float) -> 'Vector3D':
        c = math.cos(angle)
        s = math.sin(angle)
        nx = self.x * c - self.y * s
        ny = self.x * s + self.y * c
        return Vector3D(nx, ny, self.z)

    def distance_to(self, other: 'Vector3D') -> float:
        return self.sub(other).mag()

    def distance_squared_to(self, other: 'Vector3D') -> float:
        return self.sub(other).mag_sq()

    def lerp(self, other: 'Vector3D', t: float) -> 'Vector3D':
        return self.add(other.sub(self).mul(t))

    def project_onto(self, other: 'Vector3D') -> 'Vector3D':
        o_norm = other.norm()
        return o_norm.mul(self.dot(o_norm))

    def reflect(self, normal: 'Vector3D') -> 'Vector3D':
        n = normal.norm()
        return self.sub(n.mul(2.0 * self.dot(n)))

    def angle_between(self, other: 'Vector3D') -> float:
        m1 = self.mag()
        m2 = other.mag()
        if m1 < 1e-12 or m2 < 1e-12:
            return 0.0
        val = self.dot(other) / (m1 * m2)
        val = max(-1.0, min(1.0, val))
        return math.acos(val)

    def is_zero(self, tolerance: float = 1e-9) -> bool:
        return self.mag_sq() < tolerance * tolerance


class Matrix4x4:
    def __init__(self):
        self.m = [[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0]]

    def multiply(self, other: 'Matrix4x4') -> 'Matrix4x4':
        res = Matrix4x4()
        for i in range(4):
            for j in range(4):
                val = 0.0
                for k in range(4):
                    val += self.m[i][k] * other.m[k][j]
                res.m[i][j] = val
        return res


class ColorMath:
    @staticmethod
    def hex_to_rgb(hex_str: str) -> Tuple[float, float, float]:
        hex_str = str(hex_str).lstrip('#')
        if len(hex_str) == 3:
            hex_str = hex_str[0] * 2 + hex_str[1] * 2 + hex_str[2] * 2
        r = int(hex_str[0:2], 16) / 255.0
        g = int(hex_str[2:4], 16) / 255.0
        b = int(hex_str[4:6], 16) / 255.0
        return r, g, b

    @staticmethod
    def rgb_to_hex(r: float, g: float, b: float) -> str:
        r_int = int(max(0.0, min(1.0, r)) * 255)
        g_int = int(max(0.0, min(1.0, g)) * 255)
        b_int = int(max(0.0, min(1.0, b)) * 255)
        return f"#{r_int:02x}{g_int:02x}{b_int:02x}"

    @staticmethod
    def lerp_color(c1: str, c2: str, t: float) -> str:
        r1, g1, b1 = ColorMath.hex_to_rgb(c1)
        r2, g2, b2 = ColorMath.hex_to_rgb(c2)
        t = max(0.0, min(1.0, t))
        r = r1 + (r2 - r1) * t
        g = g1 + (g2 - g1) * t
        b = b1 + (b2 - b1) * t
        return ColorMath.rgb_to_hex(r, g, b)


class BezierCurve:
    def __init__(self, p0: Vector3D, p1: Vector3D, p2: Vector3D, p3: Vector3D):
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

    def get_point(self, t: float) -> Vector3D:
        u = 1.0 - t
        tt = t * t
        uu = u * u
        uuu = uu * u
        ttt = tt * t
        p = self.p0.mul(uuu)
        p = p.add(self.p1.mul(3 * uu * t))
        p = p.add(self.p2.mul(3 * u * tt))
        p = p.add(self.p3.mul(ttt))
        return p

    def get_derivative(self, t: float) -> Vector3D:
        u = 1.0 - t
        d = self.p1.sub(self.p0).mul(3 * u * u)
        d = d.add(self.p2.sub(self.p1).mul(6 * u * t))
        d = d.add(self.p3.sub(self.p2).mul(3 * t * t))
        return d

    def get_second_derivative(self, t: float) -> Vector3D:
        u = 1.0 - t
        d = self.p2.sub(self.p1).mul(2.0).sub(self.p1.sub(self.p0)).mul(6 * u)
        d = d.add(self.p3.sub(self.p2).sub(self.p2.sub(self.p1)).mul(6 * t))
        return d

    def get_normal(self, t: float) -> Vector3D:
        der = self.get_derivative(t).norm()
        return Vector3D(-der.y, der.x, 0.0)

    def get_tangent(self, t: float) -> Vector3D:
        return self.get_derivative(t).norm()


class OpticsMaterial:
    def __init__(self, index: float):
        self.index = index


class OpticsPhysics:
    @staticmethod
    def calculate_fresnel(dir_in: Vector3D, normal: Vector3D, n1: float, n2: float) -> float:
        cos_i = -dir_in.dot(normal)
        if cos_i < 0:
            normal = normal.mul(-1.0)
            cos_i = -cos_i
        sin_i_sq = 1.0 - cos_i * cos_i
        sin_t_sq = (n1 / n2) ** 2 * sin_i_sq
        if sin_t_sq >= 1.0:
            return 1.0
        cos_t = math.sqrt(1.0 - sin_t_sq)
        rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
        rp = (n1 * cos_t - n2 * cos_i) / (n1 * cos_t + n2 * cos_i)
        return (rs * rs + rp * rp) / 2.0

    @staticmethod
    def calculate_snell(dir_in: Vector3D, normal: Vector3D, n1: float, n2: float) -> Vector3D:
        cos_i = -dir_in.dot(normal)
        if cos_i < 0:
            normal = normal.mul(-1.0)
            cos_i = -cos_i
        eta = n1 / n2
        sin_t_sq = eta * eta * (1.0 - cos_i * cos_i)
        if sin_t_sq >= 1.0:
            return dir_in.reflect(normal)
        cos_t = math.sqrt(1.0 - sin_t_sq)
        term1 = dir_in.mul(eta)
        term2 = normal.mul(eta * cos_i - cos_t)
        return term1.add(term2).norm()


class OpticalBoundary:
    def __init__(self, start: Vector3D, end: Vector3D, mat_in: OpticsMaterial, mat_out: OpticsMaterial):
        self.start = start
        self.end = end
        self.mat_in = mat_in
        self.mat_out = mat_out
        self.dir = end.sub(start)
        self.length = self.dir.mag()
        self.dir = self.dir.norm()
        self.normal = Vector3D(-self.dir.y, self.dir.x, 0.0).norm()


class IntersectionRecord:
    def __init__(self, hit: bool, t: float, point: Vector3D, normal: Vector3D, mat_in: OpticsMaterial,
                 mat_out: OpticsMaterial):
        self.hit = hit
        self.t = t
        self.point = point
        self.normal = normal
        self.mat_in = mat_in
        self.mat_out = mat_out


class RaySegment:
    def __init__(self, origin: Vector3D, direction: Vector3D):
        self.origin = origin
        self.direction = direction.norm()


class IntersectionEngine:
    def __init__(self, boundaries: List[OpticalBoundary]):
        self.boundaries = boundaries

    def find_nearest(self, ray: RaySegment) -> IntersectionRecord:
        best = IntersectionRecord(False, float('inf'), Vector3D(0, 0, 0), Vector3D(0, 0, 0), OpticsMaterial(1.0),
                                  OpticsMaterial(1.0))
        for bnd in self.boundaries:
            det = ray.direction.cross_z(bnd.dir)
            if abs(det) < 1e-9:
                continue
            diff = bnd.start.sub(ray.origin)
            t = diff.cross_z(bnd.dir) / det
            u = diff.cross_z(ray.direction) / det
            if t > 1e-6 and 0.0 <= u <= bnd.length:
                n = bnd.normal.clone()
                if ray.direction.dot(n) > 0:
                    n = n.mul(-1.0)
                if t < best.t:
                    best = IntersectionRecord(True, t, ray.origin.add(ray.direction.mul(t)), n, bnd.mat_in, bnd.mat_out)
        return best


class PathNode:
    def __init__(self, point: Vector3D, cumulative_dist: float, is_bounce: bool):
        self.point = point
        self.cumulative_dist = cumulative_dist
        self.is_bounce = is_bounce


class OpticalPath:
    def __init__(self):
        self.nodes: List[PathNode] = []
        self.total_length = 0.0

    def add_point(self, point: Vector3D, is_bounce: bool):
        if not self.nodes:
            self.nodes.append(PathNode(point, 0.0, is_bounce))
        else:
            prev = self.nodes[-1].point
            d = prev.distance_to(point)
            self.total_length += d
            self.nodes.append(PathNode(point, self.total_length, is_bounce))

    def extract_subpath(self, target_dist: float) -> List[np.ndarray]:
        if not self.nodes:
            return []
        if target_dist <= 0:
            return [self.nodes[0].point.to_array(), self.nodes[0].point.to_array()]
        result = []
        for i in range(len(self.nodes) - 1):
            n1 = self.nodes[i]
            n2 = self.nodes[i + 1]
            result.append(n1.point.to_array())
            if n2.cumulative_dist >= target_dist:
                seg_len = n2.cumulative_dist - n1.cumulative_dist
                if seg_len > 1e-9:
                    ratio = (target_dist - n1.cumulative_dist) / seg_len
                    interp = n1.point.lerp(n2.point, ratio)
                    result.append(interp.to_array())
                else:
                    result.append(n2.point.to_array())
                return result
        result.append(self.nodes[-1].point.to_array())
        return result


class FiberOpticModel:
    def __init__(self):
        self.p0 = Vector3D(-7.5, 3.5, 0.0)
        self.p1 = Vector3D(-2.0, 3.5, 0.0)
        self.p2 = Vector3D(2.0, -3.5, 0.0)
        self.p3 = Vector3D(7.5, -3.5, 0.0)
        self.curve = BezierCurve(self.p0, self.p1, self.p2, self.p3)
        self.thickness = 1.05
        self.clad_thickness = 0.45
        self.jacket_thickness = 0.8
        self.resolution = 250
        self.mat_core = OpticsMaterial(1.65)
        self.mat_clad = OpticsMaterial(1.00)
        self.pts_core_top: List[Vector3D] = []
        self.pts_core_bot: List[Vector3D] = []
        self.pts_clad_top: List[Vector3D] = []
        self.pts_clad_bot: List[Vector3D] = []
        self.pts_jack_top: List[Vector3D] = []
        self.pts_jack_bot: List[Vector3D] = []
        self.boundaries: List[OpticalBoundary] = []
        self.generate()

    def generate(self):
        for i in range(self.resolution):
            t = i / float(self.resolution - 1)
            p = self.curve.get_point(t)
            n = self.curve.get_normal(t)
            core_top = p.add(n.mul(self.thickness / 2.0))
            core_bot = p.sub(n.mul(self.thickness / 2.0))
            clad_top = core_top.add(n.mul(self.clad_thickness))
            clad_bot = core_bot.sub(n.mul(self.clad_thickness))
            jack_top = clad_top.add(n.mul(self.jacket_thickness))
            jack_bot = clad_bot.sub(n.mul(self.jacket_thickness))
            self.pts_core_top.append(core_top)
            self.pts_core_bot.append(core_bot)
            self.pts_clad_top.append(clad_top)
            self.pts_clad_bot.append(clad_bot)
            self.pts_jack_top.append(jack_top)
            self.pts_jack_bot.append(jack_bot)
        for i in range(self.resolution - 1):
            self.boundaries.append(
                OpticalBoundary(self.pts_core_top[i], self.pts_core_top[i + 1], self.mat_core, self.mat_clad))
            self.boundaries.append(
                OpticalBoundary(self.pts_core_bot[i], self.pts_core_bot[i + 1], self.mat_core, self.mat_clad))
        self.boundaries.append(
            OpticalBoundary(self.pts_core_top[0], self.pts_core_bot[0], self.mat_clad, self.mat_core))
        self.boundaries.append(
            OpticalBoundary(self.pts_core_top[-1], self.pts_core_bot[-1], self.mat_core, self.mat_clad))

    def create_3d_ribbon(self, pts_inner: List[Vector3D], pts_outer: List[Vector3D], layers: int,
                         is_top: bool) -> VGroup:
        grp = VGroup()
        for i in range(layers):
            t1 = i / float(layers)
            t2 = (i + 1) / float(layers)
            intensity = 0.15 + 0.85 * math.sin(t1 * math.pi)
            if not is_top:
                intensity = 0.05 + 0.60 * math.sin(t1 * math.pi)
            col_hex = ColorMath.lerp_color("#000511", "#0044CC", intensity)
            poly_pts = []
            for j in range(len(pts_inner)):
                poly_pts.append(pts_inner[j].lerp(pts_outer[j], t1).to_array())
            for j in reversed(range(len(pts_inner))):
                poly_pts.append(pts_inner[j].lerp(pts_outer[j], t2).to_array())
            poly = Polygon(*poly_pts, stroke_width=0)
            poly.set_fill(color=col_hex, opacity=1.0)
            grp.add(poly)
        return grp

    def create_geometry(self) -> VGroup:
        grp = VGroup()
        ribbon_top = self.create_3d_ribbon(self.pts_clad_top, self.pts_jack_top, 25, True)
        ribbon_bot = self.create_3d_ribbon(self.pts_clad_bot, self.pts_jack_bot, 25, False)

        clad_arr = []
        for p in self.pts_clad_top:
            clad_arr.append(p.to_array())
        for p in reversed(self.pts_clad_bot):
            clad_arr.append(p.to_array())
        cladding_layer = Polygon(*clad_arr, stroke_width=0)
        cladding_layer.set_fill(color="#0B111A", opacity=1.0)

        core_arr = []
        for p in self.pts_core_top:
            core_arr.append(p.to_array())
        for p in reversed(self.pts_core_bot):
            core_arr.append(p.to_array())
        core_layer = Polygon(*core_arr, stroke_width=0)
        core_layer.set_fill(color="#000000", opacity=1.0)

        core_boundary_top = VMobject()
        core_boundary_top.set_points_as_corners([p.to_array() for p in self.pts_core_top])
        core_boundary_top.set_stroke(color="#2A3A4A", width=2.0, opacity=0.8)

        core_boundary_bot = VMobject()
        core_boundary_bot.set_points_as_corners([p.to_array() for p in self.pts_core_bot])
        core_boundary_bot.set_stroke(color="#2A3A4A", width=2.0, opacity=0.8)

        jack_boundary_top = VMobject()
        jack_boundary_top.set_points_as_corners([p.to_array() for p in self.pts_jack_top])
        jack_boundary_top.set_stroke(color="#113377", width=2.5, opacity=0.7)

        jack_boundary_bot = VMobject()
        jack_boundary_bot.set_points_as_corners([p.to_array() for p in self.pts_jack_bot])
        jack_boundary_bot.set_stroke(color="#113377", width=2.5, opacity=0.7)

        grp.add(ribbon_top, ribbon_bot, cladding_layer, core_layer, core_boundary_top, core_boundary_bot,
                jack_boundary_top, jack_boundary_bot)
        return grp


class PathSimulator:
    def __init__(self, model: FiberOpticModel):
        self.model = model
        self.engine = IntersectionEngine(model.boundaries)

    def trace(self, start_pos: Vector3D, start_dir: Vector3D, max_bounces: int) -> OpticalPath:
        path = OpticalPath()
        path.add_point(start_pos, False)
        ray = RaySegment(start_pos, start_dir)
        for _ in range(max_bounces):
            rec = self.engine.find_nearest(ray)
            if not rec.hit:
                path.add_point(ray.origin.add(ray.direction.mul(15.0)), False)
                break
            path.add_point(rec.point, True)
            if rec.point.x > 7.2:
                path.add_point(rec.point.add(ray.direction.mul(5.0)), False)
                break
            n1 = rec.mat_in.index
            n2 = rec.mat_out.index
            r = OpticsPhysics.calculate_fresnel(ray.direction, rec.normal, n1, n2)
            if r > 0.999:
                refl = ray.direction.reflect(rec.normal)
                ray = RaySegment(rec.point.add(refl.mul(2e-4)), refl)
            else:
                refr = OpticsPhysics.calculate_snell(ray.direction, rec.normal, n1, n2)
                ray = RaySegment(rec.point.add(refr.mul(2e-4)), refr)
        return path


class BeamGlowRenderer(VGroup):
    def __init__(self, path: OpticalPath):
        super().__init__()
        self.path = path
        self.layer_data = [
            {"col": "#FF1100", "wid": 85.0, "op": 0.04},
            {"col": "#FF2200", "wid": 55.0, "op": 0.08},
            {"col": "#FF6600", "wid": 30.0, "op": 0.25},
            {"col": "#FFCC00", "wid": 14.0, "op": 0.55},
            {"col": "#FFFF99", "wid": 6.0, "op": 0.90},
            {"col": "#FFFFFF", "wid": 2.5, "op": 1.00}
        ]
        self.polys = VGroup()
        for ld in self.layer_data:
            poly = Line(ORIGIN, ORIGIN)
            poly.set_stroke(color=ld["col"], width=ld["wid"], opacity=0.0)
            self.polys.add(poly)
        self.add(self.polys)

    def set_distance(self, dist: float):
        pts = self.path.extract_subpath(dist)
        if len(pts) < 2:
            for p in self.polys:
                p.set_points_as_corners([ORIGIN, ORIGIN])
                p.set_stroke(opacity=0.0)
            return
        for i, ld in enumerate(self.layer_data):
            poly = self.polys[i]
            poly.set_points_as_corners(pts)
            poly.set_stroke(color=ld["col"], width=ld["wid"], opacity=ld["op"])


class AmbientDustEffect(VGroup):
    def __init__(self, paths: List[OpticalPath], count: int):
        super().__init__()
        self.paths = paths
        self.count = count
        self.dots = VGroup()
        self.meta = []
        self.add(self.dots)
        self.init_particles()

    def init_particles(self):
        colors = ["#FFFFFF", "#FFFFCC", "#FFDD44", "#FF8800", "#FF2200", "#00AAFF"]
        for _ in range(self.count):
            d = Dot(radius=random.uniform(0.015, 0.04), color=random.choice(colors))
            d.set_fill(opacity=0.0)
            self.dots.add(d)
            self.meta.append({
                "path": random.choice(self.paths),
                "offset": random.uniform(-4.0, 0.0),
                "speed": random.uniform(0.7, 1.3)
            })

    def update_particles(self, base_dist: float):
        for i, dot in enumerate(self.dots):
            m = self.meta[i]
            local_dist = (base_dist + m["offset"]) * m["speed"]
            if local_dist <= 0 or local_dist >= m["path"].total_length:
                dot.set_fill(opacity=0.0)
                continue
            pts = m["path"].extract_subpath(local_dist)
            if len(pts) > 1:
                dot.move_to(pts[-1])
                dot.set_fill(opacity=random.uniform(0.5, 1.0))


class PureTotalInternalReflection(Scene):
    def construct(self):
        bg = Rectangle(width=config.frame_width * 2, height=config.frame_height * 2, stroke_width=0)
        bg.set_fill(color="#000000", opacity=1.0)
        self.add(bg)

        hud_elements = VGroup()
        hud_pos = [
            (np.array([-6.5, 3.5, 0]), 0),
            (np.array([6.5, 3.5, 0]), -math.pi / 2),
            (np.array([6.5, -3.5, 0]), math.pi),
            (np.array([-6.5, -3.5, 0]), math.pi / 2)
        ]
        for pos, rot in hud_pos:
            corner = VGroup()
            l1 = Line(ORIGIN, RIGHT * 0.5).set_stroke(color="#224466", width=3)
            l2 = Line(ORIGIN, UP * 0.5).set_stroke(color="#224466", width=3)
            corner.add(l1, l2).move_to(pos).rotate(rot, about_point=pos)
            hud_elements.add(corner)
        self.add(hud_elements)

        model = FiberOpticModel()
        visuals = model.create_geometry()
        self.add(visuals)
        sim = PathSimulator(model)
        paths = []
        max_len = 0.0
        c0 = model.curve.get_point(0.0)
        d0 = model.curve.get_tangent(0.0)
        n0 = model.curve.get_normal(0.0)

        emitter_ang = math.atan2(d0.y, d0.x)
        emitter_box = RoundedRectangle(corner_radius=0.15, width=1.0, height=1.6)
        emitter_box.set_stroke(color="#334455", width=2.5)
        emitter_box.set_fill(color="#050505", opacity=1.0)
        emitter_box.move_to(c0.sub(d0.mul(0.7)).to_array())
        emitter_box.rotate(emitter_ang)
        self.add(emitter_box)

        for r_size in [0.65, 0.85, 1.05]:
            arc = Arc(radius=r_size, start_angle=-math.pi / 3, angle=math.pi * 2 / 3,
                      arc_center=c0.sub(d0.mul(0.7)).to_array())
            arc.rotate(emitter_ang, about_point=c0.sub(d0.mul(0.7)).to_array())
            arc.set_stroke(color="#112233", width=2)
            self.add(arc)

        emitter_lens = Line(ORIGIN, ORIGIN)
        lens_p1 = c0.add(n0.mul(0.4)).sub(d0.mul(0.2)).to_array()
        lens_p2 = c0.sub(n0.mul(0.4)).sub(d0.mul(0.2)).to_array()
        emitter_lens.set_points_as_corners([lens_p1, lens_p2])
        emitter_lens.set_stroke(color="#2288FF", width=5.0, opacity=0.9)
        self.add(emitter_lens)

        emitter_glow = Line(ORIGIN, ORIGIN)
        emitter_glow.set_points_as_corners([lens_p1, lens_p2])
        emitter_glow.set_stroke(color="#55AAFF", width=15.0, opacity=0.3)
        self.add(emitter_glow)

        ray_params = [
            (0.0, 0.46),
            (0.0, -0.46),
            (0.18, 0.32),
            (-0.18, -0.32),
            (0.0, 0.0),
            (0.1, 0.15),
            (-0.1, -0.15)
        ]

        for offset, ang in ray_params:
            start_p = c0.add(n0.mul(offset)).sub(d0.mul(0.3))
            start_d = d0.rotate_z(ang)
            p = sim.trace(start_p, start_d, 160)
            paths.append(p)
            if p.total_length > max_len:
                max_len = p.total_length

        if max_len < 1.0:
            max_len = 25.0

        beams = VGroup()
        for p in paths:
            beams.add(BeamGlowRenderer(p))
        self.add(beams)

        dust = AmbientDustEffect(paths, 480)
        self.add(dust)

        timer = ValueTracker(0.0)

        def master_update(mob):
            val = timer.get_value()
            current_dist = max_len * (val / 15.0)
            for b in beams:
                b.set_distance(current_dist)
            dust.update_particles(current_dist)

        beams.add_updater(master_update)
        self.play(timer.animate.set_value(18.0), run_time=12.0, rate_func=linear)
        beams.remove_updater(master_update)
        self.wait(1)


if __name__ == "__main__":
    with tempconfig({"quality": "low_quality", "preview": True}):
        scene = PureTotalInternalReflection()
        scene.render()