from manim import *
import numpy as np
import math
import random
from typing import List, Tuple, Dict, Optional, Union, Callable


class Vector2D:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def mag(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def mag_sq(self) -> float:
        return self.x ** 2 + self.y ** 2

    def norm(self) -> 'Vector2D':
        m = self.mag()
        if m < 1e-12:
            return Vector2D(0.0, 0.0)
        return Vector2D(self.x / m, self.y / m)

    def add(self, other: 'Vector2D') -> 'Vector2D':
        return Vector2D(self.x + other.x, self.y + other.y)

    def sub(self, other: 'Vector2D') -> 'Vector2D':
        return Vector2D(self.x - other.x, self.y - other.y)

    def mul(self, scalar: float) -> 'Vector2D':
        return Vector2D(self.x * scalar, self.y * scalar)

    def dot(self, other: 'Vector2D') -> float:
        return self.x * other.x + self.y * other.y


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

    def lerp(self, other: 'Vector3D', t: float) -> 'Vector3D':
        return self.add(other.sub(self).mul(t))

    def project_onto(self, other: 'Vector3D') -> 'Vector3D':
        o_norm = other.norm()
        return o_norm.mul(self.dot(o_norm))

    def reflect_plane(self, normal: 'Vector3D') -> 'Vector3D':
        n = normal.norm()
        return self.sub(n.mul(2.0 * self.dot(n)))


class Matrix3x3:
    def __init__(self):
        self.m = [[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]]

    def determinant(self) -> float:
        return (self.m[0][0] * (self.m[1][1] * self.m[2][2] - self.m[1][2] * self.m[2][1]) -
                self.m[0][1] * (self.m[1][0] * self.m[2][2] - self.m[1][2] * self.m[2][0]) +
                self.m[0][2] * (self.m[1][0] * self.m[2][1] - self.m[1][1] * self.m[2][0]))

    def transpose(self) -> 'Matrix3x3':
        res = Matrix3x3()
        for i in range(3):
            for j in range(3):
                res.m[i][j] = self.m[j][i]
        return res


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

    def translate(self, tx: float, ty: float, tz: float) -> 'Matrix4x4':
        t_mat = Matrix4x4()
        t_mat.m[0][3] = tx
        t_mat.m[1][3] = ty
        t_mat.m[2][3] = tz
        return self.multiply(t_mat)

    def scale(self, sx: float, sy: float, sz: float) -> 'Matrix4x4':
        s_mat = Matrix4x4()
        s_mat.m[0][0] = sx
        s_mat.m[1][1] = sy
        s_mat.m[2][2] = sz
        return self.multiply(s_mat)

    def rotate_z(self, angle: float) -> 'Matrix4x4':
        r_mat = Matrix4x4()
        c = math.cos(angle)
        s = math.sin(angle)
        r_mat.m[0][0] = c
        r_mat.m[0][1] = -s
        r_mat.m[1][0] = s
        r_mat.m[1][1] = c
        return self.multiply(r_mat)


class Quaternion:
    def __init__(self, w: float, x: float, y: float, z: float):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def multiply(self, q: 'Quaternion') -> 'Quaternion':
        w = self.w * q.w - self.x * q.x - self.y * q.y - self.z * q.z
        x = self.w * q.x + self.x * q.w + self.y * q.z - self.z * q.y
        y = self.w * q.y - self.x * q.z + self.y * q.w + self.z * q.x
        z = self.w * q.z + self.x * q.y - self.y * q.x + self.z * q.w
        return Quaternion(w, x, y, z)

    def normalize(self) -> 'Quaternion':
        mag = math.sqrt(self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2)
        if mag < 1e-12:
            return Quaternion(1.0, 0.0, 0.0, 0.0)
        return Quaternion(self.w / mag, self.x / mag, self.y / mag, self.z / mag)

    def inverse(self) -> 'Quaternion':
        mag_sq = self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2
        if mag_sq < 1e-12:
            return Quaternion(1.0, 0.0, 0.0, 0.0)
        return Quaternion(self.w / mag_sq, -self.x / mag_sq, -self.y / mag_sq, -self.z / mag_sq)


class ComplexMath:
    def __init__(self, real: float, imag: float):
        self.real = real
        self.imag = imag

    def add(self, other: 'ComplexMath') -> 'ComplexMath':
        return ComplexMath(self.real + other.real, self.imag + other.imag)

    def sub(self, other: 'ComplexMath') -> 'ComplexMath':
        return ComplexMath(self.real - other.real, self.imag - other.imag)

    def mul(self, other: 'ComplexMath') -> 'ComplexMath':
        r = self.real * other.real - self.imag * other.imag
        i = self.real * other.imag + self.imag * other.real
        return ComplexMath(r, i)

    def div(self, other: 'ComplexMath') -> 'ComplexMath':
        denom = other.real ** 2 + other.imag ** 2
        if denom < 1e-12:
            return ComplexMath(0.0, 0.0)
        r = (self.real * other.real + self.imag * other.imag) / denom
        i = (self.imag * other.real - self.real * other.imag) / denom
        return ComplexMath(r, i)

    def magnitude(self) -> float:
        return math.sqrt(self.real ** 2 + self.imag ** 2)

    def phase(self) -> float:
        return math.atan2(self.imag, self.real)


class Tensor3D:
    def __init__(self):
        self.t = [[0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0]]

    def trace(self) -> float:
        return self.t[0][0] + self.t[1][1] + self.t[2][2]

    def add(self, other: 'Tensor3D') -> 'Tensor3D':
        res = Tensor3D()
        for i in range(3):
            for j in range(3):
                res.t[i][j] = self.t[i][j] + other.t[i][j]
        return res

    def multiply_vector(self, vec: Vector3D) -> Vector3D:
        x = self.t[0][0] * vec.x + self.t[0][1] * vec.y + self.t[0][2] * vec.z
        y = self.t[1][0] * vec.x + self.t[1][1] * vec.y + self.t[1][2] * vec.z
        z = self.t[2][0] * vec.x + self.t[2][1] * vec.y + self.t[2][2] * vec.z
        return Vector3D(x, y, z)


class JonesVector:
    def __init__(self, ex: ComplexMath, ey: ComplexMath):
        self.ex = ex
        self.ey = ey

    def intensity(self) -> float:
        return self.ex.magnitude() ** 2 + self.ey.magnitude() ** 2

    def normalize(self) -> 'JonesVector':
        intensity = self.intensity()
        if intensity < 1e-12:
            return JonesVector(ComplexMath(0.0, 0.0), ComplexMath(0.0, 0.0))
        factor = math.sqrt(intensity)
        return JonesVector(
            ComplexMath(self.ex.real / factor, self.ex.imag / factor),
            ComplexMath(self.ey.real / factor, self.ey.imag / factor)
        )


class JonesMatrix:
    def __init__(self, j00: ComplexMath, j01: ComplexMath, j10: ComplexMath, j11: ComplexMath):
        self.j00 = j00
        self.j01 = j01
        self.j10 = j10
        self.j11 = j11

    def multiply_vector(self, vec: JonesVector) -> JonesVector:
        nx = self.j00.mul(vec.ex).add(self.j01.mul(vec.ey))
        ny = self.j10.mul(vec.ex).add(self.j11.mul(vec.ey))
        return JonesVector(nx, ny)

    def multiply_matrix(self, other: 'JonesMatrix') -> 'JonesMatrix':
        n00 = self.j00.mul(other.j00).add(self.j01.mul(other.j10))
        n01 = self.j00.mul(other.j01).add(self.j01.mul(other.j11))
        n10 = self.j10.mul(other.j00).add(self.j11.mul(other.j10))
        n11 = self.j10.mul(other.j01).add(self.j11.mul(other.j11))
        return JonesMatrix(n00, n01, n10, n11)


class StokesVector:
    def __init__(self, s0: float, s1: float, s2: float, s3: float):
        self.s0 = s0
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3

    def degree_of_polarization(self) -> float:
        if self.s0 < 1e-12:
            return 0.0
        return math.sqrt(self.s1 ** 2 + self.s2 ** 2 + self.s3 ** 2) / self.s0


class MuellerMatrix:
    def __init__(self):
        self.m = [[0.0 for _ in range(4)] for _ in range(4)]

    def set_identity(self):
        for i in range(4):
            for j in range(4):
                self.m[i][j] = 1.0 if i == j else 0.0

    def multiply_vector(self, vec: StokesVector) -> StokesVector:
        v_arr = [vec.s0, vec.s1, vec.s2, vec.s3]
        res = [0.0, 0.0, 0.0, 0.0]
        for i in range(4):
            for j in range(4):
                res[i] += self.m[i][j] * v_arr[j]
        return StokesVector(res[0], res[1], res[2], res[3])


class ElectromagneticField:
    def __init__(self, e_vec: Vector3D, b_vec: Vector3D):
        self.electric = e_vec
        self.magnetic = b_vec

    def poynting_vector(self) -> Vector3D:
        return self.electric.cross(self.magnetic)

    def energy_density(self, epsilon: float, mu: float) -> float:
        return 0.5 * (epsilon * self.electric.mag_sq() + (1.0 / mu) * self.magnetic.mag_sq())


class DipoleRadiationEngine:
    def __init__(self, dipole_moment: float, frequency: float, c: float = 3e8):
        self.dipole_moment = dipole_moment
        self.frequency = frequency
        self.c = c
        self.omega = 2 * math.pi * frequency
        self.k = self.omega / c

    def intensity_at_angle(self, theta: float, distance: float) -> float:
        if distance < 1e-12:
            return 0.0
        amplitude_factor = (self.dipole_moment * self.omega ** 4) / (32 * math.pi ** 2 * self.c ** 3 * 8.854e-12)
        return amplitude_factor * (math.sin(theta) ** 2) / (distance ** 2)


class OpticsPhysics:
    @staticmethod
    def brewster_angle(n1: float, n2: float) -> float:
        return math.atan2(n2, n1)

    @staticmethod
    def snells_law_angle(theta_i: float, n1: float, n2: float) -> float:
        val = (n1 / n2) * math.sin(theta_i)
        if val > 1.0 or val < -1.0:
            return math.pi / 2.0
        return math.asin(val)


class AdvancedTransverseWave(VGroup):
    def __init__(self, start_pt: np.ndarray, end_pt: np.ndarray, amplitude: float, wavelength: float, num_arrows: int,
                 color: str, **kwargs):
        super().__init__(**kwargs)
        self.start_pt = start_pt
        self.end_pt = end_pt
        self.amplitude = amplitude
        self.wavelength = wavelength
        self.num_arrows = num_arrows
        self.wave_color = color

        vec = end_pt - start_pt
        self.length = np.linalg.norm(vec)
        if self.length < 1e-12:
            self.dir_vec = np.array([1.0, 0.0, 0.0])
        else:
            self.dir_vec = vec / self.length

        self.perp_vec = np.array([-self.dir_vec[1], self.dir_vec[0], 0.0])

        self.central_glow = Line(start_pt, end_pt, color=color, stroke_width=12, stroke_opacity=0.15)
        self.central_core = Line(start_pt, end_pt, color=WHITE, stroke_width=1.5, stroke_opacity=0.6)
        self.add(self.central_glow, self.central_core)

        self.wave_curve = VMobject()
        self.wave_curve.set_stroke(color=color, width=3.5, opacity=0.85)
        self.add(self.wave_curve)

        self.arrows = VGroup()
        for _ in range(num_arrows):
            base_temp = start_pt
            tip_temp = start_pt + self.perp_vec * 0.001
            arr = Arrow(base_temp, tip_temp, buff=0, color=color, stroke_width=2.5, max_tip_length_to_length_ratio=0.2)
            self.arrows.add(arr)
        self.add(self.arrows)

        self.phase = 0.0
        self.update_wave()

    def update_wave(self):
        pts = []
        steps = int(self.length * 30)
        k = 2 * math.pi / self.wavelength
        for i in range(steps + 1):
            x = (i / steps) * self.length
            y = self.amplitude * math.sin(k * x - self.phase)
            pt = self.start_pt + self.dir_vec * x + self.perp_vec * y
            pts.append(pt)

        if len(pts) > 1:
            self.wave_curve.set_points_as_corners(pts)

        for i, arr in enumerate(self.arrows):
            x = (i / (self.num_arrows - 1)) * self.length if self.num_arrows > 1 else 0
            y = self.amplitude * math.sin(k * x - self.phase)
            base_pt = self.start_pt + self.dir_vec * x
            tip_pt = base_pt + self.perp_vec * y
            dist = np.linalg.norm(tip_pt - base_pt)

            if dist > 0.01:
                arr.put_start_and_end_on(base_pt, tip_pt)
                arr.set_opacity(0.8 + 0.2 * (dist / self.amplitude))
            else:
                arr.put_start_and_end_on(base_pt, base_pt + self.perp_vec * 0.001)
                arr.set_opacity(0.0)

    def advance_time(self, dt: float, speed: float = 4.0):
        self.phase += dt * speed
        self.update_wave()


class AtomCore(VGroup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.core = Dot(radius=0.12, color=WHITE)
        self.halo1 = Circle(radius=0.25, color="#00FFFF", stroke_width=2.5, fill_color="#00FFFF", fill_opacity=0.3)
        self.halo2 = Circle(radius=0.45, color="#00FFFF", stroke_width=1.0, fill_opacity=0.0)
        self.add(self.halo2, self.halo1, self.core)
        self.time = 0.0

    def update_glow(self, dt: float):
        self.time += dt * 6.0
        scale_factor = 1.0 + 0.15 * math.sin(self.time)
        self.halo1.scale(scale_factor / self.halo1.radius * 0.25)
        self.halo2.scale((scale_factor + 0.2) / self.halo2.radius * 0.45)
        self.halo2.set_stroke(opacity=max(0.0, 1.0 - math.sin(self.time)))


class DipoleOscillator(VGroup):
    def __init__(self, direction: np.ndarray, color: str = "#FF00FF", **kwargs):
        super().__init__(**kwargs)
        self.direction = direction / np.linalg.norm(direction)
        self.base_color = color
        self.arrow1 = Arrow(ORIGIN, self.direction * 0.001, buff=0, color=color, stroke_width=5,
                            max_tip_length_to_length_ratio=0.25)
        self.arrow2 = Arrow(ORIGIN, -self.direction * 0.001, buff=0, color=color, stroke_width=5,
                            max_tip_length_to_length_ratio=0.25)
        self.add(self.arrow1, self.arrow2)
        self.max_length = 2.2

    def update_oscillation(self, phase: float):
        val = math.sin(phase)
        mag = abs(val) * self.max_length
        if mag > 0.01:
            if val > 0:
                self.arrow1.put_start_and_end_on(ORIGIN, self.direction * mag)
                self.arrow2.put_start_and_end_on(ORIGIN, -self.direction * mag)
            else:
                self.arrow1.put_start_and_end_on(ORIGIN, -self.direction * mag)
                self.arrow2.put_start_and_end_on(ORIGIN, self.direction * mag)
            self.arrow1.set_opacity(1.0)
            self.arrow2.set_opacity(1.0)
        else:
            self.arrow1.put_start_and_end_on(ORIGIN, self.direction * 0.001)
            self.arrow2.put_start_and_end_on(ORIGIN, -self.direction * 0.001)
            self.arrow1.set_opacity(0.0)
            self.arrow2.set_opacity(0.0)


class DynamicRadiationLobe(VMobject):
    def __init__(self, axis_dir: np.ndarray, max_radius: float = 3.8, color: str = "#FF00FF", **kwargs):
        super().__init__(**kwargs)
        self.axis_angle = math.atan2(axis_dir[1], axis_dir[0])
        self.max_radius = max_radius
        self.pattern_color = color
        self.set_stroke(color=color, width=4)
        self.set_fill(color=color, opacity=0.2)
        self.build_pattern()

    def build_pattern(self):
        pts = []
        steps = 250
        for i in range(steps + 1):
            t = i * 2 * math.pi / steps
            r = self.max_radius * (math.sin(t - self.axis_angle) ** 2)
            x = r * math.cos(t)
            y = r * math.sin(t)
            pts.append(np.array([x, y, 0]))
        self.set_points_as_corners(pts)


class RadiationRipple(VGroup):
    def __init__(self, axis_dir: np.ndarray, color: str = "#FF00FF", **kwargs):
        super().__init__(**kwargs)
        self.lobe1 = DynamicRadiationLobe(axis_dir, max_radius=0.1, color=color)
        self.lobe1.set_fill(opacity=0.0)
        self.lobe2 = DynamicRadiationLobe(axis_dir, max_radius=0.1, color=color)
        self.lobe2.set_fill(opacity=0.0)
        self.lobe3 = DynamicRadiationLobe(axis_dir, max_radius=0.1, color=color)
        self.lobe3.set_fill(opacity=0.0)
        self.add(self.lobe1, self.lobe2, self.lobe3)
        self.time = 0.0

    def update_ripples(self, dt: float):
        self.time += dt * 1.5
        r1 = (self.time % 2.0) * 3.5
        r2 = ((self.time + 0.66) % 2.0) * 3.5
        r3 = ((self.time + 1.33) % 2.0) * 3.5

        self.lobe1.scale(r1 / max(self.lobe1.max_radius, 0.001))
        self.lobe1.max_radius = r1
        self.lobe1.set_stroke(opacity=max(0.0, 1.0 - r1 / 3.5))

        self.lobe2.scale(r2 / max(self.lobe2.max_radius, 0.001))
        self.lobe2.max_radius = r2
        self.lobe2.set_stroke(opacity=max(0.0, 1.0 - r2 / 3.5))

        self.lobe3.scale(r3 / max(self.lobe3.max_radius, 0.001))
        self.lobe3.max_radius = r3
        self.lobe3.set_stroke(opacity=max(0.0, 1.0 - r3 / 3.5))


class RightAngleMarker(VGroup):
    def __init__(self, v1: np.ndarray, v2: np.ndarray, size: float = 0.6, color: str = YELLOW, **kwargs):
        super().__init__(**kwargs)
        u1 = v1 / np.linalg.norm(v1)
        u2 = v2 / np.linalg.norm(v2)
        p1 = u1 * size
        p2 = u2 * size
        p3 = p1 + p2
        l1 = Line(p1, p3, color=color, stroke_width=3.5)
        l2 = Line(p2, p3, color=color, stroke_width=3.5)
        self.bisector = p3 / np.linalg.norm(p3)
        self.corner_pt = p3
        self.add(l1, l2)


class ZeroIntensityMarker(VGroup):
    def __init__(self, size: float = 0.7, **kwargs):
        super().__init__(**kwargs)
        self.circle = Circle(radius=size * 0.6, color=RED, stroke_width=4, fill_color="#220000", fill_opacity=0.8)
        self.diag = Line(np.array([-size * 0.4, -size * 0.4, 0]), np.array([size * 0.4, size * 0.4, 0]), color=RED,
                         stroke_width=4)
        self.text = MathTex("I = 0", font_size=28, color=WHITE).move_to(DOWN * size * 1.1)
        self.add(self.circle, self.diag, self.text)


class MicroscopicBrewsterScene(Scene):
    def construct(self):
        bg_air = Rectangle(width=config.frame_width, height=config.frame_height / 2, fill_color="#030303",
                           fill_opacity=1.0, stroke_width=0)
        bg_air.move_to(np.array([0.0, config.frame_height / 4, 0.0]))

        bg_glass = Rectangle(width=config.frame_width, height=config.frame_height / 2, fill_color="#001433",
                             fill_opacity=1.0, stroke_width=0)
        bg_glass.move_to(np.array([0.0, -config.frame_height / 4, 0.0]))

        interface = Line(np.array([-config.frame_width / 2, 0.0, 0.0]), np.array([config.frame_width / 2, 0.0, 0.0]),
                         color="#00FFFF", stroke_width=2.5)
        interface_glow = Line(np.array([-config.frame_width / 2, 0.0, 0.0]),
                              np.array([config.frame_width / 2, 0.0, 0.0]), color="#00FFFF", stroke_width=18,
                              stroke_opacity=0.25)

        normal_line = DashedLine(np.array([0.0, -4.0, 0.0]), np.array([0.0, 4.0, 0.0]), color=WHITE, dash_length=0.15,
                                 stroke_opacity=0.6)

        n1 = 1.0
        n2 = 1.5
        theta_B = OpticsPhysics.brewster_angle(n1, n2)
        theta_t = OpticsPhysics.snells_law_angle(theta_B, n1, n2)

        beam_len = 12.0
        inc_start = np.array([-beam_len * math.sin(theta_B), beam_len * math.cos(theta_B), 0.0])
        refracted_end = np.array([beam_len * math.sin(theta_t), -beam_len * math.cos(theta_t), 0.0])
        expected_ref_end = np.array([beam_len * math.sin(theta_B), beam_len * math.cos(theta_B), 0.0])

        dipole_dir = np.array([math.cos(theta_t), math.sin(theta_t), 0.0])

        wave_inc = AdvancedTransverseWave(inc_start, ORIGIN, amplitude=0.45, wavelength=2.2, num_arrows=25,
                                          color="#00FFFF")
        wave_trans = AdvancedTransverseWave(ORIGIN, refracted_end, amplitude=0.35, wavelength=1.46, num_arrows=25,
                                            color="#00FFFF")

        atom = AtomCore()
        atom.move_to(ORIGIN)

        dipole = DipoleOscillator(dipole_dir, color="#FF00FF")
        dipole.move_to(ORIGIN)

        global_time = ValueTracker(0.0)

        def inc_updater(mob, dt):
            mob.advance_time(dt, speed=5.5)

        def trans_updater(mob, dt):
            mob.advance_time(dt, speed=5.5)

        def atom_updater(mob, dt):
            mob.update_glow(dt)

        def dipole_updater(mob):
            t = global_time.get_value()
            mob.update_oscillation(t * 5.5)

        def time_updater(mob, dt):
            mob.set_value(mob.get_value() + dt)

        self.add(bg_air, bg_glass, interface_glow, interface, normal_line)
        self.play(FadeIn(atom), run_time=1.0)

        wave_inc.add_updater(inc_updater)
        self.play(FadeIn(wave_inc, scale=1.0), run_time=1.5)

        wave_trans.add_updater(trans_updater)
        global_time.add_updater(time_updater)
        dipole.add_updater(dipole_updater)
        atom.add_updater(atom_updater)

        self.play(FadeIn(wave_trans, scale=1.0), FadeIn(dipole, scale=1.0), run_time=1.5)
        self.wait(3.0)

        wave_inc.remove_updater(inc_updater)
        wave_trans.remove_updater(trans_updater)
        global_time.remove_updater(time_updater)
        dipole.remove_updater(dipole_updater)

        dipole.update_oscillation(math.pi / 2.0)

        expected_line = DashedLine(ORIGIN, expected_ref_end, color="#FF8800", dash_length=0.15, stroke_width=4)
        self.play(Create(expected_line), run_time=1.5)

        right_angle = RightAngleMarker(refracted_end, expected_ref_end, size=0.6, color=YELLOW)
        arc_text = MathTex("90^\\circ", font_size=38, color=YELLOW)

        text_pos = right_angle.corner_pt + right_angle.bisector * 0.5
        arc_text.move_to(text_pos)

        self.play(Create(right_angle), Write(arc_text), run_time=1.5)

        rad_pattern = DynamicRadiationLobe(dipole_dir, max_radius=3.8, color="#FF00FF")
        rad_pattern.move_to(ORIGIN)
        rad_ripples = RadiationRipple(dipole_dir, color="#FF00FF")
        rad_ripples.move_to(ORIGIN)

        def ripple_updater(mob, dt):
            mob.update_ripples(dt)

        self.play(Create(rad_pattern), run_time=2.0)
        rad_ripples.add_updater(ripple_updater)
        self.add(rad_ripples)

        self.wait(1.5)

        axis_line = Line(np.array([-4.5 * dipole_dir[0], -4.5 * dipole_dir[1], 0.0]),
                         np.array([4.5 * dipole_dir[0], 4.5 * dipole_dir[1], 0.0]), color=WHITE, stroke_width=6)
        self.play(Create(axis_line), run_time=1.0)
        self.play(FadeOut(axis_line), run_time=0.4)
        self.play(FadeIn(axis_line), run_time=0.4)
        self.play(FadeOut(axis_line), run_time=0.4)

        z_mark = ZeroIntensityMarker(size=0.8)
        z_mark.move_to(np.array([3.0 * math.sin(theta_B), 3.0 * math.cos(theta_B), 0.0]))
        self.play(FadeIn(z_mark, scale=0.1), run_time=0.5)
        self.play(z_mark.animate.scale(1.2), run_time=0.2)
        self.play(z_mark.animate.scale(1 / 1.2), run_time=0.2)

        ui_bg = RoundedRectangle(corner_radius=0.15, width=9.5, height=1.0, fill_color="#000000", fill_opacity=0.9,
                                 stroke_color="#4488AA", stroke_width=2)
        ui_bg.move_to(np.array([0.0, 3.4, 0.0]))

        ui_text1 = Text("偶极子无轴向辐射", font_size=26, color=WHITE, font="Microsoft YaHei").move_to(
            ui_bg.get_center() + LEFT * 2.5)
        ui_arrow1 = MathTex("\\rightarrow", font_size=32, color=WHITE).next_to(ui_text1, RIGHT, buff=0.3)
        ui_text2 = Text("反射波为零", font_size=26, color=WHITE, font="Microsoft YaHei").next_to(ui_arrow1, RIGHT,
                                                                                                 buff=0.3)
        ui_arrow2 = MathTex("\\rightarrow", font_size=32, color=WHITE).next_to(ui_text2, RIGHT, buff=0.3)
        ui_text3 = Text("p偏振光全透射", font_size=26, color="#00FFFF", font="Microsoft YaHei").next_to(ui_arrow2,
                                                                                                        RIGHT, buff=0.3)

        ui_group = VGroup(ui_bg, ui_text1, ui_arrow1, ui_text2, ui_arrow2, ui_text3)
        ui_group.move_to(np.array([0.0, 3.4, 0.0]))

        self.play(FadeIn(ui_bg, shift=DOWN * 0.3), run_time=0.8)
        self.play(Write(ui_text1), run_time=0.8)
        self.play(FadeIn(ui_arrow1), Write(ui_text2), run_time=0.8)
        self.play(FadeIn(ui_arrow2), Write(ui_text3), run_time=0.8)

        self.wait(4.0)

        rad_ripples.remove_updater(ripple_updater)
        atom.remove_updater(atom_updater)


if __name__ == "__main__":
    with tempconfig({"quality": "high_quality", "preview": True}):
        scene = MicroscopicBrewsterScene()
        scene.render()