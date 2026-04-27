from manim import *
import numpy as np
import math
import random
from typing import List, Tuple, Dict, Optional, Union


class Vector2D:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def mag(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def norm(self) -> 'Vector2D':
        m = self.mag()
        if m < 1e-12:
            return Vector2D(0.0, 0.0)
        return Vector2D(self.x / m, self.y / m)


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

    def reflect(self, normal: 'Vector3D') -> 'Vector3D':
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

    def rotate_x(self, angle: float) -> 'Matrix4x4':
        r_mat = Matrix4x4()
        c = math.cos(angle)
        s = math.sin(angle)
        r_mat.m[1][1] = c
        r_mat.m[1][2] = -s
        r_mat.m[2][1] = s
        r_mat.m[2][2] = c
        return self.multiply(r_mat)

    def rotate_y(self, angle: float) -> 'Matrix4x4':
        r_mat = Matrix4x4()
        c = math.cos(angle)
        s = math.sin(angle)
        r_mat.m[0][0] = c
        r_mat.m[0][2] = s
        r_mat.m[2][0] = -s
        r_mat.m[2][2] = c
        return self.multiply(r_mat)

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

    def conjugate(self) -> 'ComplexMath':
        return ComplexMath(self.real, -self.imag)

    def magnitude(self) -> float:
        return math.sqrt(self.real ** 2 + self.imag ** 2)

    def phase(self) -> float:
        return math.atan2(self.imag, self.real)


class Tensor2D:
    def __init__(self):
        self.t = [[0.0, 0.0], [0.0, 0.0]]

    def trace(self) -> float:
        return self.t[0][0] + self.t[1][1]


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


class OpticsPhysics:
    @staticmethod
    def calculate_fresnel_p(theta_i: float, n1: float, n2: float) -> float:
        sin_t_sq = (n1 / n2) ** 2 * (math.sin(theta_i) ** 2)
        if sin_t_sq >= 1.0:
            return 1.0
        theta_t = math.asin(math.sqrt(sin_t_sq))
        num = n2 * math.cos(theta_i) - n1 * math.cos(theta_t)
        den = n2 * math.cos(theta_i) + n1 * math.cos(theta_t)
        return (num / den) ** 2

    @staticmethod
    def calculate_evanescent_decay(theta_i: float, n1: float, n2: float, wavelength: float) -> float:
        sin_i = math.sin(theta_i)
        if (n1 / n2) * sin_i <= 1.0:
            return 0.0
        alpha = (2 * math.pi / wavelength) * n2 * math.sqrt((n1 / n2) ** 2 * sin_i ** 2 - 1.0)
        return alpha


class TravelingWaveBeam(VGroup):
    def __init__(self, start_pt: np.ndarray, end_pt: np.ndarray, color: str = "#00FFFF", amplitude: float = 0.35,
                 wave_len: float = 1.5, num_points: int = 200, **kwargs):
        super().__init__(**kwargs)
        self.start_pt = start_pt
        self.end_pt = end_pt
        self.amplitude = amplitude
        self.wave_len = wave_len
        self.num_points = num_points
        self.wave_color = color

        direction = end_pt - start_pt
        self.length = np.linalg.norm(direction)
        if self.length < 1e-8:
            self.dir_vec = np.array([1.0, 0.0, 0.0])
        else:
            self.dir_vec = direction / self.length

        self.perp_vec = np.array([-self.dir_vec[1], self.dir_vec[0], 0.0])

        self.core_glow_thick = Line(start_pt, end_pt, color=color, stroke_width=25, stroke_opacity=0.15)
        self.core_glow_thin = Line(start_pt, end_pt, color=color, stroke_width=8, stroke_opacity=0.4)
        self.core_center = Line(start_pt, end_pt, color=WHITE, stroke_width=2, stroke_opacity=0.8)

        self.wave_curve = VMobject()
        self.wave_curve.set_stroke(color=color, width=3.5, opacity=0.9)

        self.add(self.core_glow_thick, self.core_glow_thin, self.core_center, self.wave_curve)
        self.phase = 0.0
        self.update_geometry()

    def update_geometry(self):
        pts = []
        k = 2 * math.pi / self.wave_len
        for i in range(self.num_points):
            x = (i / (self.num_points - 1)) * self.length
            envelope = math.sin(math.pi * x / self.length)
            y = self.amplitude * envelope * math.sin(k * x - self.phase)
            pt = self.start_pt + self.dir_vec * x + self.perp_vec * y
            pts.append(pt)
        self.wave_curve.set_points_as_corners(pts)

    def advance_wave(self, dt: float, speed: float = 8.0):
        self.phase += dt * speed
        self.update_geometry()


class AmbientEvanescentHalo(VGroup):
    def __init__(self, color: str = "#00FFFF", max_height: float = 4.5, num_layers: int = 250, decay_rate: float = 5.5,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.max_height = max_height
        self.decay_rate = decay_rate
        self.zone_width = config.frame_width * 1.2

        self.gradient_layers = VGroup()
        layer_h = max_height / num_layers

        for i in range(num_layers):
            y_offset = i * layer_h
            opacity = 0.95 * math.exp(-decay_rate * (y_offset / max_height))
            rect = Rectangle(width=self.zone_width, height=layer_h, fill_color=color, fill_opacity=opacity,
                             stroke_width=0)
            rect.move_to(np.array([0, y_offset + layer_h / 2, 0]))
            self.gradient_layers.add(rect)

        self.add(self.gradient_layers)

        self.particles = VGroup()
        self.particle_data = []
        for _ in range(120):
            dot = Dot(radius=random.uniform(0.015, 0.04), color=WHITE, fill_opacity=0.0)
            self.particles.add(dot)
            self.particle_data.append({
                "x": random.uniform(-self.zone_width / 2, self.zone_width / 2),
                "y": random.uniform(0, max_height),
                "speed": random.uniform(0.15, 0.4),
                "phase": random.uniform(0, 2 * math.pi)
            })
        self.add(self.particles)

    def animate_halo(self, dt: float):
        for i, dot in enumerate(self.particles):
            meta = self.particle_data[i]
            meta["phase"] += dt * meta["speed"] * 4.0
            meta["y"] += math.sin(meta["phase"]) * 0.012

            if meta["y"] < 0:
                meta["y"] = 0
            elif meta["y"] > self.max_height:
                meta["y"] = self.max_height

            dot.move_to(np.array([meta["x"], meta["y"], 0]))
            base_op = math.exp(-self.decay_rate * (meta["y"] / self.max_height))
            dot.set_fill(opacity=base_op * (0.5 + 0.5 * math.sin(meta["phase"])))


class PreciselyAnchoredGraph(VGroup):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bg = RoundedRectangle(corner_radius=0.2, width=6.8, height=4.6, fill_color="#000814", fill_opacity=0.92,
                                   stroke_color="#00FFFF", stroke_width=2)

        self.ax = Axes(
            x_range=[0, 3, 1], y_range=[0, 1.2, 0.5],
            x_length=4.8, y_length=2.2,
            axis_config={"include_ticks": False, "color": "#4488AA"}
        )
        self.ax.move_to(self.bg.get_center()).shift(UP * 0.3)

        self.y_label = Text("E (电场强度)", font_size=18, color=WHITE, font="Microsoft YaHei")
        self.y_label.move_to(self.ax.c2p(0, 1.2) + UP * 0.4 + RIGHT * 0.5)

        self.x_label = Text("z (穿透深度)", font_size=18, color=WHITE, font="Microsoft YaHei")
        self.x_label.move_to(self.ax.c2p(3, 0) + RIGHT * 0.8 + DOWN * 0.25)

        self.formula = MathTex("E(z) = E_0 e^{-\\alpha z}", font_size=42, color="#00FFFF")
        self.formula.move_to(self.ax.c2p(1.5, 0) + DOWN * 1.0)

        self.add(self.bg, self.ax, self.y_label, self.x_label, self.formula)


class MacroEvanescentWaveScene(Scene):
    def construct(self):
        water_bg = Rectangle(width=config.frame_width, height=config.frame_height / 2, stroke_width=0,
                             fill_color="#001533", fill_opacity=1.0)
        water_bg.next_to(ORIGIN, DOWN, buff=0)

        air_bg = Rectangle(width=config.frame_width, height=config.frame_height / 2, stroke_width=0,
                           fill_color="#000000", fill_opacity=1.0)
        air_bg.next_to(ORIGIN, UP, buff=0)

        interface_bright = Line(LEFT * config.frame_width / 2, RIGHT * config.frame_width / 2, color="#00FFFF",
                                stroke_width=4)
        interface_glow = Line(LEFT * config.frame_width / 2, RIGHT * config.frame_width / 2, color="#00FFFF",
                              stroke_width=30, stroke_opacity=0.25)

        normal_dashed = DashedLine(DOWN * 4, UP * 4, color=WHITE, dash_length=0.15, stroke_opacity=0.6)

        label_w = Text("光密介质 (水)", font_size=28, color="#00AAFF", font="Microsoft YaHei").to_corner(DL).shift(
            UP * 0.5 + RIGHT * 0.5)
        label_a = Text("光疏介质 (空气)", font_size=28, color="#888888", font="Microsoft YaHei").to_corner(UL).shift(
            DOWN * 0.5 + RIGHT * 0.5)

        self.add(water_bg, air_bg, interface_glow, interface_bright, normal_dashed, label_w, label_a)

        angle = 60 * DEGREES
        beam_len = 12.0

        p_start = np.array([-beam_len * math.sin(angle), -beam_len * math.cos(angle), 0])
        p_end = np.array([beam_len * math.sin(angle), -beam_len * math.cos(angle), 0])

        laser_inc = TravelingWaveBeam(p_start, ORIGIN, color="#00FFFF", amplitude=0.35, wave_len=1.6)
        laser_ref = TravelingWaveBeam(ORIGIN, p_end, color="#00FFFF", amplitude=0.35, wave_len=1.6)

        def inc_anim(mob, dt):
            mob.advance_wave(dt, speed=10.0)

        def ref_anim(mob, dt):
            mob.advance_wave(dt, speed=10.0)

        self.play(FadeIn(laser_inc), run_time=1.5)
        laser_inc.add_updater(inc_anim)

        self.wait(0.2)

        impact_core = Circle(radius=0.05, color=WHITE, fill_color=WHITE, fill_opacity=1.0, stroke_width=0)
        self.play(FadeIn(impact_core), run_time=0.1)
        self.play(impact_core.animate.scale(40).set_fill(opacity=0.0), run_time=0.6)

        self.play(FadeIn(laser_ref), run_time=1.5)
        laser_ref.add_updater(ref_anim)

        ev_halo = AmbientEvanescentHalo(color="#00FFFF", max_height=4.8, num_layers=250, decay_rate=6.0)
        ev_halo.move_to(UP * 2.4)

        def halo_anim(mob, dt):
            mob.animate_halo(dt)

        self.play(FadeIn(ev_halo), run_time=2.5, rate_func=rate_functions.ease_out_cubic)
        ev_halo.add_updater(halo_anim)

        panel = PreciselyAnchoredGraph()
        panel.to_corner(UR).shift(DOWN * 0.5 + LEFT * 0.5)

        self.play(
            FadeIn(panel.bg),
            Create(panel.ax),
            run_time=1.5
        )
        self.play(
            Write(panel.x_label),
            Write(panel.y_label),
            Write(panel.formula),
            run_time=1.5
        )

        self.wait(0.5)

        decay_curve = panel.ax.plot(lambda x: math.exp(-3.5 * x), x_range=[0, 3], color="#00FFFF", stroke_width=4.5)
        decay_area = panel.ax.get_area(decay_curve, x_range=[0, 3], color="#00FFFF", opacity=0.35)

        self.play(
            Create(decay_curve),
            FadeIn(decay_area),
            run_time=4.0,
            rate_func=rate_functions.ease_in_out_sine
        )

        self.wait(6.0)

        laser_inc.remove_updater(inc_anim)
        laser_ref.remove_updater(ref_anim)
        ev_halo.remove_updater(halo_anim)


if __name__ == "__main__":
    with tempconfig({"quality": "high_quality", "preview": True}):
        scene = MacroEvanescentWaveScene()
        scene.render()