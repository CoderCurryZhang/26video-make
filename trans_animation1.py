from manim import *
import numpy as np
import math


class GlowLine(VGroup):
    def __init__(self, start, end, base_color, **kwargs):
        super().__init__(**kwargs)
        self.layer_data = [
            {"width": 38.0, "opacity": 0.015},
            {"width": 24.0, "opacity": 0.04},
            {"width": 12.0, "opacity": 0.12},
            {"width": 5.0, "opacity": 0.40},
            {"width": 1.5, "opacity": 1.00}
        ]
        self.lines = VGroup()
        for ld in self.layer_data:
            self.lines.add(
                Line(
                    start,
                    end,
                    color=base_color,
                    stroke_width=ld["width"],
                    stroke_opacity=ld["opacity"]
                )
            )
        self.add(self.lines)

    def put_start_and_end_on(self, start, end):
        for line in self.lines:
            line.put_start_and_end_on(start, end)

    def set_glow_opacity(self, alpha):
        alpha = max(0.0, min(1.0, alpha))
        for line, ld in zip(self.lines, self.layer_data):
            line.set_stroke(opacity=ld["opacity"] * alpha)


class RayArrow(Polygon):
    def __init__(self, color, **kwargs):
        super().__init__(
            np.array([-0.18, -0.28, 0]),
            np.array([0.18, -0.28, 0]),
            np.array([0, 0.28, 0]),
            color=color,
            fill_opacity=1.0,
            stroke_width=0,
            **kwargs
        )
        self.base_color = color

    def update_pose(self, start_pt, end_pt, alpha, visible_boost):
        if visible_boost < 0.001:
            self.set_fill(opacity=0)
            return
        self.set_fill(opacity=visible_boost)
        direction = end_pt - start_pt
        length = np.linalg.norm(direction)
        if length < 0.001:
            self.set_fill(opacity=0)
            return
        unit_dir = direction / length
        angle = math.atan2(unit_dir[1], unit_dir[0])
        pos = start_pt + direction * alpha

        p1 = np.array([-0.15, -0.22, 0])
        p2 = np.array([0.15, -0.22, 0])
        p3 = np.array([0, 0.22, 0])

        c = math.cos(angle - math.pi / 2)
        s = math.sin(angle - math.pi / 2)

        def rot(p):
            return np.array([p[0] * c - p[1] * s, p[0] * s + p[1] * c, 0])

        self.become(
            Polygon(
                rot(p1) + pos,
                rot(p2) + pos,
                rot(p3) + pos,
                color=self.base_color,
                fill_opacity=visible_boost,
                stroke_width=0
            )
        )


class LightSource(VGroup):
    def __init__(self, color, **kwargs):
        super().__init__(**kwargs)
        self.core = Dot(radius=0.12, color=WHITE)
        self.halo1 = Dot(radius=0.25, color=color, fill_opacity=0.7)
        self.halo2 = Dot(radius=0.55, color=color, fill_opacity=0.25)
        self.halo3 = Dot(radius=0.90, color=color, fill_opacity=0.08)
        self.add(self.halo3, self.halo2, self.halo1, self.core)


class TotalInternalReflectionProcess(Scene):
    def construct(self):
        n1 = 1.33
        n2 = 1.00
        critical_angle = math.asin(n2 / n1)

        water_rect = Rectangle(width=config.frame_width, height=config.frame_height / 2, stroke_width=0)
        water_rect.set_fill(color="#002244", opacity=0.6)
        water_rect.next_to(ORIGIN, DOWN, buff=0)

        air_rect = Rectangle(width=config.frame_width, height=config.frame_height / 2, stroke_width=0)
        air_rect.set_fill(color="#112233", opacity=0.15)
        air_rect.next_to(ORIGIN, UP, buff=0)

        boundary = Line(LEFT * 8, RIGHT * 8, color="#88AACC", stroke_width=3)
        normal = DashedLine(DOWN * 5, UP * 5, color="#FFFFFF", dash_length=0.12, stroke_opacity=0.5)

        water_text = Text("水 (光密介质)", font_size=28, weight=BOLD, color="#AADDFF")
        water_label = MathTex("n_1 = 1.33", color="#AADDFF")
        water_group = VGroup(water_text, water_label).arrange(DOWN, buff=0.2)
        water_group.to_corner(DR).shift(LEFT * 0.5 + UP * 0.5)

        air_text = Text("空气 (光疏介质)", font_size=28, weight=BOLD, color="#FFFFFF")
        air_label = MathTex("n_2 = 1.00", color="#FFFFFF")
        air_group = VGroup(air_text, air_label).arrange(DOWN, buff=0.2)
        air_group.to_corner(UR).shift(LEFT * 0.5 + DOWN * 0.5)

        tracker = ValueTracker(30 * DEGREES)

        col_inc = "#FF2200"
        col_ref = "#FF8800"
        col_tra = "#00AAFF"

        incident_ray = GlowLine(ORIGIN, ORIGIN, base_color=col_inc)
        reflected_ray = GlowLine(ORIGIN, ORIGIN, base_color=col_ref)
        refracted_ray = GlowLine(ORIGIN, ORIGIN, base_color=col_tra)

        arr_inc = RayArrow(color=col_inc)
        arr_ref = RayArrow(color=col_ref)
        arr_tra = RayArrow(color=col_tra)

        source = LightSource(color=col_inc)

        info_panel_bg = RoundedRectangle(corner_radius=0.15, width=4.0, height=2.2, color="#445566",
                                         fill_color="#000000", fill_opacity=0.8)
        info_panel_bg.to_corner(DL).shift(RIGHT * 0.5 + UP * 0.5)

        val_inc = Text("入射角: 00.0°", font_size=20, color=col_inc)
        val_tra = Text("折射角: 00.0°", font_size=20, color=col_tra)
        val_cri = Text(f"临界角: {critical_angle * 180 / math.pi:.1f}°", font_size=20, color=YELLOW)

        text_group = VGroup(val_inc, val_tra, val_cri).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        text_group.move_to(info_panel_bg.get_center())
        info_panel = VGroup(info_panel_bg, text_group)

        def update_incident(mob):
            theta = tracker.get_value()
            start_pt = np.array([-9 * math.sin(theta), -9 * math.cos(theta), 0])
            mob.put_start_and_end_on(start_pt, ORIGIN)
            arr_inc.update_pose(start_pt, ORIGIN, 0.45, 1.0)
            source.move_to(start_pt)
            val_inc.become(Text(f"入射光线角度: {theta * 180 / math.pi:.1f}°", font_size=20, color=col_inc).move_to(
                val_inc.get_center(), aligned_edge=LEFT))

        def update_reflected(mob):
            theta = tracker.get_value()
            val = (n1 / n2) * math.sin(theta)
            end_pt = np.array([9 * math.sin(theta), -9 * math.cos(theta), 0])
            mob.put_start_and_end_on(ORIGIN, end_pt)
            boost = min(1.0, 0.15 + 0.85 * (val ** 6))
            if val >= 1.0:
                boost = 1.0
            mob.set_glow_opacity(boost)
            arr_ref.update_pose(ORIGIN, end_pt, 0.55, boost)

        def update_refracted(mob):
            theta = tracker.get_value()
            val = (n1 / n2) * math.sin(theta)
            if val >= 0.9999:
                mob.set_glow_opacity(0)
                mob.put_start_and_end_on(ORIGIN, ORIGIN)
                arr_tra.update_pose(ORIGIN, ORIGIN, 0.5, 0.0)
                val_tra.become(
                    Text("折射光线角度: 无 (全反射)", font_size=20, color=col_ref).move_to(val_tra.get_center(),
                                                                                           aligned_edge=LEFT))
            else:
                theta_t = math.asin(val)
                end_pt = np.array([9 * math.sin(theta_t), 9 * math.cos(theta_t), 0])
                mob.put_start_and_end_on(ORIGIN, end_pt)
                fade = max(0.0, 1.0 - (val ** 18))
                mob.set_glow_opacity(fade)
                arr_tra.update_pose(ORIGIN, end_pt, 0.55, fade)
                val_tra.become(
                    Text(f"折射光线角度: {theta_t * 180 / math.pi:.1f}°", font_size=20, color=col_tra).move_to(
                        val_tra.get_center(), aligned_edge=LEFT))

        incident_ray.add_updater(update_incident)
        reflected_ray.add_updater(update_reflected)
        refracted_ray.add_updater(update_refracted)

        arc = always_redraw(
            lambda: Arc(
                radius=1.5,
                start_angle=-math.pi / 2,
                angle=tracker.get_value(),
                color=WHITE,
                stroke_width=2.5
            )
        )

        def update_theta_label(mob):
            theta = tracker.get_value()
            r = 2.1
            bisector = -math.pi / 2 + theta / 2
            pos = np.array([r * math.cos(bisector), r * math.sin(bisector), 0])
            mob.move_to(pos)

        theta_label = MathTex(r"\theta_1", font_size=32)
        theta_label.add_updater(update_theta_label)

        self.play(
            FadeIn(water_rect), FadeIn(air_rect),
            Create(boundary), Create(normal),
            Write(water_group), Write(air_group),
            FadeIn(info_panel),
            run_time=1.5
        )

        self.play(
            FadeIn(source),
            FadeIn(incident_ray), FadeIn(arr_inc),
            FadeIn(reflected_ray), FadeIn(arr_ref),
            FadeIn(refracted_ray), FadeIn(arr_tra),
            Create(arc),
            FadeIn(theta_label),
            run_time=1.5
        )

        target_angle = critical_angle + 16 * DEGREES
        self.play(
            tracker.animate.set_value(target_angle),
            run_time=5.0,
            rate_func=linear
        )

        self.wait(1)

        formula_bg = RoundedRectangle(corner_radius=0.2, width=5.5, height=2.0, color=YELLOW, fill_color=BLACK,
                                      fill_opacity=0.85)
        formula_bg.to_corner(UL).shift(RIGHT * 0.5 + DOWN * 0.5)

        formula_text = Text("全反射临界角公式", font_size=22, color=YELLOW)
        formula_math = MathTex(r"\sin C = \frac{n_2}{n_1}")
        formula_math.scale(1.3)

        formula_group = VGroup(formula_text, formula_math).arrange(DOWN, buff=0.3)
        formula_group.move_to(formula_bg.get_center())

        final_formula = VGroup(formula_bg, formula_group)

        self.play(FadeIn(final_formula, shift=DOWN * 0.5), run_time=1.2)
        self.wait(3)


if __name__ == "__main__":
    with tempconfig({"quality": "low_quality", "preview": True}):
        scene = TotalInternalReflectionProcess()
        scene.render()