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
        for i in range(len(self.layer_data)):
            if i < len(self.lines):
                self.lines[i].set_stroke(opacity=self.layer_data[i]["opacity"] * alpha)


class RayArrow(Polygon):
    def __init__(self, color, **kwargs):
        super().__init__(
            np.array([-0.20, -0.15, 0]),
            np.array([-0.20, 0.15, 0]),
            np.array([0.20, 0, 0]),
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

        p1 = np.array([-0.18, -0.12, 0])
        p2 = np.array([-0.18, 0.12, 0])
        p3 = np.array([0.18, 0.00, 0])

        c = math.cos(angle)
        s = math.sin(angle)

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
        self.core = Dot(radius=0.10, color=WHITE)
        self.halo1 = Dot(radius=0.20, color=color, fill_opacity=0.7)
        self.halo2 = Dot(radius=0.45, color=color, fill_opacity=0.25)
        self.halo3 = Dot(radius=0.75, color=color, fill_opacity=0.08)
        self.add(self.halo3, self.halo2, self.halo1, self.core)


class BrewsterTransmission(Scene):
    def construct(self):
        n1 = 1.00
        n2 = 1.50
        theta_B = math.atan(n2 / n1)

        col_inc = "#FF2200"
        col_ref = "#FF8800"
        col_tra = "#00AAFF"

        air_rect = Rectangle(width=config.frame_width, height=config.frame_height / 2, stroke_width=0)
        air_rect.set_fill(color="#112233", opacity=0.15)
        air_rect.next_to(ORIGIN, UP, buff=0)

        glass_rect = Rectangle(width=config.frame_width, height=config.frame_height / 2, stroke_width=0)
        glass_rect.set_fill(color="#002244", opacity=0.6)
        glass_rect.next_to(ORIGIN, DOWN, buff=0)

        boundary = Line(LEFT * 8, RIGHT * 8, color="#88AACC", stroke_width=3)
        normal = DashedLine(DOWN * 5, UP * 5, color="#FFFFFF", dash_length=0.12, stroke_opacity=0.5)

        air_text = Text("空气 (n₁=1.0)", font_size=28, weight=BOLD, color="#FFFFFF", font="Microsoft YaHei")
        air_text.to_corner(UR).shift(LEFT * 0.5 + DOWN * 0.5)

        glass_text = Text("玻璃 (n₂=1.5)", font_size=28, weight=BOLD, color="#AADDFF", font="Microsoft YaHei")
        glass_text.to_corner(DR).shift(LEFT * 0.5 + UP * 0.5)

        tracker = ValueTracker(30 * DEGREES)

        inc_ray = GlowLine(ORIGIN, ORIGIN, base_color=col_inc)
        ref_ray = GlowLine(ORIGIN, ORIGIN, base_color=col_ref)
        tra_ray = GlowLine(ORIGIN, ORIGIN, base_color=col_tra)

        arr_inc = RayArrow(color=col_inc)
        arr_ref = RayArrow(color=col_ref)
        arr_tra = RayArrow(color=col_tra)

        src = LightSource(color=col_inc)

        p_label = Text("p 偏振光", font_size=22, color=WHITE, font="Microsoft YaHei")

        info_panel_bg = RoundedRectangle(corner_radius=0.15, width=4.5, height=1.8, color="#445566",
                                         fill_color="#000000", fill_opacity=0.8)
        info_panel_bg.to_corner(DL).shift(RIGHT * 0.5 + UP * 0.5)

        val_inc = Text("入射角: 00.0°", font_size=20, color=col_inc, font="Microsoft YaHei")
        val_rp = Text("反射率 (Rp): 0.0%", font_size=20, color=col_ref, font="Microsoft YaHei")

        text_group = VGroup(val_inc, val_rp).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        text_group.move_to(info_panel_bg.get_center())
        info_panel = VGroup(info_panel_bg, text_group)

        val_inc_pos = val_inc.get_center()
        val_rp_pos = val_rp.get_center()

        def update_inc(mob):
            th = tracker.get_value()
            pt = np.array([-8 * math.sin(th), 8 * math.cos(th), 0])
            mob.put_start_and_end_on(pt, ORIGIN)

        def update_inc_arr(mob):
            th = tracker.get_value()
            pt = np.array([-8 * math.sin(th), 8 * math.cos(th), 0])
            mob.update_pose(pt, ORIGIN, 0.45, 1.0)
            src.move_to(pt)
            p_label.next_to(src, UP + LEFT, buff=0.1)
            val_inc.become(
                Text(f"入射角: {th * 180 / math.pi:.1f}°", font_size=20, color=col_inc, font="Microsoft YaHei").move_to(
                    val_inc_pos, aligned_edge=LEFT))

        def update_ref(mob):
            th1 = tracker.get_value()
            th2 = math.asin((n1 / n2) * math.sin(th1))
            num = n2 * math.cos(th1) - n1 * math.cos(th2)
            den = n2 * math.cos(th1) + n1 * math.cos(th2)
            rp = num / den
            Rp_intensity = rp ** 2

            val_rp.become(Text(f"反射强度 (Rp): {Rp_intensity * 100:.2f}%", font_size=20, color=col_ref,
                               font="Microsoft YaHei").move_to(val_rp_pos, aligned_edge=LEFT))

            boost = min(1.0, abs(rp) * 8.0)
            if boost < 0.001:
                safe_pt = np.array([0.001, 0.001, 0])
                mob.put_start_and_end_on(ORIGIN, safe_pt)
                mob.set_glow_opacity(0.0)
            else:
                pt = np.array([8 * math.sin(th1), 8 * math.cos(th1), 0])
                mob.put_start_and_end_on(ORIGIN, pt)
                mob.set_glow_opacity(boost)

        def update_ref_arr(mob):
            th1 = tracker.get_value()
            th2 = math.asin((n1 / n2) * math.sin(th1))
            rp = (n2 * math.cos(th1) - n1 * math.cos(th2)) / (n2 * math.cos(th1) + n1 * math.cos(th2))
            boost = min(1.0, abs(rp) * 8.0)
            if boost < 0.001:
                safe_pt = np.array([0.001, 0.001, 0])
                mob.update_pose(ORIGIN, safe_pt, 0.55, 0.0)
            else:
                pt = np.array([8 * math.sin(th1), 8 * math.cos(th1), 0])
                mob.update_pose(ORIGIN, pt, 0.55, boost)

        def update_tra(mob):
            th1 = tracker.get_value()
            th2 = math.asin((n1 / n2) * math.sin(th1))
            pt = np.array([8 * math.sin(th2), -8 * math.cos(th2), 0])
            mob.put_start_and_end_on(ORIGIN, pt)
            mob.set_glow_opacity(1.0)

        def update_tra_arr(mob):
            th1 = tracker.get_value()
            th2 = math.asin((n1 / n2) * math.sin(th1))
            pt = np.array([8 * math.sin(th2), -8 * math.cos(th2), 0])
            mob.update_pose(ORIGIN, pt, 0.55, 1.0)

        update_inc(inc_ray)
        update_inc_arr(arr_inc)
        update_ref(ref_ray)
        update_ref_arr(arr_ref)
        update_tra(tra_ray)
        update_tra_arr(arr_tra)

        inc_ray.add_updater(update_inc)
        arr_inc.add_updater(update_inc_arr)
        ref_ray.add_updater(update_ref)
        arr_ref.add_updater(update_ref_arr)
        tra_ray.add_updater(update_tra)
        arr_tra.add_updater(update_tra_arr)

        self.play(
            FadeIn(air_rect), FadeIn(glass_rect),
            Create(boundary), Create(normal),
            Write(air_text), Write(glass_text),
            FadeIn(info_panel),
            run_time=1.5
        )

        self.play(
            FadeIn(src), FadeIn(p_label),
            FadeIn(inc_ray), FadeIn(arr_inc),
            FadeIn(ref_ray), FadeIn(arr_ref),
            FadeIn(tra_ray), FadeIn(arr_tra),
            run_time=1.5
        )

        total_angle = 75 * DEGREES - 30 * DEGREES
        angle_p1 = theta_B - 30 * DEGREES
        angle_p2 = 75 * DEGREES - theta_B

        time_p1 = 6.0 * (angle_p1 / total_angle)
        time_p2 = 6.0 * (angle_p2 / total_angle)

        self.play(
            tracker.animate.set_value(theta_B),
            run_time=time_p1,
            rate_func=linear
        )

        formula_bg = RoundedRectangle(corner_radius=0.2, width=5.5, height=2.0, color=YELLOW, fill_color=BLACK,
                                      fill_opacity=0.85)
        formula_bg.to_corner(UL).shift(RIGHT * 0.5 + DOWN * 0.5)

        formula_text = Text("布儒斯特角公式 (全透射)", font_size=22, color=YELLOW, font="Microsoft YaHei")
        formula_math = MathTex(r"\tan \theta_B = \frac{n_2}{n_1}")
        formula_math.scale(1.3)

        formula_group = VGroup(formula_text, formula_math).arrange(DOWN, buff=0.3)
        formula_group.move_to(formula_bg.get_center())

        final_formula = VGroup(formula_bg, formula_group)

        self.play(FadeIn(final_formula, shift=DOWN * 0.5), run_time=1.5)

        self.play(
            tracker.animate.set_value(75 * DEGREES),
            run_time=time_p2,
            rate_func=linear
        )

        self.wait(2)


if __name__ == "__main__":
    with tempconfig({"quality": "low_quality", "preview": True}):
        scene = BrewsterTransmission()
        scene.render()