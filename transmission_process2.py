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


class SplitBrewsterComparison(Scene):
    def construct(self):
        tracker = ValueTracker(20 * DEGREES)

        col_water = "#002244"
        col_air = "#112233"
        n_water = 1.33
        n_air = 1.00

        w_half = config.frame_width / 2
        h_half = config.frame_height / 2
        ray_len = 3.2

        def build_system(offset_x, n_top, n_bot, color_top, color_bot, label_top, label_bot):
            center = np.array([offset_x, 0, 0])

            rect_top = Rectangle(width=w_half, height=h_half, stroke_width=0, fill_color=color_top, fill_opacity=0.6)
            rect_top.move_to(center + UP * h_half / 2)

            rect_bot = Rectangle(width=w_half, height=h_half, stroke_width=0, fill_color=color_bot, fill_opacity=0.6)
            rect_bot.move_to(center + DOWN * h_half / 2)

            boundary = Line(center + LEFT * w_half / 2, center + RIGHT * w_half / 2, color="#88AACC", stroke_width=3)
            normal = DashedLine(center + DOWN * 3.5, center + UP * 3.5, color=WHITE, dash_length=0.12,
                                stroke_opacity=0.5)

            t_top = Text(label_top, font_size=20, color=WHITE, font="Microsoft YaHei").move_to(
                center + UP * (h_half - 0.4))
            t_bot = Text(label_bot, font_size=20, color=WHITE, font="Microsoft YaHei").move_to(
                center + DOWN * (h_half - 0.4))

            col_inc = "#FF2200"
            col_ref = "#FF8800"
            col_tra = "#00AAFF"

            inc_ray = GlowLine(ORIGIN, ORIGIN, base_color=col_inc)
            ref_ray = GlowLine(ORIGIN, ORIGIN, base_color=col_ref)
            tra_ray = GlowLine(ORIGIN, ORIGIN, base_color=col_tra)

            arr_inc = RayArrow(color=col_inc)
            arr_ref = RayArrow(color=col_ref)
            arr_tra = RayArrow(color=col_tra)

            src = LightSource(color=col_inc)

            arc_mob = VMobject()
            theta_label = Tex(r"$\theta_1$", font_size=28, color=WHITE)

            theta_B = math.atan(n_bot / n_top)

            info_panel_bg = RoundedRectangle(corner_radius=0.15, width=3.8, height=1.6, color="#445566",
                                             fill_color="#000000", fill_opacity=0.8)
            info_panel_bg.move_to(center + LEFT * 1.8 + DOWN * 2.5)

            val_inc = Text("入射角: 00.0°", font_size=16, color=col_inc, font="Microsoft YaHei")
            val_rp = Text("反射率 (Rp): 0.0%", font_size=16, color=col_ref, font="Microsoft YaHei")
            val_tb = Text(f"布儒斯特角: {theta_B * 180 / math.pi:.1f}°", font_size=16, color=YELLOW,
                          font="Microsoft YaHei")

            text_group = VGroup(val_inc, val_rp, val_tb).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
            text_group.move_to(info_panel_bg.get_center())
            info_panel = VGroup(info_panel_bg, text_group)

            val_inc_pos = val_inc.get_center()
            val_rp_pos = val_rp.get_center()

            def update_inc(mob):
                th = tracker.get_value()
                pt = center + np.array([-ray_len * math.sin(th), ray_len * math.cos(th), 0])
                mob.put_start_and_end_on(pt, center)

            def update_inc_arr(mob):
                th = tracker.get_value()
                pt = center + np.array([-ray_len * math.sin(th), ray_len * math.cos(th), 0])
                mob.update_pose(pt, center, 0.45, 1.0)
                src.move_to(pt)
                val_inc.become(Text(f"入射角: {th * 180 / math.pi:.1f}°", font_size=16, color=col_inc,
                                    font="Microsoft YaHei").move_to(val_inc_pos, aligned_edge=LEFT))

            def update_ref(mob):
                th = tracker.get_value()
                sin_t = (n_top / n_bot) * math.sin(th)
                if sin_t >= 0.9999:
                    pt = center + np.array([ray_len * math.sin(th), ray_len * math.cos(th), 0])
                    mob.put_start_and_end_on(center, pt)
                    mob.set_glow_opacity(1.0)
                    val_rp.become(
                        Text("反射率 (Rp): 全反射", font_size=16, color=col_ref, font="Microsoft YaHei").move_to(
                            val_rp_pos, aligned_edge=LEFT))
                else:
                    th_t = math.asin(sin_t)
                    rp = (n_bot * math.cos(th) - n_top * math.cos(th_t)) / (
                                n_bot * math.cos(th) + n_top * math.cos(th_t))
                    boost = min(1.0, abs(rp) * 8.0)
                    if boost < 0.001:
                        safe_pt = center + np.array([0.001, 0.001, 0])
                        mob.put_start_and_end_on(center, safe_pt)
                        mob.set_glow_opacity(0.0)
                    else:
                        pt = center + np.array([ray_len * math.sin(th), ray_len * math.cos(th), 0])
                        mob.put_start_and_end_on(center, pt)
                        mob.set_glow_opacity(boost)
                    val_rp.become(Text(f"反射率 (Rp): {(rp ** 2) * 100:.2f}%", font_size=16, color=col_ref,
                                       font="Microsoft YaHei").move_to(val_rp_pos, aligned_edge=LEFT))

            def update_ref_arr(mob):
                th = tracker.get_value()
                sin_t = (n_top / n_bot) * math.sin(th)
                if sin_t >= 0.9999:
                    pt = center + np.array([ray_len * math.sin(th), ray_len * math.cos(th), 0])
                    mob.update_pose(center, pt, 0.55, 1.0)
                else:
                    th_t = math.asin(sin_t)
                    rp = (n_bot * math.cos(th) - n_top * math.cos(th_t)) / (
                                n_bot * math.cos(th) + n_top * math.cos(th_t))
                    boost = min(1.0, abs(rp) * 8.0)
                    if boost < 0.001:
                        safe_pt = center + np.array([0.001, 0.001, 0])
                        mob.update_pose(center, safe_pt, 0.55, 0.0)
                    else:
                        pt = center + np.array([ray_len * math.sin(th), ray_len * math.cos(th), 0])
                        mob.update_pose(center, pt, 0.55, boost)

            def update_tra(mob):
                th = tracker.get_value()
                sin_t = (n_top / n_bot) * math.sin(th)
                if sin_t >= 0.9999:
                    safe_pt = center + np.array([0.001, 0.001, 0])
                    mob.put_start_and_end_on(center, safe_pt)
                    mob.set_glow_opacity(0.0)
                else:
                    th_t = math.asin(sin_t)
                    pt = center + np.array([ray_len * math.sin(th_t), -ray_len * math.cos(th_t), 0])
                    mob.put_start_and_end_on(center, pt)
                    mob.set_glow_opacity(1.0)

            def update_tra_arr(mob):
                th = tracker.get_value()
                sin_t = (n_top / n_bot) * math.sin(th)
                if sin_t >= 0.9999:
                    safe_pt = center + np.array([0.001, 0.001, 0])
                    mob.update_pose(center, safe_pt, 0.5, 0.0)
                else:
                    th_t = math.asin(sin_t)
                    pt = center + np.array([ray_len * math.sin(th_t), -ray_len * math.cos(th_t), 0])
                    mob.update_pose(center, pt, 0.55, 1.0)

            def update_arc(mob):
                th = tracker.get_value()
                new_arc = Arc(arc_center=center, radius=1.2, start_angle=math.pi / 2, angle=-th, color=WHITE,
                              stroke_width=2.5)
                new_arc.add_tip(tip_length=0.15, tip_width=0.15)
                mob.become(new_arc)

            def update_theta_label(mob):
                th = tracker.get_value()
                r = 1.6
                bisector = math.pi / 2 - th / 2
                pos = center + np.array([r * math.cos(bisector), r * math.sin(bisector), 0])
                mob.become(
                    Tex(f"$\\theta_1 = {th * 180 / math.pi:.1f}^\\circ$", font_size=28, color=WHITE).move_to(pos))

            update_inc(inc_ray)
            update_inc_arr(arr_inc)
            update_ref(ref_ray)
            update_ref_arr(arr_ref)
            update_tra(tra_ray)
            update_tra_arr(arr_tra)
            update_arc(arc_mob)
            update_theta_label(theta_label)

            def attach_updaters():
                inc_ray.add_updater(update_inc)
                arr_inc.add_updater(update_inc_arr)
                ref_ray.add_updater(update_ref)
                arr_ref.add_updater(update_ref_arr)
                tra_ray.add_updater(update_tra)
                arr_tra.add_updater(update_tra_arr)
                arc_mob.add_updater(update_arc)
                theta_label.add_updater(update_theta_label)

            static_grp = VGroup(rect_top, rect_bot, boundary, normal, t_top, t_bot, info_panel)
            dynamic_grp = VGroup(inc_ray, ref_ray, tra_ray, arr_inc, arr_ref, arr_tra, src, arc_mob, theta_label)

            return static_grp, dynamic_grp, attach_updaters

        left_offset = -w_half / 2
        right_offset = w_half / 2

        left_static, left_dynamic, attach_left = build_system(
            left_offset, n_air, n_water, col_air, col_water,
            "空气 (n=1.00)", "水 (n=1.33)"
        )

        right_static, right_dynamic, attach_right = build_system(
            right_offset, n_water, n_air, col_water, col_air,
            "水 (n=1.33)", "空气 (n=1.00)"
        )

        divider = Line(UP * config.frame_height / 2, DOWN * config.frame_height / 2, color=WHITE, stroke_width=4)

        self.play(
            FadeIn(left_static),
            FadeIn(right_static),
            Create(divider),
            run_time=1.5
        )

        self.play(
            FadeIn(left_dynamic),
            FadeIn(right_dynamic),
            run_time=1.5
        )

        attach_left()
        attach_right()

        self.play(
            tracker.animate.set_value(65 * DEGREES),
            run_time=9.0,
            rate_func=linear
        )

        self.wait(1)

        summary_bg = RoundedRectangle(corner_radius=0.2, width=9.5, height=1.2, color=YELLOW, fill_color=BLACK,
                                      fill_opacity=0.85)
        summary_bg.to_edge(DOWN, buff=0.5)
        summary_text = Text("与全反射不同，全透射(布儒斯特角)在任何介质界面均可发生", font_size=24, color=YELLOW,
                            font="Microsoft YaHei")
        summary_text.move_to(summary_bg.get_center())

        final_summary = VGroup(summary_bg, summary_text)

        self.play(FadeIn(final_summary, shift=UP * 0.5), run_time=1.2)
        self.wait(3)


if __name__ == "__main__":
    with tempconfig({"quality": "high_quality", "preview": True}):
        scene = SplitBrewsterComparison()
        scene.render()