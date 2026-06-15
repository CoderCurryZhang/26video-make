from manim import *
import math
import numpy as np

from MacroBrewsterTransmissionScene import (
    EnergyArrow,
    RealisticEnergyBeam,
    RealisticGlassMedium,
    RealisticLightSource,
)


class MacroTotalInternalReflectionScene(Scene):
    def construct(self):
        tracker = ValueTracker(25 * DEGREES)

        n_glass = 1.50
        n_air = 1.00
        critical_angle = math.asin(n_air / n_glass)
        target_angle = 90 * DEGREES

        col_glass = "#00254D"
        col_air = "#050505"
        col_beams = "#66FFFF"
        ray_len = 5.0

        grid = NumberPlane(
            x_range=[-10, 10, 1],
            y_range=[-10, 10, 1],
            background_line_style={
                "stroke_color": "#2A3A4A",
                "stroke_width": 1.5,
                "stroke_opacity": 0.25,
            },
        )
        self.add(grid)

        glass_rect = RealisticGlassMedium(
            width=config.frame_width,
            height=config.frame_height / 2,
            base_color=col_glass,
        )
        glass_rect.next_to(ORIGIN, DOWN, buff=0)

        air_rect = Rectangle(
            width=config.frame_width,
            height=config.frame_height / 2,
            stroke_width=0,
        )
        air_rect.set_fill(color=col_air, opacity=0.8)
        air_rect.next_to(ORIGIN, UP, buff=0)

        boundary = Line(LEFT * 8, RIGHT * 8, color="#88AACC", stroke_width=3)
        normal = DashedLine(
            DOWN * 5,
            UP * 5,
            color=WHITE,
            dash_length=0.12,
            stroke_opacity=0.5,
        )

        glass_text = Text(
            "玻璃 (n=1.50)",
            font_size=28,
            weight=BOLD,
            color=WHITE,
            font="Microsoft YaHei",
        )
        glass_text.to_corner(DL).shift(RIGHT * 0.5 + UP * 1.35)

        air_text = Text(
            "空气 (n=1.00)",
            font_size=28,
            weight=BOLD,
            color=WHITE,
            font="Microsoft YaHei",
        )
        air_text.to_corner(UL).shift(RIGHT * 0.5 + DOWN * 0.5)

        incident_ray = RealisticEnergyBeam(intensity=1.0, color=col_beams)
        reflected_ray = RealisticEnergyBeam(intensity=0.2, color=col_beams)
        refracted_ray = RealisticEnergyBeam(intensity=1.0, color=col_beams)

        incident_arrow = EnergyArrow(color=col_beams)
        reflected_arrow = EnergyArrow(color=col_beams)
        refracted_arrow = EnergyArrow(color=col_beams)
        source = RealisticLightSource(color=col_beams)

        info_panel_bg = RoundedRectangle(
            corner_radius=0.15,
            width=4.5,
            height=2.2,
            color="#445566",
            fill_color="#000000",
            fill_opacity=0.8,
        )
        info_panel_bg.to_corner(UR).shift(LEFT * 0.5 + DOWN * 0.5)

        val_inc = Text(
            "入射角: 00.0°",
            font_size=20,
            color=WHITE,
            font="Microsoft YaHei",
        )
        val_ref = Text(
            "反射率: 0.0%",
            font_size=20,
            color=col_beams,
            font="Microsoft YaHei",
        )
        val_cri = Text(
            f"临界角: {critical_angle * 180 / math.pi:.1f}°",
            font_size=20,
            color=YELLOW,
            font="Microsoft YaHei",
        )

        text_group = VGroup(val_inc, val_ref, val_cri).arrange(
            DOWN,
            aligned_edge=LEFT,
            buff=0.25,
        )
        text_group.move_to(info_panel_bg.get_center())
        info_panel = VGroup(info_panel_bg, text_group)

        val_inc_ref = val_inc.get_left()
        val_ref_ref = val_ref.get_left()

        arc_mob = VMobject()
        theta_label = MathTex(r"\theta_1", font_size=36, color=WHITE)

        def fresnel_p(theta):
            sin_t = (n_glass / n_air) * math.sin(theta)
            if sin_t > 1.0 + 1e-9:
                return 1.0, 0.0, None

            sin_t = min(1.0, sin_t)
            theta_t = math.asin(sin_t)
            if math.isclose(sin_t, 1.0, abs_tol=1e-9):
                return 1.0, 0.0, theta_t

            term1 = n_air * math.cos(theta)
            term2 = n_glass * math.cos(theta_t)
            reflectance = ((term1 - term2) / (term1 + term2)) ** 2
            return reflectance, 1.0 - reflectance, theta_t

        def update_incident(mob):
            theta = tracker.get_value()
            start = incident_start(theta)
            mob.put_start_and_end_on(start, ORIGIN)

        def incident_start(theta):
            return np.array(
                [-ray_len * math.sin(theta), -ray_len * math.cos(theta), 0]
            )

        def update_incident_arrow(mob):
            theta = tracker.get_value()
            start = incident_start(theta)
            mob.update_pose(start, ORIGIN, 0.45, 1.0, 1.0)

        def update_source(mob):
            theta = tracker.get_value()
            start = incident_start(theta)
            mob.move_to(start)

        def update_reflected(mob):
            theta = tracker.get_value()
            reflectance, _, _ = fresnel_p(theta)
            end = np.array(
                [ray_len * math.sin(theta), -ray_len * math.cos(theta), 0]
            )
            mob.put_start_and_end_on(ORIGIN, end)
            mob.set_beam_intensity(min(1.0, reflectance * 8.0))

        def update_reflected_arrow(mob):
            theta = tracker.get_value()
            reflectance, _, _ = fresnel_p(theta)
            end = np.array(
                [ray_len * math.sin(theta), -ray_len * math.cos(theta), 0]
            )
            visibility = min(1.0, reflectance * 8.0)
            mob.update_pose(ORIGIN, end, 0.55, visibility, visibility)

        def update_refracted(mob):
            theta = tracker.get_value()
            _, transmittance, theta_t = fresnel_p(theta)
            if theta_t is None:
                mob.put_start_and_end_on(ORIGIN, ORIGIN + RIGHT * 0.001)
                mob.set_beam_intensity(0.0)
                return

            end = np.array(
                [ray_len * math.sin(theta_t), ray_len * math.cos(theta_t), 0]
            )
            mob.put_start_and_end_on(ORIGIN, end)
            visibility = (
                0.4
                if math.isclose(theta, critical_angle, abs_tol=1e-9)
                else transmittance
            )
            mob.set_beam_intensity(visibility)

        def update_refracted_arrow(mob):
            theta = tracker.get_value()
            _, transmittance, theta_t = fresnel_p(theta)
            if theta_t is None:
                mob.update_pose(
                    ORIGIN,
                    ORIGIN + RIGHT * 0.001,
                    0.55,
                    0.0,
                    0.0,
                )
                return

            end = np.array(
                [ray_len * math.sin(theta_t), ray_len * math.cos(theta_t), 0]
            )
            visibility = (
                0.4
                if math.isclose(theta, critical_angle, abs_tol=1e-9)
                else transmittance
            )
            mob.update_pose(
                ORIGIN,
                end,
                0.55,
                visibility,
                visibility,
            )

        def update_arc(mob):
            theta = tracker.get_value()
            new_arc = Arc(
                radius=1.8,
                start_angle=-math.pi / 2,
                angle=-theta,
                color=WHITE,
                stroke_width=3.5,
            )
            new_arc.add_tip(tip_length=0.25, tip_width=0.25)
            mob.become(new_arc)

        def update_theta_label(mob):
            theta = tracker.get_value()
            radius = 2.4
            bisector = -math.pi / 2 - theta / 2
            position = np.array(
                [
                    radius * math.cos(bisector),
                    radius * math.sin(bisector),
                    0,
                ]
            )
            mob.become(
                MathTex(
                    f"\\theta_1 = {theta * 180 / math.pi:.1f}^\\circ",
                    font_size=32,
                    color=WHITE,
                ).move_to(position)
            )

        def update_val_inc(mob):
            theta = tracker.get_value()
            new_text = Text(
                f"入射角: {theta * 180 / math.pi:.1f}°",
                font_size=20,
                color=WHITE,
                font="Microsoft YaHei",
            )
            new_text.move_to(val_inc_ref, aligned_edge=LEFT)
            mob.become(new_text)

        def update_val_ref(mob):
            theta = tracker.get_value()
            reflectance, _, theta_t = fresnel_p(theta)
            if theta_t is None:
                label = "反射率: 100% (全反射)"
                color = YELLOW
            elif math.isclose(theta, critical_angle, abs_tol=1e-9):
                label = "反射率: 100% (临界状态)"
                color = YELLOW
            else:
                label = f"反射率: {reflectance * 100:.1f}%"
                color = col_beams

            new_text = Text(
                label,
                font_size=20,
                color=color,
                font="Microsoft YaHei",
            )
            new_text.move_to(val_ref_ref, aligned_edge=LEFT)
            mob.become(new_text)

        def incident_flow(mob, dt):
            mob.advance_flow(dt, global_speed=2.2)

        def reflected_flow(mob, dt):
            mob.advance_flow(dt, global_speed=2.2)

        def refracted_flow(mob, dt):
            mob.advance_flow(dt, global_speed=2.2)

        update_incident(incident_ray)
        update_incident_arrow(incident_arrow)
        update_source(source)
        update_reflected(reflected_ray)
        update_reflected_arrow(reflected_arrow)
        update_refracted(refracted_ray)
        update_refracted_arrow(refracted_arrow)
        update_arc(arc_mob)
        update_theta_label(theta_label)
        update_val_inc(val_inc)
        update_val_ref(val_ref)

        self.play(
            FadeIn(glass_rect),
            FadeIn(air_rect),
            Create(boundary),
            Create(normal),
            Write(glass_text),
            Write(air_text),
            FadeIn(info_panel),
            run_time=1.5,
        )

        self.play(
            FadeIn(source),
            FadeIn(incident_ray),
            FadeIn(incident_arrow),
            FadeIn(reflected_ray),
            FadeIn(reflected_arrow),
            FadeIn(refracted_ray),
            FadeIn(refracted_arrow),
            Create(arc_mob),
            FadeIn(theta_label),
            run_time=1.5,
        )
        self.bring_to_front(info_panel)

        incident_ray.add_updater(update_incident)
        incident_arrow.add_updater(update_incident_arrow)
        source.add_updater(update_source)
        reflected_ray.add_updater(update_reflected)
        reflected_arrow.add_updater(update_reflected_arrow)
        refracted_ray.add_updater(update_refracted)
        refracted_arrow.add_updater(update_refracted_arrow)
        arc_mob.add_updater(update_arc)
        theta_label.add_updater(update_theta_label)
        val_inc.add_updater(update_val_inc)
        val_ref.add_updater(update_val_ref)

        incident_ray.add_updater(incident_flow)
        reflected_ray.add_updater(reflected_flow)
        refracted_ray.add_updater(refracted_flow)

        self.play(
            tracker.animate.set_value(critical_angle),
            run_time=4.0,
            rate_func=rate_functions.ease_in_out_sine,
        )

        self.wait(2.0)

        self.play(
            tracker.animate.set_value(target_angle),
            run_time=5.0,
            rate_func=rate_functions.ease_in_out_sine,
        )

        self.wait(1)

        summary_bg = RoundedRectangle(
            corner_radius=0.2,
            width=9.5,
            height=1.2,
            color=YELLOW,
            fill_color=BLACK,
            fill_opacity=0.85,
        )
        summary_bg.move_to(UP * 0.85)
        summary_text = Text(
            "玻璃射向空气时，入射角超过临界角，折射光消失并发生全反射",
            font_size=26,
            color=YELLOW,
            font="Microsoft YaHei",
        )
        summary_text.scale_to_fit_width(summary_bg.width - 0.7)
        summary_text.move_to(summary_bg.get_center())
        final_summary = VGroup(summary_bg, summary_text)

        self.play(
            FadeOut(info_panel),
            FadeIn(final_summary, shift=UP * 0.5),
            run_time=1.2,
        )
        self.wait(4)

        incident_ray.remove_updater(incident_flow)
        reflected_ray.remove_updater(reflected_flow)
        refracted_ray.remove_updater(refracted_flow)


if __name__ == "__main__":
    with tempconfig({"quality": "high_quality", "preview": True}):
        scene = MacroTotalInternalReflectionScene()
        scene.render()
