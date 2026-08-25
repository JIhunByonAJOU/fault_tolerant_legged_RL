import unittest

import numpy as np

from legged_gym.evaluation.teacher243_viewer_overlay import (
    ACTUAL_PATH_COLOR,
    NORMAL_COLOR,
    SEVERITY_STYLES,
    TARGET_PATH_COLOR,
    advance_reference,
    panel_rows,
    rigid_body_name_for_dof,
    severity_style,
    terrain_heights_at_xy,
)


class Teacher243FailureViewerTest(unittest.TestCase):
    def test_requested_color_legend_is_exact_and_ordered(self):
        self.assertEqual([item.degradation for item in SEVERITY_STYLES], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        self.assertEqual(SEVERITY_STYLES[0].rgb, NORMAL_COLOR)
        self.assertEqual(SEVERITY_STYLES[1].name, "20% degraded")
        self.assertEqual(SEVERITY_STYLES[2].name, "40% degraded")
        self.assertEqual(SEVERITY_STYLES[3].name, "60% degraded")
        self.assertEqual(SEVERITY_STYLES[4].name, "80% degraded")
        self.assertEqual(SEVERITY_STYLES[5].rgb, (0.0, 0.0, 0.0))
        self.assertEqual(TARGET_PATH_COLOR, (0.10, 0.90, 0.10))
        self.assertEqual(ACTUAL_PATH_COLOR, (0.10, 0.30, 1.00))

    def test_severity_is_nearest_configured_level(self):
        self.assertEqual(severity_style(0.0).degradation, 0.0)
        self.assertEqual(severity_style(0.39).degradation, 0.4)
        self.assertEqual(severity_style(1.0).degradation, 1.0)
        with self.assertRaises(ValueError):
            severity_style(1.1)

    def test_a1_joint_maps_to_driven_link(self):
        bodies = ("trunk", "FR_hip", "FR_thigh", "FR_calf", "FR_foot")
        self.assertEqual(rigid_body_name_for_dof("FR_calf_joint", bodies), "FR_calf")
        with self.assertRaises(ValueError):
            rigid_body_name_for_dof("RL_calf_joint", bodies)

    def test_reference_integration_uses_body_frame_command(self):
        xy, yaw = advance_reference(
            np.array([[0.0, 0.0], [1.0, 2.0]]),
            np.array([0.0, np.pi / 2.0]),
            np.array([[1.0, 0.0, 0.5], [1.0, 0.0, -0.5]]),
            0.2,
        )
        np.testing.assert_allclose(xy, [[0.2, 0.0], [1.0, 2.2]], atol=1e-8)
        np.testing.assert_allclose(yaw, [0.1, np.pi / 2.0 - 0.1], atol=1e-8)

    def test_measured_body_heading_rotates_body_velocity_into_world(self):
        xy, yaw = advance_reference(
            np.array([[0.0, 0.0], [0.0, 0.0]]),
            np.array([1.2, -0.8]),
            np.array([[1.0, 0.0, 0.9], [1.0, 0.0, -0.9]]),
            0.5,
            body_heading=np.array([0.0, np.pi / 2.0]),
        )
        np.testing.assert_allclose(xy, [[0.5, 0.0], [0.0, 0.5]], atol=1e-8)
        np.testing.assert_allclose(yaw, [0.0, np.pi / 2.0], atol=1e-8)

    def test_panel_rows_explain_joint_severity_and_remaining_output(self):
        names = ["FR_hip_joint", "FR_thigh_joint"]
        rows = panel_rows([-1, 1], [0.0, 0.8], names, [1.0, 0.22])
        self.assertEqual(rows[0]["joint"], "Normal")
        self.assertEqual(rows[0]["color"], "#FFFFFF")
        self.assertEqual(rows[1]["joint"], "FR_thigh_joint")
        self.assertAlmostEqual(rows[1]["remaining"], 0.2)
        self.assertAlmostEqual(rows[1]["effective_strength"], 0.22)
        self.assertEqual(rows[1]["color"], "#732E0D")

    def test_terrain_height_sampling_follows_world_xy_and_clips_edges(self):
        grid = np.array(
            [
                [0, 0, 0, 0],
                [0, 2, 2, 0],
                [0, 2, 4, 0],
                [0, 0, 0, 0],
            ]
        )
        sampled = terrain_heights_at_xy(
            np.array([[0.0, 0.0], [1.0, 1.0], [99.0, 99.0]]),
            grid,
            border_size=1.0,
            horizontal_scale=1.0,
            vertical_scale=0.1,
        )
        np.testing.assert_allclose(sampled, [0.2, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
