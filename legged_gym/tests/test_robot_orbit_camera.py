import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from legged_gym.evaluation.robot_orbit_camera import RobotOrbitCamera


class OrbitInputTest(unittest.TestCase):
    def test_drag_release_follow_and_respawn(self):
        gym = Mock()
        gym.query_viewer_has_closed.return_value = False
        gym.query_viewer_action_events.return_value = []
        gym.get_viewer_mouse_position.return_value = SimpleNamespace(x=0.5, y=0.5)
        root = Mock()
        root.__getitem__ = Mock(return_value=root)
        root.detach.return_value = root
        root.cpu.return_value = root
        root.tolist.side_effect = lambda: list(self.xyz)
        self.xyz = [0., 0., 0.3]
        env = SimpleNamespace(gym=gym, viewer=1, sim=1, device='cpu',
                              root_states=root, enable_viewer_sync=True,
                              set_camera=Mock())
        api = SimpleNamespace(MOUSE_RIGHT_BUTTON=1, MOUSE_SCROLL_UP=2, MOUSE_SCROLL_DOWN=3)
        camera = RobotOrbitCamera(env, api)
        gym.subscribe_viewer_mouse_event.assert_any_call(1, 2, 'orbit_near')
        gym.subscribe_viewer_mouse_event.assert_any_call(1, 3, 'orbit_far')
        gym.subscribe_viewer_keyboard_event.assert_not_called()
        camera.render(False)
        before = np.array(env.set_camera.call_args.args[0])
        gym.query_viewer_action_events.return_value = [
            SimpleNamespace(action='robot_orbit_drag', value=1)]
        camera.render(False)
        gym.query_viewer_action_events.return_value = []
        gym.get_viewer_mouse_position.return_value = SimpleNamespace(x=0.7, y=0.6)
        camera.render(False)
        position, target = map(np.array, env.set_camera.call_args.args)
        self.assertGreater(np.linalg.norm(position - before), 0.5)
        self.assertAlmostEqual(np.linalg.norm(position - target), 4.0)
        offset = position - target
        gym.query_viewer_action_events.return_value = [
            SimpleNamespace(action='robot_orbit_drag', value=0)]
        camera.render(False)
        gym.query_viewer_action_events.return_value = []
        gym.get_viewer_mouse_position.return_value = SimpleNamespace(x=0.2, y=0.2)
        for xyz in ([1., 2., 0.4], [20., -10., 0.3]):
            self.xyz = xyz
            camera.render(False)
            position, target = map(np.array, env.set_camera.call_args.args)
            np.testing.assert_allclose(position - target, offset)
            np.testing.assert_allclose(target, np.array(xyz) + [0, 0, 0.15])
        angles = (camera.yaw, camera.pitch)
        for action, expected in [('orbit_near', 4.0 / 1.15), ('orbit_far', 4.0)]:
            gym.query_viewer_action_events.return_value = [
                SimpleNamespace(action=action, value=1)]
            camera.render(False)
            position, target = map(np.array, env.set_camera.call_args.args)
            self.assertAlmostEqual(np.linalg.norm(position - target), expected)
            self.assertEqual((camera.yaw, camera.pitch), angles)


if __name__ == '__main__':
    unittest.main()
