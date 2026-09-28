"""Viewer-local right-drag orbit; owns input polling and camera placement."""

import math


class RobotOrbitCamera:
    def __init__(self, env, gymapi):
        self.env = env
        self.selected = 0
        self.yaw = math.atan2(-3.0, -2.2)
        self.pitch = 0.36
        self.distance = 4.0
        self.dragging = False
        self.last_mouse = None
        env.gym.subscribe_viewer_mouse_event(
            env.viewer, gymapi.MOUSE_RIGHT_BUTTON, "robot_orbit_drag"
        )
        for button, action in ((gymapi.MOUSE_SCROLL_UP, "orbit_near"),
                               (gymapi.MOUSE_SCROLL_DOWN, "orbit_far")):
            env.gym.subscribe_viewer_mouse_event(env.viewer, button, action)

    def render(self, sync_frame_time=True):
        env = self.env
        if env.viewer is None:
            return
        if env.gym.query_viewer_has_closed(env.viewer):
            raise SystemExit
        # Consume events exactly once here, instead of BaseTask.render consuming
        # and discarding our mouse actions before the camera can see them.
        mouse = env.gym.get_viewer_mouse_position(env.viewer)
        current = (mouse.x, mouse.y)
        for event in env.gym.query_viewer_action_events(env.viewer):
            if event.action == "QUIT" and event.value > 0:
                raise SystemExit
            if event.action == "toggle_viewer_sync" and event.value > 0:
                env.enable_viewer_sync = not env.enable_viewer_sync
            if event.action == "robot_orbit_drag":
                self.dragging = event.value > 0
                self.last_mouse = current
            if event.action == "orbit_near" and event.value > 0:
                self.distance = max(0.6, self.distance / 1.15)
            if event.action == "orbit_far" and event.value > 0:
                self.distance = min(30.0, self.distance * 1.15)
        if self.dragging and self.last_mouse is not None:
            self.yaw -= (current[0] - self.last_mouse[0]) * 2.0 * math.pi
            self.pitch = max(-1.4, min(1.4, self.pitch +
                (current[1] - self.last_mouse[1]) * math.pi))
        self.last_mouse = current
        target = env.root_states[self.selected, :3].detach().cpu().tolist()
        target[2] += 0.15
        horizontal = self.distance * math.cos(self.pitch)
        position = [target[0] + horizontal * math.cos(self.yaw),
                    target[1] + horizontal * math.sin(self.yaw),
                    target[2] + self.distance * math.sin(self.pitch)]
        env.set_camera(position, target)
        if env.device != "cpu":
            env.gym.fetch_results(env.sim, True)
        if env.enable_viewer_sync:
            env.gym.step_graphics(env.sim)
            env.gym.draw_viewer(env.viewer, env.sim, True)
            if sync_frame_time:
                env.gym.sync_frame_time(env.sim)
        else:
            env.gym.poll_viewer_events(env.viewer)
