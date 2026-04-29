from pathlib import Path
import argparse

import numpy as np
import sapien
from sapien.utils import Viewer
from PIL import Image


ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "objects" / "woodcube"
WOODCUBE_OBJ_PATH = ASSET_DIR / "woodcube_3DGS_LOD0.obj"

# Bounds measured from woodcube_3DGS_LOD0.obj. The visual mesh is offset so the
# actor origin sits at the center of its collision box.
WOODCUBE_CENTER = np.array([-0.003797842674234902, -0.003517248422993826, 0.000984349872937253])
WOODCUBE_HALF_SIZE = np.array([0.03052506364116126, 0.008655129620889047, 0.015265799836947623])


class WoodcubeSapienEnv:
    def __init__(self):
        self.scene = sapien.Scene()
        self.scene.set_timestep(1 / 100.0)
        self.woodcube = None
        self._load_scene()

    def _load_scene(self):
        if not WOODCUBE_OBJ_PATH.exists():
            raise FileNotFoundError(f"Woodcube OBJ not found: {WOODCUBE_OBJ_PATH}")

        self.scene.add_ground(altitude=0, render_half_size=[0.5, 0.5])
        self.scene.set_ambient_light([0.55, 0.55, 0.55])
        self.scene.add_directional_light([0.2, 0.4, -1.0], [1.0, 0.95, 0.85], shadow=True)
        self.scene.add_point_light([-0.25, -0.35, 0.45], [1.6, 1.4, 1.1])

        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=WOODCUBE_HALF_SIZE.tolist(), density=650)
        builder.add_visual_from_file(
            filename=str(WOODCUBE_OBJ_PATH),
            pose=sapien.Pose((-WOODCUBE_CENTER).tolist()),
            material=sapien.render.RenderMaterial(base_color=[0.52, 0.36, 0.2, 1.0]),
        )
        self.woodcube = builder.build(name="woodcube_3DGS_LOD0")
        self.woodcube.set_pose(sapien.Pose([0, 0, WOODCUBE_HALF_SIZE[2] + 0.002]))

    def step(self):
        self.scene.step()
        self.scene.update_render()

    def save_screenshot(self, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)

        camera_mount = self.scene.create_actor_builder().build_kinematic(name="preview_camera_mount")
        camera_mount.set_pose(_look_at_pose(np.array([-0.16, -0.18, 0.12]), np.array([0, 0, 0.016])))
        camera = self.scene.add_mounted_camera(
            "woodcube_preview_camera",
            camera_mount,
            sapien.Pose(),
            1280,
            900,
            0.75,
            0.01,
            10,
        )

        self.scene.update_render()
        camera.take_picture()
        color = camera.get_picture("Color")
        image = (color[..., :3].clip(0, 1) * 255).astype(np.uint8)
        Image.fromarray(image).save(output_path)
        return output_path

    def show(self):
        viewer = Viewer()
        viewer.set_scene(self.scene)
        viewer.set_camera_xyz(x=-0.18, y=-0.18, z=0.12)
        viewer.set_camera_rpy(r=0, p=-0.45, y=0.75)
        viewer.window.set_camera_parameters(near=0.01, far=10, fovy=0.75)

        while not viewer.closed:
            self.step()
            viewer.render()


def main():
    parser = argparse.ArgumentParser(description="Show woodcube_3DGS_LOD0.obj in a standalone SAPIEN scene.")
    parser.add_argument("--screenshot", type=Path, default=None, help="Save a preview PNG instead of opening the viewer.")
    args = parser.parse_args()

    env = WoodcubeSapienEnv()
    print(f"Loaded woodcube mesh: {WOODCUBE_OBJ_PATH}")
    print(f"Half size: {WOODCUBE_HALF_SIZE.tolist()}")
    if args.screenshot is not None:
        print(f"Saved screenshot: {env.save_screenshot(args.screenshot)}")
    else:
        env.show()


def _look_at_pose(camera_position: np.ndarray, target: np.ndarray):
    forward = target - camera_position
    forward = forward / np.linalg.norm(forward)
    up_hint = np.array([0.0, 0.0, 1.0])
    left = np.cross(up_hint, forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)
    rotation = np.column_stack([forward, left, up])
    quat_xyzw = _matrix_to_quat_xyzw(rotation)
    quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
    return sapien.Pose(camera_position.tolist(), quat_wxyz)


def _matrix_to_quat_xyzw(matrix: np.ndarray):
    trace = np.trace(matrix)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        return np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / s,
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[1, 0] - matrix[0, 1]) / s,
                0.25 * s,
            ]
        )

    axis = int(np.argmax(np.diag(matrix)))
    if axis == 0:
        s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
        return np.array(
            [
                0.25 * s,
                (matrix[0, 1] + matrix[1, 0]) / s,
                (matrix[0, 2] + matrix[2, 0]) / s,
                (matrix[2, 1] - matrix[1, 2]) / s,
            ]
        )
    if axis == 1:
        s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
        return np.array(
            [
                (matrix[0, 1] + matrix[1, 0]) / s,
                0.25 * s,
                (matrix[1, 2] + matrix[2, 1]) / s,
                (matrix[0, 2] - matrix[2, 0]) / s,
            ]
        )

    s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
    return np.array(
        [
            (matrix[0, 2] + matrix[2, 0]) / s,
            (matrix[1, 2] + matrix[2, 1]) / s,
            0.25 * s,
            (matrix[1, 0] - matrix[0, 1]) / s,
        ]
    )


if __name__ == "__main__":
    main()
