import bpy
from blenderproc.python.camera import CameraUtility

from typing import Tuple

def get_baseline_and_fx_from_scene() -> Tuple[float, float]:
    cam_ob = bpy.context.scene.camera
    if cam_ob is None:
        raise RuntimeError("No active camera in scene.")
    cam = cam_ob.data

    baseline = float(cam.stereo.interocular_distance)
    if baseline <= 0:
        raise RuntimeError("Stereo baseline is not set (cam.stereo.interocular_distance == 0).")

    K = CameraUtility.get_intrinsics_as_K_matrix()
    fx = float(K[0, 0])
    return baseline, fx
