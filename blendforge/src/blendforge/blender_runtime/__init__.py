import sys
if sys.version_info < (3, 10):
    raise RuntimeError("blendforge.blender_runtime требует Python >= 3.10 (запускать внутри Blender/BlenderProc).")
