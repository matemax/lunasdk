"""
Patch env for load binary library
"""
import os
import platform
import sys


def _patchLoadLib():
    """
    Add a directory with binary library to env.
    """
    if not os.getenv("FSDK_ROOT"):
        return
    if platform.system() == "Windows":
        # path to .dll
        pathToBinary = os.path.join(os.getenv("FSDK_ROOT"), "bin", "vs2015", "x64")
    else:
        # path to .so
        pathToBinary = os.path.join(os.getenv("FSDK_ROOT"), "lib", "gcc4", "x64")
    if platform.system() == "Windows":
        if sys.version.startswith("3.7") or sys.version.startswith("3.6"):
            os.environ["PATH"] = f"{os.getenv('PATH')};{pathToBinary}"
        else:
            # 3.8 or higher
            os.add_dll_directory(pathToBinary)
            # sdk 5.5.0 fix
            os.environ["PATH"] = f"{os.getenv('PATH')};{pathToBinary}"
    else:
        os.environ["LD_LIBRARY_PATH"] = f"{os.getenv('LD_LIBRARY_PATH')}:{pathToBinary}"


_patchLoadLib()
