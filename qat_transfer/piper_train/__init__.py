import os
import pathlib

import torch

# Checkpoints in this project were originally saved on a Linux/WSL2 machine,
# so hparams that embed a dataset Path get pickled as PosixPath. PosixPath
# can't be instantiated on Windows (pathlib.Path.__new__ dispatches by OS),
# so unpickling raises NotImplementedError even with weights_only=False.
# Aliasing it to WindowsPath makes old checkpoints loadable cross-platform;
# torch's weights_only unpickler resolves the pickled "pathlib.PosixPath"
# global via getattr(pathlib, "PosixPath") too, so the safe-globals allowlist
# below ends up (correctly) allowlisting WindowsPath once aliased.
if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath

torch.serialization.add_safe_globals([pathlib.PosixPath])
