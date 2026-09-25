"""
Optional metric monocular depth backend based on ZoeDepth (transformers).

**Model**     : ZoeDepth (``Intel/zoedepth-nyu-kitti`` hub checkpoint)
**Backend**   : ``ZoeDepthForDepthEstimation`` + ``AutoProcessor``
**Output**    : metric camera-axis depth ``Z`` in meters (NOT Euclidean range)
**CPU**       : fully supported (``device="cpu"``); no CUDA requirement
**Deps**      : ``torch``, ``transformers``, ``Pillow``, ``numpy``
**Download**  : one-time hub download of the checkpoint (~ a few hundred MB)

This module is deliberately lazily-imported: ``import`` here does NOT load
torch/transformers. They are imported only when ``load()`` / ``estimate()``
runs, so the core estimator stays lightweight.

Limitations
-----------
* Metric Z-depth is affine-per-pixel correct but assumes training data
  conditions; a depth-validated sensor is required for production accuracy.
* It is camera-axis depth, not Euclidean range. Convert with camera intrinsics
  (the estimator does this during ROI extraction when intrinsics are present).
* Useful range is roughly 0.5-25 m depending on the scene.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..depth import DepthBackendError, DepthFrame, DepthSource

_ZOE_MODEL_ID = "Intel/zoedepth-nyu-kitti"


class ZoeDepthBackend(DepthSource):
    """Metric monocular depth via the ZoeDepth transformer (lazy loaded)."""

    def __init__(self, model_id: str = _ZOE_MODEL_ID, device: Optional[str] = None) -> None:
        self.model_id = model_id
        self.device = device or "cpu"
        self._model = None
        self._processor = None
        self._name = f"ZoeDepth({model_id})"

    @property
    def name(self) -> str:
        return self._name

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the model (downloads weights on first call). Raises on failure."""
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoProcessor, ZoeDepthForDepthEstimation  # noqa: F401
        except ImportError as exc:
            raise DepthBackendError(
                "ZoeDepth backend requires 'torch' and 'transformers' "
                "(pip install transformers) before use"
            ) from exc

        try:
            from transformers import AutoProcessor, ZoeDepthForDepthEstimation

            self._model = ZoeDepthForDepthEstimation.from_pretrained(self.model_id)
            self._processor = AutoProcessor.from_pretrained(self.model_id)
        except Exception as exc:
            raise DepthBackendError(
                f"failed to download/load ZoeDepth model {self.model_id!r}: {exc}"
            ) from exc

        try:
            import torch

            self._model.to(self.device)
            self._model.eval()
        except Exception as exc:  # pragma: no cover - device placement
            raise DepthBackendError(f"failed to place ZoeDepth model on {self.device}: {exc}") from exc

    def estimate(self, image: Any) -> DepthFrame:
        self.load()

        import torch

        from transformers import AutoProcessor, ZoeDepthForDepthEstimation  # noqa: F401

        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise DepthBackendError("image must be an HxWx3 array")

        inputs = self._processor(images=arr, return_tensors="pt")
        with torch.no_grad():
            prediction = self._model(**inputs)
        depth = prediction.predicted_depth.squeeze().cpu().float().numpy()

        height, width = arr.shape[:2]
        if depth.shape != (height, width):
            depth_t = torch.from_numpy(depth[None, None])
            depth_t = torch.nn.functional.interpolate(
                depth_t, size=(height, width), mode="bilinear", align_corners=False
            )
            depth = depth_t.squeeze().numpy()

        valid = np.isfinite(depth) & (depth > 0.0)
        depth = depth.astype(np.float64)
        depth[~valid] = np.nan
        return DepthFrame(depth_m=depth, valid_mask=valid, source=self._name)