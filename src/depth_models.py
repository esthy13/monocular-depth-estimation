"""Pluggable depth model interface.

Add new models by subclassing DepthModel and registering them in build_depth_model().
Currently supported:
  - 'depth_anything_v2': Depth Anything V2 via HuggingFace transformers (relative depth).
  - 'dac':               Depth Any Camera via local clone + HuggingFace weights (metric depth).
  - 'unidac':            UniDAC via local clone + HuggingFace weights (metric depth).

To use DAC, first clone the repo into third_party/:
    git clone https://github.com/yuliangguo/depth_any_camera.git third_party/depth_any_camera

To use UniDAC, clone the pinned upstream repo into third_party/UniDAC. The Colab
notebook in notebooks/unidac_colab.ipynb performs this setup automatically.

Note: the dac/models/ops/ C++ extension (deformable attention) is NOT required because
the outdoor ResNet101 config uses attn_dec=false (BasePixelDecoder). The compiled module
is mocked automatically if absent.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class DepthModel(ABC):
    """Abstract interface for monocular depth estimation models."""

    # True if predict() returns metric depth in metres (value grows with distance).
    # False for inverse-depth / disparity-like output (value grows as objects near).
    # This controls colormap direction so that near is always rendered bright.
    is_metric: bool = False

    @abstractmethod
    def load(self) -> None:
        """Download weights (if needed) and load them into memory."""
        ...

    @abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth for a single BGR uint8 image (H, W, 3).

        Returns a float32 depth array (H, W).
        Values are relative (arbitrary scale) unless the model provides metric depth.
        """
        ...


# ---------------------------------------------------------------------------
# Depth Anything V2
# ---------------------------------------------------------------------------

class DepthAnythingV2(DepthModel):
    """Depth Anything V2 loaded via the HuggingFace transformers library.

    Model page: https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf

    Args:
        encoder: one of 'small', 'base', 'large'.
        device: 'cpu', 'cuda', or 'mps'. Auto-detected if None.
    """

    # Depth Anything V2 predicts inverse depth (near = large value), not metric.
    is_metric = False

    _MODEL_IDS = {
        "small": "depth-anything/Depth-Anything-V2-Small-hf",
        "base":  "depth-anything/Depth-Anything-V2-Base-hf",
        "large": "depth-anything/Depth-Anything-V2-Large-hf",
    }

    def __init__(self, encoder: str = "small", device: str | None = None) -> None:
        if encoder not in self._MODEL_IDS:
            raise ValueError(
                f"encoder must be one of {list(self._MODEL_IDS)}; got {encoder!r}"
            )
        self.encoder = encoder
        self.model_id = self._MODEL_IDS[encoder]
        self._model = None
        self._processor = None

        if device is None:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

    def load(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        print(f"Loading {self.model_id} on {self.device} ...")
        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
        self._model.to(self.device).eval()
        print("Model loaded.")

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict depth for a BGR uint8 image. Returns float32 (H, W)."""
        if self._model is None or self._processor is None:
            raise RuntimeError("Call load() before predict().")

        import torch
        import torch.nn.functional as F
        import cv2
        from PIL import Image

        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        inputs = self._processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            raw_depth = outputs.predicted_depth  # shape: (1, H', W')

        # Resize back to original image resolution
        depth = F.interpolate(
            raw_depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy().astype(np.float32)

        return depth


# ---------------------------------------------------------------------------
# Depth Any Camera (DAC)
# ---------------------------------------------------------------------------

def _ensure_dac_on_path() -> None:
    """Add third_party/depth_any_camera to sys.path if present and not already there."""
    import sys
    from pathlib import Path

    # Search relative to this file's location (src/ → project root → third_party/)
    candidates = [
        Path(__file__).parent.parent / "third_party" / "depth_any_camera",
        Path.cwd() / "third_party" / "depth_any_camera",
    ]
    for p in candidates:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
            return


def _mock_dac_ops_if_missing() -> None:
    """Mock the DAC compiled C++ deformable-attention ops when they are not built.

    The outdoor ResNet101 config has attn_dec=false: MSDeformAttn is imported at
    module level inside idisc.py → defattn_decoder.py → ops/modules → ops/functions
    → MultiScaleDeformableAttention (compiled C++ extension). Since it is never
    *instantiated*, mocking the full import chain is safe and lets IDisc load on
    any platform (Mac MPS, CPU-only Linux) without compiling the CUDA extension.
    """
    import sys
    try:
        import MultiScaleDeformableAttention  # noqa: F401 — compiled, all good
        return
    except (ImportError, ModuleNotFoundError):
        pass

    from unittest.mock import MagicMock

    # Remove any partially-loaded entries left by a previous failed import attempt
    stale = [k for k in sys.modules
             if k.startswith("dac.models.ops") or k == "MultiScaleDeformableAttention"]
    for k in stale:
        del sys.modules[k]

    # Mock the compiled C++ module and the entire ops subpackage
    mock = MagicMock()
    for mod in (
        "MultiScaleDeformableAttention",
        "dac.models.ops",
        "dac.models.ops.modules",
        "dac.models.ops.modules.ms_deform_attn",
        "dac.models.ops.functions",
        "dac.models.ops.functions.ms_deform_attn_func",
    ):
        sys.modules[mod] = mock


class DepthAnyCamera(DepthModel):
    """Depth Any Camera (DAC) — zero-shot metric depth for fisheye and perspective cameras.

    Uses an ERP (Equirectangular Projection) pipeline:
      fisheye/perspective image  →  cam_to_erp_patch_fast  →  IDisc model
      →  ERP depth  →  back-project via fisheye undistortion + cv2.remap  →  depth in camera space

    Weights download: https://huggingface.co/yuliangguo/depth-any-camera
    Code (clone to third_party/): https://github.com/yuliangguo/depth_any_camera

    Args:
        variant:    one of the keys in VARIANTS (default 'dac-outdoor-resnet101').
        cam_params: DAC-format camera intrinsics dict from intrinsics_to_dac_cam_params().
        crop_wfov:  horizontal FoV of the ERP patch in degrees (180 for fisheye, ~100 for ZED).
        device:     'cpu', 'cuda', or 'mps'. Auto-detected if None.
    """

    # DAC predicts metric depth in metres (near = small value).
    is_metric = True

    HF_REPO = "yuliangguo/depth-any-camera"
    VARIANTS: dict[str, tuple[str, str]] = {
        "dac-outdoor-resnet101": ("dac_resnet101_outdoor.json", "dac_resnet101_outdoor.pt"),
        "dac-outdoor-swinl":     ("dac_swinl_outdoor.json",     "dac_swinl_outdoor.pt"),
        "dac-indoor-resnet101":  ("dac_resnet101_indoor.json",  "dac_resnet101_indoor.pt"),
        "dac-indoor-swinl":      ("dac_swinl_indoor.json",      "dac_swinl_indoor.pt"),
    }

    def __init__(
        self,
        variant: str = "dac-outdoor-resnet101",
        cam_params: dict | None = None,
        crop_wfov: float = 180.0,
        fwd_sz: tuple[int, int] | None = None,
        device: str | None = None,
    ) -> None:
        if variant not in self.VARIANTS:
            raise ValueError(
                f"variant must be one of {list(self.VARIANTS)}; got {variant!r}"
            )
        self.variant = variant
        self._cam_params = cam_params
        self._crop_wfov = crop_wfov
        # Model input patch size (H, W). Square for a symmetric fisheye crop so the
        # ERP patch spans equal latitude/longitude; defaults are set in predict().
        self._fwd_sz = fwd_sz
        self._model = None
        self._config = None
        self._grid_cache: dict = {}

        if device is None:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # DAC uses operations not fully supported on MPS; fall back to CPU
                self.device = "cpu"
            else:
                self.device = "cpu"
        else:
            self.device = device

    def load(self) -> None:
        import json
        import torch
        from huggingface_hub import hf_hub_download

        _ensure_dac_on_path()
        _mock_dac_ops_if_missing()

        config_file, weights_file = self.VARIANTS[self.variant]

        print(f"Downloading DAC config  : {config_file}")
        config_path = hf_hub_download(repo_id=self.HF_REPO, filename=config_file)
        print(f"Downloading DAC weights : {weights_file}  (first run ~700 MB)")
        weights_path = hf_hub_download(repo_id=self.HF_REPO, filename=weights_file)

        with open(config_path) as f:
            config = json.load(f)
        self._config = config

        model_cls_name = config["model_name"]
        if model_cls_name == "IDisc":
            from dac.models.idisc import IDisc as ModelCls
        elif model_cls_name == "IDiscERP":
            from dac.models.idisc_erp import IDiscERP as ModelCls
        elif model_cls_name == "CNNDepth":
            from dac.models.cnn_depth import CNNDepth as ModelCls
        else:
            raise ValueError(f"Unsupported DAC model_name in config: {model_cls_name!r}")

        print(f"Building {model_cls_name} ({self.variant}) on {self.device} …")
        model = ModelCls.build(config)

        # DAC checkpoints contain optimizer state that requires weights_only=False.
        # PyTorch ≥ 2.6 changed the default to True, so we patch torch.load temporarily.
        _orig_load = torch.load
        torch.load = lambda f, **kw: _orig_load(f, weights_only=False, **kw)
        try:
            model.load_pretrained(weights_path)
        finally:
            torch.load = _orig_load

        model.to(self.device).eval()
        self._model = model
        print("DAC model loaded.")

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict metric depth (metres) for a BGR uint8 image. Returns float32 (H, W)."""
        if self._model is None or self._config is None:
            raise RuntimeError("Call load() before predict().")
        if self._cam_params is None:
            raise RuntimeError(
                "cam_params must be set before predict(). "
                "Pass intrinsics via build_depth_model(..., cam_params=...) or set "
                "model._cam_params directly."
            )

        import torch
        import cv2
        import torchvision.transforms.functional as TF

        from dac.utils.erp_geometry import (
            cam_to_erp_patch_fast,
            erp_patch_to_cam_fast,
        )
        from dac.dataloders.dataset import resize_for_input

        config = self._config
        model_name = config["model_name"]
        cam_params = dict(self._cam_params)  # copy so we can add a 'dataset' key
        cam_params.setdefault("dataset", "generic")
        is_fisheye = cam_params.get("camera_model") == "OPENCV_FISHEYE"
        h, w = image.shape[:2]

        cano_sz = config["cano_sz"]           # ERP size the model was trained on, e.g. [1400, 1400]

        # Model input patch size (H, W). A square patch keeps equal lat/lon span,
        # which is the correct geometry for a symmetric circular fisheye.
        fwd_sz = self._fwd_sz or (700, 700)

        # ------------------------------------------------------------------
        # Step 1 — project the camera image into an ERP patch (DAC native)
        # ------------------------------------------------------------------
        rgb01 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        depth_in = np.zeros((h, w, 1), dtype=np.float32)
        mask_in  = np.ones( (h, w, 1), dtype=np.float32)

        theta = 0.0
        phi   = np.array(0).astype(np.float32)
        roll  = np.array(0).astype(np.float32)

        # crop the ERP sphere to crop_wfov degrees of longitude, matched in latitude
        crop_w = int(cano_sz[0] * self._crop_wfov / 180)
        crop_h = int(crop_w * fwd_sz[0] / fwd_sz[1])

        erp_img, erp_depth, _, erp_mask, latitude, longitude = cam_to_erp_patch_fast(
            rgb01, depth_in, mask_in, theta, phi,
            crop_h, crop_w, cano_sz[0], cano_sz[0] * 2,
            cam_params, roll, scale_fac=None,
        )
        lat_range  = torch.tensor([float(np.min(latitude)),  float(np.max(latitude))])
        long_range = torch.tensor([float(np.min(longitude)), float(np.max(longitude))])

        # ------------------------------------------------------------------
        # Step 2 — resize the ERP patch to the model input size
        # resize_for_input returns the pred_scale_factor that rescales metric depth
        # ------------------------------------------------------------------
        erp_img_u8 = (erp_img * 255.0).astype(np.uint8)
        erp_img_r, erp_depth_r, _pad, pred_scale_factor, attn_mask = resize_for_input(
            erp_img_u8, erp_depth, fwd_sz, None,
            [erp_img.shape[0], erp_img.shape[1]], 1.0,
            padding_rgb=[0, 0, 0], mask=erp_mask,
        )

        # ------------------------------------------------------------------
        # Step 3 — model inference (metric depth in ERP space)
        # ------------------------------------------------------------------
        norm_stats = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        img_tensor = TF.normalize(TF.to_tensor(erp_img_r), **norm_stats).unsqueeze(0)
        attn_tensor = TF.to_tensor((attn_mask > 0).astype(np.float32)).unsqueeze(0)

        with torch.no_grad():
            if model_name == "IDiscERP":
                preds, _, _ = self._model(
                    img_tensor.to(self.device),
                    lat_range.unsqueeze(0).to(self.device),
                    long_range.unsqueeze(0).to(self.device),
                )
            else:
                preds, _, _ = self._model(img_tensor.to(self.device))
        preds = preds.detach().cpu() * pred_scale_factor   # apply metric rescale
        if preds.dim() == 3:
            preds = preds.unsqueeze(1)

        # ------------------------------------------------------------------
        # Step 4 — back-project ERP depth into the original camera image (DAC native)
        # ------------------------------------------------------------------
        # erp_h scaled by pred_scale_factor mirrors the demo (depth scale ≡ resize ratio)
        erp_h_bp = cano_sz[0] * pred_scale_factor
        erp_w_bp = erp_h_bp * 2

        if is_fisheye:
            # OPENCV_FISHEYE has no closed-form ERP inversion → provide a ray lookup grid.
            # 'scannetpp' triggers the grid path inside erp_patch_to_cam_fast.
            grid = self._fisheye_ray_grid(h, w)
            bp_cam_params = {"dataset": "scannetpp"}
        else:
            grid = None
            bp_cam_params = cam_params  # PINHOLE: uses fx/fy/cx/cy inverse directly

        img_out, depth_out, valid_out, active_out = erp_patch_to_cam_fast(
            img_tensor[0], preds[0], attn_tensor[0],
            0.0, 0.0, out_h=h, out_w=w,
            erp_h=erp_h_bp, erp_w=erp_w_bp,
            cam_params=bp_cam_params, fisheye_grid2ray=grid,
        )

        depth_cam = depth_out[0, 0].numpy().astype(np.float32)
        # Pixels outside the back-projected valid region → 0 (masked later as invalid)
        active = active_out[0].numpy() if active_out.dim() == 3 else active_out.numpy()
        depth_cam[active < 0.5] = 0.0
        return depth_cam

    def _fisheye_ray_grid(self, h: int, w: int) -> np.ndarray:
        """Build the (H, W, 4) OPENCV_FISHEYE pixel→ray lookup table DAC expects.

        Channels [0:3] are the unit ray direction (X, Y, Z) for each pixel; channel 3
        flags invalid pixels. This is the inverse-distortion grid DAC normally
        precomputes per camera; we derive it from the K matrix and k1–k4 coefficients
        using the same arctan-polynomial model as DAC's create_fisheye_grid scripts.
        """
        key = (h, w)
        if key in self._grid_cache:
            return self._grid_cache[key]

        cp = self._cam_params
        fx, fy = cp["fl_x"], cp["fl_y"]
        cx, cy = cp["cx"], cp["cy"]
        k1, k2, k3, k4 = cp["k1"], cp["k2"], cp["k3"], cp["k4"]

        # Monotonic lookup mapping undistorted radius ro → distorted radius theta_d
        ro = np.linspace(0.0, 15.0, 500_000, dtype=np.float64)
        theta = np.arctan(ro)
        theta_d = theta * (1 + k1 * theta**2 + k2 * theta**4
                           + k3 * theta**6 + k4 * theta**8)

        u, v = np.meshgrid(np.arange(w), np.arange(h))
        x = (u.ravel().astype(np.float64) + 0.5 - cx) / fx
        y = (v.ravel().astype(np.float64) + 0.5 - cy) / fy
        dist = np.sqrt(x * x + y * y)   # distorted radius in normalized coords

        # Invert: find ro whose theta_d matches the observed distorted radius
        idx = np.clip(np.searchsorted(theta_d, dist), 0, len(ro) - 1)
        scale = ro[idx] / (theta_d[idx] + 1e-9)
        xu, yu = x * scale, y * scale

        z = 1.0 / np.sqrt(1.0 + xu * xu + yu * yu)
        X, Y, Z = xu * z, yu * z, z
        isnan = (~np.isfinite(X)) | (~np.isfinite(Y)) | (~np.isfinite(Z))

        grid = np.stack(
            [X, Y, Z, isnan.astype(np.float64)], axis=-1
        ).reshape(h, w, 4).astype(np.float32)
        self._grid_cache[key] = grid
        return grid


# ---------------------------------------------------------------------------
# UniDAC
# ---------------------------------------------------------------------------

class UniDACDepth(DepthAnyCamera):
    """UniDAC universal metric depth with native camera-geometry handling.

    UniDAC uses the same camera-to-ERP and ERP-to-camera geometry as DAC, so this
    adapter inherits DAC's calibrated OpenCV-fisheye ray-grid implementation.
    The model itself is the newer DINOv3-based UniDAC checkpoint.

    Args:
        cam_params: DAC/UniDAC camera parameters from
            :func:`src.utils.intrinsics_to_dac_cam_params`.
        crop_wfov: horizontal FoV of the ERP patch in degrees.
        fwd_sz: optional model input size (H, W). Defaults to the UniDAC config.
        device: ``cuda`` or ``cpu``. CUDA is selected automatically when present.
        repo_dir: path to the cloned official UniDAC repository. Defaults to
            ``third_party/UniDAC`` relative to this project.
        checkpoint_path: optional local ``unidac.pt``. When omitted, the official
            checkpoint is downloaded from Hugging Face and cached.
    """

    is_metric = True
    HF_REPO = "girish1511/UniDAC"
    CHECKPOINT_FILE = "unidac.pt"
    CONFIG_FILE = "configs/test/dac_dinov3l+dpt_indoor_test_scannetpp.json"

    def __init__(
        self,
        cam_params: dict | None = None,
        crop_wfov: float = 180.0,
        fwd_sz: tuple[int, int] | None = None,
        device: str | None = None,
        repo_dir: str | None = None,
        checkpoint_path: str | None = None,
    ) -> None:
        # Do not call DepthAnyCamera.__init__: UniDAC has no DAC variant and uses
        # a different config/checkpoint format. We only inherit its ray-grid code.
        self._cam_params = cam_params
        self._crop_wfov = float(crop_wfov)
        self._fwd_sz = fwd_sz
        self._model = None
        self._config = None
        self._grid_cache: dict = {}
        self._repo_dir = repo_dir
        self._checkpoint_path = checkpoint_path

        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def _resolve_repo_dir(self):
        """Return the UniDAC clone path and make its package importable."""
        import sys
        from pathlib import Path

        candidates = []
        if self._repo_dir is not None:
            candidates.append(Path(self._repo_dir).expanduser())
        candidates.extend(
            [
                Path(__file__).parent.parent / "third_party" / "UniDAC",
                Path.cwd() / "third_party" / "UniDAC",
            ]
        )
        for candidate in candidates:
            candidate = candidate.resolve()
            if (candidate / "unidac").is_dir():
                if str(candidate) not in sys.path:
                    sys.path.insert(0, str(candidate))
                return candidate
        checked = "\n  - ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            "UniDAC repository not found. Run the Colab setup cell or clone "
            "https://github.com/girish1511/UniDAC into third_party/UniDAC.\n"
            f"Checked:\n  - {checked}"
        )

    def set_camera(self, cam_params: dict, crop_wfov: float) -> None:
        """Switch calibrated cameras while keeping the large model loaded once."""
        if cam_params != self._cam_params:
            self._grid_cache.clear()
        self._cam_params = cam_params
        self._crop_wfov = float(crop_wfov)

    def load(self) -> None:
        import json
        from pathlib import Path

        import torch
        from huggingface_hub import hf_hub_download

        repo_dir = self._resolve_repo_dir()
        config_path = repo_dir / self.CONFIG_FILE
        if not config_path.exists():
            raise FileNotFoundError(f"UniDAC config not found: {config_path}")

        if self._checkpoint_path is None:
            print(f"Downloading/caching UniDAC checkpoint: {self.CHECKPOINT_FILE}")
            checkpoint_path = Path(
                hf_hub_download(repo_id=self.HF_REPO, filename=self.CHECKPOINT_FILE)
            )
        else:
            checkpoint_path = Path(self._checkpoint_path).expanduser().resolve()
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"UniDAC checkpoint not found: {checkpoint_path}")

        with config_path.open() as file:
            config = json.load(file)

        # The released config points to the authors' local DINOv3 backbone file.
        # The complete UniDAC checkpoint replaces all parameters immediately after
        # construction, so initialize the backbone locally without that unavailable
        # pretraining file.
        pixel_encoder = config["model"]["pixel_encoder"]
        pixel_encoder["pretrained"] = None
        pixel_encoder.pop("weights", None)

        from unidac.models.unidac import UniDAC

        print(f"Building UniDAC on {self.device} ...")
        model = UniDAC.build(config)

        # PyTorch 2.6 changed torch.load's default. The official loader does not
        # specify weights_only, so make checkpoint loading stable across Colab images.
        original_torch_load = torch.load

        def load_checkpoint(file, *args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_torch_load(file, *args, **kwargs)

        torch.load = load_checkpoint
        try:
            model.load_pretrained(str(checkpoint_path))
        finally:
            torch.load = original_torch_load

        model.to(self.device).eval()
        self._config = config
        self._model = model
        print("UniDAC model loaded.")

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict metric Euclidean depth in metres for a BGR uint8 image."""
        if self._model is None or self._config is None:
            raise RuntimeError("Call load() before predict().")
        if self._cam_params is None:
            raise RuntimeError(
                "cam_params must be set before predict(); load the selected camera "
                "calibration with intrinsics_to_dac_cam_params()."
            )

        import cv2
        import torch
        import torchvision.transforms.functional as TF

        from unidac.dataloaders.dataset import resize_for_input
        from unidac.utils.erp_geometry import (
            cam_to_erp_patch_fast,
            erp_patch_to_cam_fast,
        )

        cam_params = dict(self._cam_params)
        cam_params.setdefault("dataset", "generic")
        is_fisheye = cam_params.get("camera_model") == "OPENCV_FISHEYE"
        height, width = image.shape[:2]
        canonical_size = self._config["data"]["cano_sz"]
        forward_size = self._fwd_sz or tuple(self._config["data"]["fwd_sz"])

        rgb01 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        dummy_depth = np.zeros((height, width, 1), dtype=np.float32)
        valid_input = np.ones((height, width, 1), dtype=np.float32)

        theta = 0.0
        phi = np.array(0, dtype=np.float32)
        roll = np.array(0, dtype=np.float32)
        crop_width = int(canonical_size[0] * self._crop_wfov / 180.0)
        crop_height = int(crop_width * forward_size[0] / forward_size[1])

        erp_image, erp_depth, _, erp_mask, latitude, longitude = cam_to_erp_patch_fast(
            rgb01,
            dummy_depth,
            valid_input,
            theta,
            phi,
            crop_height,
            crop_width,
            canonical_size[0],
            canonical_size[0] * 2,
            cam_params,
            roll,
            scale_fac=None,
        )
        latitude_range = torch.tensor(
            [float(np.min(latitude)), float(np.max(latitude))], dtype=torch.float32
        )
        longitude_range = torch.tensor(
            [float(np.min(longitude)), float(np.max(longitude))], dtype=torch.float32
        )

        (
            resized_image,
            _resized_depth,
            _padding,
            prediction_scale,
            attention_mask,
            latitude_grid,
            _longitude_grid,
        ) = resize_for_input(
            (erp_image * 255.0).astype(np.uint8),
            erp_depth,
            forward_size,
            None,
            [erp_image.shape[0], erp_image.shape[1]],
            1.0,
            padding_rgb=[0, 0, 0],
            mask=erp_mask,
            lat_grid=latitude,
            long_grid=longitude,
        )

        normalization = {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
        image_tensor = TF.normalize(
            TF.to_tensor(resized_image), **normalization
        ).unsqueeze(0).to(self.device)
        attention_tensor = TF.to_tensor(
            (attention_mask > 0).astype(np.float32)
        ).unsqueeze(0).to(self.device)
        latitude_tensor = torch.as_tensor(
            latitude_grid, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.inference_mode():
            prediction, _, _ = self._model(
                image_tensor,
                latitude_range.unsqueeze(0).to(self.device),
                longitude_range.unsqueeze(0).to(self.device),
                attn_mask=attention_tensor,
                lat_grid=latitude_tensor,
            )
            prediction = prediction * float(prediction_scale)
            if prediction.dim() == 3:
                prediction = prediction.unsqueeze(1)

            erp_height = canonical_size[0] * float(prediction_scale)
            erp_width = erp_height * 2.0
            if is_fisheye:
                ray_grid = self._fisheye_ray_grid(height, width)
                backproject_params = {"dataset": "scannetpp"}
            else:
                ray_grid = None
                backproject_params = cam_params

            _, camera_depth, valid_output, active_output = erp_patch_to_cam_fast(
                image_tensor[0],
                prediction[0],
                attention_tensor[0],
                0.0,
                0.0,
                out_h=height,
                out_w=width,
                erp_h=erp_height,
                erp_w=erp_width,
                cam_params=backproject_params,
                fisheye_grid2ray=ray_grid,
            )

        depth = camera_depth[0, 0].detach().cpu().numpy().astype(np.float32)
        valid = valid_output[0, 0].detach().cpu().numpy() >= 0.5
        active = active_output[0].detach().cpu().numpy() >= 0.5
        depth[~(valid & active & np.isfinite(depth) & (depth > 0))] = 0.0
        return depth


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_depth_model(name: str = "depth_anything_v2", **kwargs) -> DepthModel:
    """Instantiate a depth model by name.

    Args:
        name: model identifier string.
        **kwargs: forwarded to the model's __init__.

    Supported names:
        'depth_anything_v2'      — Depth Anything V2 (HuggingFace, relative depth)
        'dac'                    — Depth Any Camera, pass variant= to select checkpoint
        'unidac'                 — UniDAC universal metric depth
        'dac-outdoor-resnet101'  — DAC outdoor ResNet101 shorthand
        'dac-indoor-resnet101'   — DAC indoor  ResNet101 shorthand
    """
    # Allow passing the full DAC variant name as the model name
    if name in DepthAnyCamera.VARIANTS:
        kwargs.setdefault("variant", name)
        name = "dac"

    registry: dict[str, type[DepthModel]] = {
        "depth_anything_v2": DepthAnythingV2,
        "dac": DepthAnyCamera,
        "unidac": UniDACDepth,
    }
    if name not in registry:
        raise ValueError(
            f"Unknown depth model: {name!r}. Available: {list(registry)}"
        )
    return registry[name](**kwargs)
