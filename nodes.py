from __future__ import annotations

import gc
import json
import os
import requests
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import comfy.utils


PIPELINE_TYPE = "IDEOGRAM4_PIPELINE"
CORE_REPO_ENV_VAR = "IDEOGRAM4_REPO"
MIN_RESOLUTION = 256
MAX_RESOLUTION = 2048
RESOLUTION_MULTIPLE = 16
MAX_ASPECT_RATIO = 6.0

WEIGHT_REPOS = {
  "4.0 NF4": "ideogram-ai/ideogram-4-nf4",
  "4.0 FP8": "ideogram-ai/ideogram-4-fp8",
}

DEFAULT_MODEL_WEIGHTS = "4.0 NF4"
DEFAULT_TORCH_DTYPE = torch.bfloat16
DEFAULT_DEVICE = "cuda"

CUSTOM_SAMPLER_PRESET = "custom"
MAGIC_PROMPT_PROVIDER_IDEOGRAM = "ideogram"
MAGIC_PROMPT_PROVIDER_OPENROUTER = "openrouter"
MAGIC_PROMPT_PROVIDERS = [
  MAGIC_PROMPT_PROVIDER_IDEOGRAM,
  MAGIC_PROMPT_PROVIDER_OPENROUTER,
]
IDEOGRAM_MAGIC_PROMPT_CORE_KEY = "ideogram-4-v1"
DEFAULT_OPENROUTER_MODEL = ""
OPENROUTER_TIMEOUT_SECONDS = 120.0
CORE_PRESET_PREFIX = f"V{4}_"
CORE_PRESET_TO_LABEL = {
  f"{CORE_PRESET_PREFIX}QUALITY_48": "4.0 Quality 48",
  f"{CORE_PRESET_PREFIX}DEFAULT_20": "4.0 Default 20",
  f"{CORE_PRESET_PREFIX}TURBO_12": "4.0 Turbo 12",
}
CORE_PRESET_LABEL_TO_KEY = {label: key for key, label in CORE_PRESET_TO_LABEL.items()}
DEFAULT_SAMPLER_PRESET_LABEL = CORE_PRESET_TO_LABEL[
  f"{CORE_PRESET_PREFIX}DEFAULT_20"
]
FALLBACK_SAMPLER_PRESETS = [
  CUSTOM_SAMPLER_PRESET,
  *CORE_PRESET_TO_LABEL.values(),
]


def _progress(total: int = 1, node_id: str | None = None) -> comfy.utils.ProgressBar:
  pbar = comfy.utils.ProgressBar(total, node_id=node_id)
  pbar.update_absolute(0)
  return pbar


def _send_progress_text(node_id: str | None, text: str) -> None:
  if not node_id:
    return
  try:
    from server import PromptServer
  except Exception:
    return
  if PromptServer.instance is not None:
    PromptServer.instance.send_progress_text(text, node_id)


def _status_text(*parts: str) -> str:
  return " | ".join(part.replace("\n", " | ") for part in parts if part)


def _http_error_details(exc: requests.RequestException) -> tuple[str | int, str]:
  response = exc.response
  status = response.status_code if response is not None else "unknown"
  body = ""
  if response is not None:
    body = response.text.strip().replace("\n", " ")
  if len(body) > 500:
    body = body[:497] + "..."
  return status, body


def _openrouter_error_message(
  exc: requests.RequestException,
  magic_prompt_provider: str,
  openrouter_model: str,
) -> str:
  status, body = _http_error_details(exc)
  details = [
    "OpenRouter request failed.",
    f"Magic Prompt provider: {magic_prompt_provider}",
    f"OpenRouter model: {openrouter_model}",
    f"HTTP status: {status}",
  ]
  if body:
    details.append(f"Response: {body}")
  details.append(
    "Check OPENROUTER_API_KEY, key budget/limits, and whether the key has access "
    "to the selected model."
  )
  return " ".join(details)


def _ideogram_api_error_message(
  exc: requests.RequestException,
  magic_prompt_provider: str,
) -> str:
  status, body = _http_error_details(exc)
  details = [
    "Ideogram API Magic Prompt request failed.",
    f"Magic Prompt provider: {magic_prompt_provider}",
    f"HTTP status: {status}",
  ]
  if body:
    details.append(f"Response: {body}")
  details.append(
    "Check IDEOGRAM_API_KEY and whether the key has access to the Ideogram "
    "Magic Prompt API."
  )
  return " ".join(details)


def _is_openrouter_magic_prompt(magic_prompt_provider: str) -> bool:
  return magic_prompt_provider == MAGIC_PROMPT_PROVIDER_OPENROUTER


KNOWN_CONFIG_KEYS = ("IDEOGRAM_API_KEY", "OPENROUTER_API_KEY", "HF_TOKEN")
_CONFIG_PATH = Path(__file__).resolve().parent / "ideogram_config.json"


def _load_config() -> dict[str, str]:
  # Re-read each call so keys saved from the UI/file apply without a restart.
  try:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as handle:
      data = json.load(handle)
  except (FileNotFoundError, ValueError, OSError):
    return {}
  if not isinstance(data, dict):
    return {}
  return {str(key): str(value) for key, value in data.items()}


MAX_CONFIG_VALUE_LEN = 8192


def _save_config(updates: dict[str, Any]) -> dict[str, bool]:
  config = _load_config()
  for key, value in updates.items():
    if key not in KNOWN_CONFIG_KEYS:
      continue
    value = str(value).strip()
    if len(value) > MAX_CONFIG_VALUE_LEN:
      continue
    if value:
      config[key] = value
    else:
      config.pop(key, None)
  _write_config_atomic(config)
  return _config_status()


def _write_config_atomic(config: dict[str, str]) -> None:
  # Create with 0600 from the start, then atomically replace, so a secret is never
  # briefly world-readable and a crash can't leave a truncated file.
  data = json.dumps(config, indent=2)
  tmp_path = _CONFIG_PATH.with_name(_CONFIG_PATH.name + f".tmp{os.getpid()}")
  fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
  try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
      handle.write(data)
  except BaseException:
    tmp_path.unlink(missing_ok=True)
    raise
  os.replace(tmp_path, _CONFIG_PATH)


def _config_status() -> dict[str, bool]:
  config = _load_config()
  return {
    key: bool(config.get(key) or os.environ.get(key))
    for key in KNOWN_CONFIG_KEYS
  }


# Whether HF_TOKEN was provided externally (env / `hf auth login` shell) at
# startup. If so we never touch it. Captured once at import.
_STARTUP_HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()


def _apply_hf_token() -> None:
  # Make a UI/file-configured HF token usable by huggingface_hub without
  # `hf auth login`. In-process only: never call huggingface_hub.login() (that
  # writes into the global HF cache). Respect an externally provided token; else
  # mirror the config file exactly so clearing/changing the UI token takes effect
  # and a previously injected token is never left stale.
  if _STARTUP_HF_TOKEN:
    return
  token = _load_config().get("HF_TOKEN", "").strip()
  if token:
    os.environ["HF_TOKEN"] = token
  else:
    os.environ.pop("HF_TOKEN", None)


def _api_key_for_magic_prompt(magic_prompt_provider: str) -> str:
  if _is_openrouter_magic_prompt(magic_prompt_provider):
    env_var = "OPENROUTER_API_KEY"
  elif magic_prompt_provider == MAGIC_PROMPT_PROVIDER_IDEOGRAM:
    env_var = "IDEOGRAM_API_KEY"
  else:
    raise ValueError(f"Unsupported magic_prompt_provider: {magic_prompt_provider}")
  # Config file (set in the UI or hand-edited) wins; env var is the fallback.
  api_key = (_load_config().get(env_var) or os.environ.get(env_var, "")).strip()
  if not api_key:
    raise RuntimeError(
      f"{env_var} is not set for magic_prompt_provider={magic_prompt_provider}. "
      "Set it in ComfyUI Settings > Ideogram 4.0, add it to "
      f"{_CONFIG_PATH.name} in the node folder, or export {env_var} before "
      "starting ComfyUI. Settings/file changes apply on the next run."
    )
  return api_key


def _resolve_path(path: str) -> Path:
  return Path(path).expanduser().resolve()


def _normalize_dimension(value: int, name: str) -> int:
  value = int(value)
  if value < MIN_RESOLUTION or value > MAX_RESOLUTION:
    raise ValueError(
      f"{name} must be between {MIN_RESOLUTION} and {MAX_RESOLUTION}, got {value}"
    )
  if value % RESOLUTION_MULTIPLE != 0:
    raise ValueError(f"{name} must be divisible by {RESOLUTION_MULTIPLE}, got {value}")
  return value


def _normalize_dimensions(width: int, height: int) -> tuple[int, int]:
  width = _normalize_dimension(width, "width")
  height = _normalize_dimension(height, "height")
  aspect_ratio = max(width, height) / min(width, height)
  if aspect_ratio > MAX_ASPECT_RATIO:
    raise ValueError(f"aspect ratio must be at most {MAX_ASPECT_RATIO:g}:1, got {width}:{height}")
  return (width, height)


def _add_core_repo_to_path(core_repo_path: str = "") -> None:
  repo_path = core_repo_path.strip() or os.environ.get(CORE_REPO_ENV_VAR, "")
  if not repo_path:
    return

  resolved = _resolve_path(repo_path)
  if not resolved.exists():
    raise FileNotFoundError(f"Ideogram 4.0 repo path does not exist: {resolved}")

  candidates = [resolved]
  src_path = resolved / "src"
  if src_path.exists():
    candidates.insert(0, src_path)

  for candidate in candidates:
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
      sys.path.insert(0, candidate_str)


def _load_pipeline_classes():
  _add_core_repo_to_path()
  try:
    from ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig
  except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
      "Could not import ideogram4. Install the core repo into ComfyUI's Python "
      f"environment or set {CORE_REPO_ENV_VAR} before starting ComfyUI."
    ) from exc
  return Ideogram4Pipeline, Ideogram4PipelineConfig


def _load_sampler_presets(core_repo_path: str = ""):
  _add_core_repo_to_path(core_repo_path)
  try:
    from ideogram4 import PRESETS
  except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
      "Could not import Ideogram 4.0 sampler presets. Install the core repo, set "
      f"{CORE_REPO_ENV_VAR}, or install the wrapper inside ComfyUI's Python environment."
    ) from exc
  return PRESETS


def _load_magic_prompt_exports(core_repo_path: str = ""):
  _add_core_repo_to_path(core_repo_path)
  try:
    from ideogram4 import DEFAULT_MAGIC_PROMPT, MAGIC_PROMPTS, aspect_ratio_from_size
  except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
      "Could not import Ideogram 4.0 magic prompt support. Install the core repo or "
      f"set {CORE_REPO_ENV_VAR} before starting ComfyUI."
    ) from exc
  return DEFAULT_MAGIC_PROMPT, MAGIC_PROMPTS, aspect_ratio_from_size


def _sampler_preset_choices() -> list[str]:
  try:
    presets = _load_sampler_presets()
  except ModuleNotFoundError:
    return FALLBACK_SAMPLER_PRESETS
  return [
    CUSTOM_SAMPLER_PRESET,
    *[_sampler_label_from_core_key(key) for key in presets.keys()],
  ]


def _default_sampler_preset() -> str:
  choices = _sampler_preset_choices()
  if DEFAULT_SAMPLER_PRESET_LABEL in choices:
    return DEFAULT_SAMPLER_PRESET_LABEL
  return choices[0]


def _sampler_label_from_core_key(key: str) -> str:
  return CORE_PRESET_TO_LABEL.get(key, key)


def _sampler_core_key_from_label(label: str) -> str:
  return CORE_PRESET_LABEL_TO_KEY.get(label, label)


def _load_custom_magic_prompt_helpers():
  _add_core_repo_to_path()
  try:
    from ideogram4.magic_prompt import (
      build_messages,
      openrouter_chat,
      strip_aspect_ratio_and_bboxes,
    )
  except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
      "Could not import Ideogram 4.0 magic prompt helpers. Install the core repo or "
      f"set {CORE_REPO_ENV_VAR} before starting ComfyUI."
    ) from exc
  return build_messages, openrouter_chat, strip_aspect_ratio_and_bboxes


def _load_caption_verifier_class():
  _add_core_repo_to_path()
  try:
    from ideogram4.caption_verifier import CaptionVerifier
  except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
      "Could not import Ideogram 4.0 caption verifier. Install the core repo or "
      f"set {CORE_REPO_ENV_VAR} before starting ComfyUI."
    ) from exc
  return CaptionVerifier


def _verify_caption(caption: str) -> list[str]:
  CaptionVerifier = _load_caption_verifier_class()
  return CaptionVerifier().verify_raw(caption)


def _raise_magic_prompt_issues(caption: str, verify_json: bool) -> None:
  if not verify_json:
    return
  issues = _verify_caption(caption)
  if issues:
    raise ValueError("Magic Prompt output failed JSON verification:\n" + "\n".join(issues))


def _expand_with_openrouter(
  openrouter_model: str,
  prompt: str,
  aspect_ratio: str,
  api_key: str,
) -> str:
  model = openrouter_model.strip()
  if not model:
    raise ValueError("openrouter_model is required when magic_prompt_provider is openrouter.")

  build_messages, openrouter_chat, _ = _load_custom_magic_prompt_helpers()
  try:
    return openrouter_chat(
      model,
      build_messages("v1.txt", prompt, aspect_ratio),
      api_key,
      temperature=1.0,
      timeout=OPENROUTER_TIMEOUT_SECONDS,
    )
  except requests.RequestException as exc:
    raise RuntimeError(_openrouter_error_message(exc, MAGIC_PROMPT_PROVIDER_OPENROUTER, model)) from exc


def _weights_repo_for_model_weights(model_weights: str) -> str:
  if model_weights not in WEIGHT_REPOS:
    raise ValueError(f"Unsupported model_weights: {model_weights}")
  return WEIGHT_REPOS[model_weights]


def _cache_space_hint() -> str:
  cache_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
  probe = cache_home
  while not probe.exists() and probe.parent != probe:
    probe = probe.parent
  try:
    free_gib = shutil.disk_usage(probe).free / (1024**3)
  except OSError:
    return f"HF_HOME={cache_home}"
  return f"HF_HOME={cache_home} ({free_gib:.1f} GiB free under {probe})"


def _raise_pipeline_load_error(exc: Exception) -> None:
  message = str(exc)
  cache_error_markers = (
    "Not enough free disk space",
    "No space left on device",
    "Internal Writer Error",
    "Background writer channel closed",
    "pytorch_model.bin",
  )
  if any(marker in message for marker in cache_error_markers):
    raise RuntimeError(
      "Failed to load Ideogram 4.0 from Hugging Face. This usually means the HF "
      "cache is missing files, out of disk space, or pointed at the wrong cache. "
      f"{_cache_space_hint()}. Start ComfyUI with HF_HOME pointing at a cache "
      "with the Ideogram 4.0 weights available."
    ) from exc
  if isinstance(exc, torch.OutOfMemoryError):
    raise RuntimeError(
      "Failed to load Ideogram 4.0 because CUDA ran out of memory. Restart ComfyUI "
      "and close other GPU jobs before loading."
    ) from exc
  raise exc


def _pil_images_to_comfy(images: list[Any]) -> torch.Tensor:
  if not images:
    raise RuntimeError("Ideogram 4.0 returned no images.")
  arrays = []
  for image in images:
    rgb = image.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    arrays.append(torch.from_numpy(arr))
  return torch.stack(arrays, dim=0)


def _generate_settings_text(
  sampler_preset: str,
  width: int,
  height: int,
  seed: int,
  kwargs: dict[str, Any],
  elapsed: float | None = None,
  multiline: bool = True,
) -> str:
  guidance_label = "preset schedule" if "guidance_schedule" in kwargs else str(kwargs["guidance_scale"])
  lines = [
    f"Preset: {sampler_preset}",
    f"Size: {width}x{height}",
    f"Seed: {seed}",
    f"Steps: {kwargs['num_steps']}",
    f"Guidance: {guidance_label}",
    f"Mu: {kwargs['mu']}",
    f"Std: {kwargs['std']}",
  ]
  if elapsed is not None:
    lines.append(f"Time: {elapsed:.2f}s")
  separator = "\n" if multiline else " | "
  return separator.join(lines)


class Ideogram4PipelineLoader:
  _CACHE: dict[tuple[Any, ...], Any] = {}

  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "model_weights": (list(WEIGHT_REPOS.keys()), {"default": DEFAULT_MODEL_WEIGHTS}),
      },
      "hidden": {"unique_id": "UNIQUE_ID"},
    }

  RETURN_TYPES = (PIPELINE_TYPE,)
  RETURN_NAMES = ("pipeline",)
  FUNCTION = "load"
  CATEGORY = "Ideogram 4.0"

  def load(
    self,
    model_weights: str,
    unique_id: str | None = None,
  ):
    print(f"Ideogram 4.0 Pipeline Loader started: {model_weights}", flush=True)
    _send_progress_text(unique_id, _status_text("Status: Starting", f"Weights: {model_weights}"))
    pbar = _progress(3, unique_id)
    if not torch.cuda.is_available():
      raise RuntimeError(
        "Ideogram 4.0 open weights require a CUDA GPU. Local CPU/MPS runs can "
        "still use the Magic Prompt node, but image generation needs CUDA."
      )
    _apply_hf_token()
    resolved_weights_repo = _weights_repo_for_model_weights(model_weights)
    _send_progress_text(
      unique_id,
      _status_text(
        "Status: Loading weights",
        f"Weights: {model_weights}",
        f"Repo: {resolved_weights_repo}",
      ),
    )
    pbar.update_absolute(1)

    cache_key = (
      model_weights,
      resolved_weights_repo,
      os.environ.get(CORE_REPO_ENV_VAR, ""),
    )
    if cache_key in self._CACHE:
      _send_progress_text(unique_id, _status_text("Status: Ready from cache", f"Weights: {model_weights}"))
      pbar.update_absolute(3)
      print(f"Ideogram 4.0 Pipeline Loader cache hit: {model_weights}", flush=True)
      return (self._CACHE[cache_key],)

    if self._CACHE:
      self._CACHE.clear()
      gc.collect()
      if torch.cuda.is_available():
        torch.cuda.empty_cache()

    Ideogram4Pipeline, Ideogram4PipelineConfig = _load_pipeline_classes()
    config = Ideogram4PipelineConfig(
      weights_repo=resolved_weights_repo,
    )
    pbar.update_absolute(2)
    try:
      pipeline = Ideogram4Pipeline.from_pretrained(
        config=config,
        device=DEFAULT_DEVICE,
        dtype=DEFAULT_TORCH_DTYPE,
      )
    except Exception as exc:
      _raise_pipeline_load_error(exc)
    self._CACHE[cache_key] = pipeline
    _send_progress_text(unique_id, _status_text("Status: Ready", f"Weights: {model_weights}"))
    pbar.update_absolute(3)
    print(f"Ideogram 4.0 Pipeline Loader finished: {model_weights}", flush=True)
    return (pipeline,)


class Ideogram4MagicPrompt:
  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "prompt": ("STRING", {"default": "", "multiline": True}),
        "width": ("INT", {"default": 2048, "min": MIN_RESOLUTION, "max": MAX_RESOLUTION, "step": 16}),
        "height": ("INT", {"default": 2048, "min": MIN_RESOLUTION, "max": MAX_RESOLUTION, "step": 16}),
        "magic_prompt_provider": (
          MAGIC_PROMPT_PROVIDERS,
          {"default": MAGIC_PROMPT_PROVIDER_IDEOGRAM},
        ),
        "openrouter_model": (
          "STRING",
          {
            "default": DEFAULT_OPENROUTER_MODEL,
            "multiline": False,
            "tooltip": "Only used when magic_prompt_provider is openrouter; leave empty for ideogram.",
          },
        ),
        "verify_json": ("BOOLEAN", {"default": True}),
      },
      "hidden": {"unique_id": "UNIQUE_ID"},
    }

  RETURN_TYPES = ("STRING",)
  RETURN_NAMES = ("expanded_prompt",)
  FUNCTION = "expand"
  CATEGORY = "Ideogram 4.0"
  OUTPUT_NODE = True

  @classmethod
  def IS_CHANGED(cls, **kwargs):
    return time.time()

  def expand(
    self,
    prompt: str,
    width: int,
    height: int,
    magic_prompt_provider: str,
    openrouter_model: str,
    verify_json: bool = True,
    unique_id: str | None = None,
  ):
    print(f"Ideogram 4.0 Magic Prompt started: {magic_prompt_provider}", flush=True)
    _send_progress_text(unique_id, _status_text("Status: Starting", f"Provider: {magic_prompt_provider}"))
    pbar = _progress(4, unique_id)
    width, height = _normalize_dimensions(width, height)
    pbar.update_absolute(1)

    _, magic_prompts, aspect_ratio_from_size = _load_magic_prompt_exports()
    if magic_prompt_provider not in MAGIC_PROMPT_PROVIDERS:
      raise ValueError(f"Unsupported magic_prompt_provider: {magic_prompt_provider}")
    api_key = _api_key_for_magic_prompt(magic_prompt_provider)
    aspect_ratio = aspect_ratio_from_size(width, height)
    _send_progress_text(
      unique_id,
      _status_text(
        "Status: Expanding prompt",
        f"Provider: {magic_prompt_provider}",
        f"Aspect ratio: {aspect_ratio}",
      ),
    )
    pbar.update_absolute(2)
    if magic_prompt_provider == MAGIC_PROMPT_PROVIDER_OPENROUTER:
      caption = _expand_with_openrouter(
        openrouter_model,
        prompt,
        aspect_ratio,
        api_key,
      )
      if verify_json:
        _, _, strip_aspect_ratio_and_bboxes = _load_custom_magic_prompt_helpers()
        caption = strip_aspect_ratio_and_bboxes(caption)
      pbar.update_absolute(3)
      _send_progress_text(unique_id, "Status: Verifying JSON" if verify_json else "Status: JSON verification skipped")
      _raise_magic_prompt_issues(caption, bool(verify_json))
      _send_progress_text(unique_id, "Status: Completed")
      pbar.update_absolute(4)
      print("Ideogram 4.0 Magic Prompt finished", flush=True)
      return (caption,)

    if IDEOGRAM_MAGIC_PROMPT_CORE_KEY not in magic_prompts:
      raise ValueError(f"Core magic prompt is missing: {IDEOGRAM_MAGIC_PROMPT_CORE_KEY}")
    magic_prompt = magic_prompts[IDEOGRAM_MAGIC_PROMPT_CORE_KEY](api_key=api_key)
    try:
      expanded_prompt = magic_prompt.expand(prompt, aspect_ratio=aspect_ratio)
    except requests.RequestException as exc:
      raise RuntimeError(_ideogram_api_error_message(exc, magic_prompt_provider)) from exc
    pbar.update_absolute(3)
    _send_progress_text(unique_id, "Status: Verifying JSON" if verify_json else "Status: JSON verification skipped")
    _raise_magic_prompt_issues(expanded_prompt, bool(verify_json))
    _send_progress_text(unique_id, "Status: Completed")
    pbar.update_absolute(4)
    print("Ideogram 4.0 Magic Prompt finished", flush=True)
    return (expanded_prompt,)


class Ideogram4Generate:
  @classmethod
  def INPUT_TYPES(cls):
    sampler_choices = _sampler_preset_choices()
    return {
      "required": {
        "pipeline": (PIPELINE_TYPE, {"forceInput": True}),
        "prompt": ("STRING", {"default": "", "multiline": True}),
        "width": ("INT", {"default": 2048, "min": MIN_RESOLUTION, "max": MAX_RESOLUTION, "step": 16}),
        "height": ("INT", {"default": 2048, "min": MIN_RESOLUTION, "max": MAX_RESOLUTION, "step": 16}),
        "sampler_preset": (
          sampler_choices,
          {"default": _default_sampler_preset()},
        ),
        "num_steps": (
          "INT",
          {
            "default": 20,
            "min": 1,
            "max": 4096,
            "step": 1,
            "tooltip": "Only used when sampler_preset is custom; ignored by named presets.",
            "advanced": True,
          },
        ),
        "guidance_scale": (
          "FLOAT",
          {
            "default": 7.0,
            "min": 0.0,
            "max": 100.0,
            "step": 0.1,
            "tooltip": "Only used when sampler_preset is custom; ignored by named presets.",
            "advanced": True,
          },
        ),
        "mu": (
          "FLOAT",
          {
            "default": 0.0,
            "min": -5.0,
            "max": 5.0,
            "step": 0.1,
            "tooltip": "Only used when sampler_preset is custom; ignored by named presets.",
            "advanced": True,
          },
        ),
        "std": (
          "FLOAT",
          {
            "default": 1.75,
            "min": 0.01,
            "max": 10.0,
            "step": 0.05,
            "tooltip": "Only used when sampler_preset is custom; ignored by named presets.",
            "advanced": True,
          },
        ),
        "seed": (
          "INT",
          {
            "default": 0,
            "min": 0,
            "max": 0xFFFFFFFFFFFFFFFF,
            "step": 1,
            "control_after_generate": True,
          },
        ),
      },
      "hidden": {"unique_id": "UNIQUE_ID"},
    }

  RETURN_TYPES = ("IMAGE",)
  RETURN_NAMES = ("image",)
  FUNCTION = "generate"
  CATEGORY = "Ideogram 4.0"

  def generate(
    self,
    pipeline,
    prompt: str,
    width: int,
    height: int,
    sampler_preset: str,
    num_steps: int,
    guidance_scale: float,
    mu: float,
    std: float,
    seed: int,
    unique_id: str | None = None,
  ):
    _send_progress_text(unique_id, "Status: Starting")
    pbar = _progress(2, unique_id)
    width, height = _normalize_dimensions(width, height)
    kwargs: dict[str, Any] = {
      "num_steps": int(num_steps),
      "guidance_scale": float(guidance_scale),
      "mu": float(mu),
      "std": float(std),
    }
    if sampler_preset != CUSTOM_SAMPLER_PRESET:
      sampler_preset_key = _sampler_core_key_from_label(sampler_preset)
      preset = _load_sampler_presets()[sampler_preset_key]
      kwargs = {
        "num_steps": preset.num_steps,
        "guidance_schedule": preset.guidance_schedule,
        "mu": preset.mu,
        "std": preset.std,
      }

    settings_text = _generate_settings_text(
      sampler_preset,
      width,
      height,
      int(seed),
      kwargs,
    )
    print(
      "Ideogram 4.0 Generate started:\n" + settings_text,
      flush=True,
    )
    _send_progress_text(
      unique_id,
      _status_text("Status: Generating image", settings_text),
    )
    pbar.update_absolute(1)
    start_time = time.perf_counter()
    images = pipeline(
      prompt,
      width=width,
      height=height,
      seed=int(seed),
      raise_on_caption_issues=False,
      **kwargs,
    )
    elapsed = time.perf_counter() - start_time
    print(
      f"Ideogram 4.0 generated {len(images)} image(s) at {width}x{height} "
      f"in {elapsed:.2f}s"
    )
    settings_text = _generate_settings_text(
      sampler_preset,
      width,
      height,
      int(seed),
      kwargs,
      elapsed,
    )
    _send_progress_text(unique_id, _status_text("Status: Completed", settings_text))
    node_text = _generate_settings_text(
      sampler_preset,
      width,
      height,
      int(seed),
      kwargs,
      elapsed,
      multiline=False,
    )
    image_tensor = _pil_images_to_comfy(images)
    pbar.update_absolute(2)
    return {
      "ui": {"text": (node_text,)},
      "result": (image_tensor,),
    }


def register_routes() -> None:
  try:
    from aiohttp import web
    from server import PromptServer
  except Exception:
    return
  instance = PromptServer.instance
  if instance is None:
    return

  @instance.routes.get("/ideogram/keys")
  async def _get_ideogram_keys(request):
    # Booleans only — never return the secret values.
    return web.json_response(_config_status())

  @instance.routes.post("/ideogram/keys")
  async def _set_ideogram_keys(request):
    try:
      body = await request.json()
    except Exception:
      return web.json_response({"error": "invalid JSON body"}, status=400)
    if not isinstance(body, dict):
      return web.json_response({"error": "expected a JSON object"}, status=400)
    return web.json_response(_save_config(body))


NODE_CLASS_MAPPINGS = {
  "Ideogram4PipelineLoader": Ideogram4PipelineLoader,
  "Ideogram4MagicPrompt": Ideogram4MagicPrompt,
  "Ideogram4Generate": Ideogram4Generate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
  "Ideogram4PipelineLoader": "Ideogram 4.0 Pipeline Loader",
  "Ideogram4MagicPrompt": "Ideogram 4.0 Magic Prompt",
  "Ideogram4Generate": "Ideogram 4.0 Generate",
}
