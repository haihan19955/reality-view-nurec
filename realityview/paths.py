"""All on-disk locations, overridable with environment variables.

RV_WORK        working dir for caches / checkpoints / outputs     (default: <repo>/work)
RV_MODELS      third-party models and weights                     (default: ~/models)
RV_CACHE       preprocessed training tensors                      (default: $RV_WORK/cache)
NUREC_SCENES   downloaded NuRec .usdz scenes                      (default: $RV_WORK/nurec_scenes)
NUREC_DATA     dataset rendered from the scenes                   (default: $RV_WORK/nurec_data)
NUREC_PROTO    generated NuRec gRPC python stubs                  (default: $RV_WORK/nurec_proto)
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env(name, default):
    return os.path.abspath(os.path.expanduser(os.environ.get(name, default)))


WORK = _env("RV_WORK", os.path.join(REPO, "work"))
MODELS = _env("RV_MODELS", "~/models")
CACHE = _env("RV_CACHE", os.path.join(WORK, "cache"))
CHECKPOINTS = os.path.join(WORK, "checkpoints")
OUTPUTS = os.path.join(WORK, "outputs")
NUREC_SCENES = _env("NUREC_SCENES", os.path.join(WORK, "nurec_scenes"))
NUREC_DATA = _env("NUREC_DATA", os.path.join(WORK, "nurec_data"))
NUREC_PROTO = _env("NUREC_PROTO", os.path.join(WORK, "nurec_proto"))


def model(*parts):
    """Path inside RV_MODELS, e.g. model("DA3METRIC-LARGE")."""
    return os.path.join(MODELS, *parts)
