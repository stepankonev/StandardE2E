from standard_e2e.caching.adapters.abstract_adapter import AbstractAdapter
from standard_e2e.caching.adapters.identity_adapters import (
    CamerasIdentityAdapter,
    Detections3DIdentityAdapter,
    FutureStatesIdentityAdapter,
    IntentIdentityAdapter,
    PastStatesIdentityAdapter,
    PreferenceTrajectoryAdapter,
)
from standard_e2e.caching.adapters.pano_adapter import PanoImageAdapter


def get_adapters_from_config(adapter_configs: list[dict]) -> list[AbstractAdapter]:
    """
    Get the adapters from the configuration dictionary.
    """
    name_to_adapter = {
        "cameras_identity_adapter": CamerasIdentityAdapter,
        "future_states_identity_adapter": FutureStatesIdentityAdapter,
        "intent_identity_adapter": IntentIdentityAdapter,
        "past_states_identity_adapter": PastStatesIdentityAdapter,
        "pano_adapter": PanoImageAdapter,
        "preference_trajectory_adapter": PreferenceTrajectoryAdapter,
        "detections_3d_identity_adapter": Detections3DIdentityAdapter,
    }
    adapters = []
    for adapter_config in adapter_configs:
        adapter_name = adapter_config["name"]
        adapter_cls = name_to_adapter.get(adapter_name)
        if adapter_cls:
            params = adapter_config.get("params", {})
            adapter = adapter_cls(**params)
            adapters.append(adapter)
        else:
            raise ValueError(f"Unknown adapter name: {adapter_name}")
    return adapters


__all__ = [
    "AbstractAdapter",
    "CamerasIdentityAdapter",
    "FutureStatesIdentityAdapter",
    "IntentIdentityAdapter",
    "PastStatesIdentityAdapter",
    "PanoImageAdapter",
    "get_adapters_from_config",
    "Detections3DIdentityAdapter",
]
