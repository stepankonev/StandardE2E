"""Identity adapter for HD-map data.

Lifts ``StandardFrameData.hd_map`` (already in ego frame, per the
processor's responsibility) into ``Modality.HD_MAP``. Kept in its own
file because the HD-map convert path lands in two passes — the
processor's adapter list usually emits nothing for the per-frame seed
(no source has per-frame HD maps) and the per-source
``HDMapEgoCropAggregator`` overwrites the slot at segment time with the
ego-cropped vector data.
"""

from standard_e2e.caching.adapters.identity_adapters import IdentityAdapter
from standard_e2e.enums import Modality


class HDMapIdentityAdapter(IdentityAdapter):
    """Identity adapter for HD-map data."""

    def __init__(self):
        super().__init__(Modality.HD_MAP, "hd_map")

    @property
    def name(self) -> str:
        return "HDMapIdentityAdapter"
