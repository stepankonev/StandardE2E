from abc import ABC, abstractmethod
from typing import Any, final

from standard_e2e.data_structures import StandardFrameData
from standard_e2e.enums import Modality, StandardFrameDataField


class AbstractAdapter(ABC):
    """Base class for converting ``StandardFrameData`` into modality payloads."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the adapter. Must be implemented by subclasses."""

    @property
    def metadata(self) -> dict[str, Any]:
        """Per-frame metadata this adapter contributes to ``aux_data``.

        Override in subclasses to surface adapter-side configuration that a
        downstream consumer needs to interpret the modality output (e.g. the
        ordered channel list of a BEV rasterizer). The default is an empty
        dict; the source dataset processor merges these into each frame's
        ``aux_data`` so the .npz remains self-describing.
        """
        return {}

    @property
    def consumes_attrs(self) -> set[StandardFrameDataField]:
        """``StandardFrameData`` fields this adapter reads.

        Used by source-dataset processors to skip building modalities that
        no adapter consumes (lazy-load). For example, an
        :class:`HDMapBEVAdapter` returns ``{StandardFrameDataField.HD_MAP}``;
        a processor whose adapter chain registers no HD-map adapter can then
        skip the (often expensive) ``_build_hd_map`` step entirely.

        Returning an empty set means "this adapter does not gate any
        modality build" — appropriate for adapters that read optional
        ``aux_data`` keys (e.g. preference trajectories) which the
        processor populates unconditionally and cheaply.
        """
        return set()

    @final
    def transform(self, standard_frame_data: StandardFrameData) -> dict[Modality, Any]:
        """Validate input frame and dispatch to subclass implementation."""
        if standard_frame_data is None:
            raise ValueError("standard_frame_data cannot be None")
        if not isinstance(standard_frame_data, StandardFrameData):
            raise TypeError(
                "standard_frame_data must be an instance of StandardFrameData, "
                f"got {type(standard_frame_data)}"
            )
        return self._transform(standard_frame_data)

    @abstractmethod
    def _transform(self, standard_frame_data: StandardFrameData) -> dict[Modality, Any]:
        """
        Transform the StandardFrameData into a format suitable for the specific adapter.
        """
