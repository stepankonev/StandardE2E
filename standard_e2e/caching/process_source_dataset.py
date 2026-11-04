import argparse
import importlib
import logging
from typing import Protocol, Type, cast

from standard_e2e.caching.adapters import get_adapters_from_config
from standard_e2e.utils import load_yaml_config

# Dataset name -> (converter class, processor class), both exported by the
# package ``standard_e2e.caching.src_datasets.<dataset name>``. Packages are
# imported on demand because each may need its own SDK (TensorFlow for Waymo,
# av2, nuplan-devkit, ...), installed only in that dataset's environment under
# ``datasets/<dataset name>/``.
DATASETS: dict[str, tuple[str, str]] = {
    "waymo_e2e": ("WaymoE2EDatasetConverter", "WaymoE2EDatasetProcessor"),
    "waymo_perception": (
        "WaymoPerceptionDatasetConverter",
        "WaymoPerceptionDatasetProcessor",
    ),
    "av2_sensor": ("Av2SensorDatasetConverter", "Av2SensorDatasetProcessor"),
    "av2_lidar": ("Av2LidarDatasetConverter", "Av2LidarDatasetProcessor"),
    "navsim": ("NavsimDatasetConverter", "NavsimDatasetProcessor"),
    "nuscenes": ("NuscenesDatasetConverter", "NuscenesDatasetProcessor"),
    "wayve_scenes": ("WayveScenesDatasetConverter", "WayveScenesDatasetProcessor"),
    "comma2k19": ("Comma2k19DatasetConverter", "Comma2k19DatasetProcessor"),
    "truckdrive": ("TruckDriveDatasetConverter", "TruckDriveDatasetProcessor"),
    "vod": ("VodDatasetConverter", "VodDatasetProcessor"),
    "kitscenes_multimodal": (
        "KITScenesMultimodalDatasetConverter",
        "KITScenesMultimodalDatasetProcessor",
    ),
    "kitscenes_longtail": (
        "KITScenesLongTailDatasetConverter",
        "KITScenesLongTailDatasetProcessor",
    ),
    "natix_multicam": (
        "NatixMulticamDatasetConverter",
        "NatixMulticamDatasetProcessor",
    ),
    "natix_edgecase": (
        "NatixEdgeCaseDatasetConverter",
        "NatixEdgeCaseDatasetProcessor",
    ),
    "muhos": ("MuhosDatasetConverter", "MuhosDatasetProcessor"),
}


class _HasArgParser(Protocol):  # pylint: disable=too-few-public-methods
    @classmethod
    def get_arg_parser(cls) -> argparse.ArgumentParser:  # pragma: no cover
        ...


def get_dataset_arg_name(argv=None):
    """Get the dataset name from the command line arguments."""
    base = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    base.add_argument("dataset_name")
    ns, rest = base.parse_known_args(argv)
    return ns.dataset_name, rest


def load_dataset_classes(dataset_name: str) -> tuple[type, type]:
    """Import a dataset's package and return its (converter, processor) classes.

    A missing third-party module means the dataset's SDK is not installed in
    the running environment; the error then names the environment to use.
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    converter_name, processor_name = DATASETS[dataset_name]
    package = f"standard_e2e.caching.src_datasets.{dataset_name}"
    try:
        module = importlib.import_module(package)
    except ModuleNotFoundError as error:
        if error.name is None or error.name.split(".")[0] == "standard_e2e":
            raise
        raise ModuleNotFoundError(
            f"Dataset {dataset_name!r} needs {error.name!r}, which is not "
            f"installed in this environment. Run it in the dataset's own "
            f"environment: `uvx stepix run datasets/{dataset_name}/README.md`, "
            f"or prefix the command with `uv run --project datasets/{dataset_name}`.",
            name=error.name,
        ) from error
    return getattr(module, converter_name), getattr(module, processor_name)


def main(argv=None):
    dataset_name, rest = get_dataset_arg_name(argv)
    dataset_converter_cls, dataset_processor_cls = load_dataset_classes(dataset_name)

    # Help mypy: assert dataset_converter_cls implements get_arg_parser
    converter_cls_typed = cast(Type[_HasArgParser], dataset_converter_cls)
    arguments = converter_cls_typed.get_arg_parser().parse_args(rest)
    config = load_yaml_config(arguments.config_file)
    adapters = get_adapters_from_config(config["preprocessing"]["adapters"])
    dataset_processor = dataset_processor_cls(
        common_output_path=arguments.output_path,
        split=arguments.split,
        adapters=adapters,
    )
    dataset_converter = dataset_converter_cls(
        source_processor=dataset_processor,
        input_path=arguments.input_path,
        split=arguments.split,
        num_workers=arguments.num_workers,
        do_parallel_processing=arguments.do_parallel_processing,
        arguments=arguments,
    )
    dataset_converter.convert()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
