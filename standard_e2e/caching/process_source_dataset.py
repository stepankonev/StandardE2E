import argparse
import logging
from typing import Protocol, Type, cast

from standard_e2e.caching.adapters import get_adapters_from_config
from standard_e2e.caching.src_datasets.waymo_e2e import (
    WaymoE2EDatasetConverter,
    WaymoE2EDatasetProcessor,
)
from standard_e2e.caching.src_datasets.waymo_perception import (
    WaymoPerceptionDatasetConverter,
    WaymoPerceptionDatasetProcessor,
)
from standard_e2e.utils import load_yaml_config


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


def main(argv=None):
    dataset_name, rest = get_dataset_arg_name(argv)
    dataset_converter_cls = {
        "waymo_e2e": WaymoE2EDatasetConverter,
        "waymo_perception": WaymoPerceptionDatasetConverter,
    }.get(dataset_name)
    if dataset_converter_cls is None:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    dataset_processor_cls = {
        "waymo_e2e": WaymoE2EDatasetProcessor,
        "waymo_perception": WaymoPerceptionDatasetProcessor,
    }.get(dataset_name)
    if dataset_processor_cls is None:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

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
