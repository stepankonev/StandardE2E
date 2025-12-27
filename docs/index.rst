StandardE2E Documentation
=========================

**Unified end-to-end autonomous driving dataset processing and loading.**

StandardE2E standardizes multimodal autonomous driving datasets into a single, unified format for training models on multiple datasets using one consistent API.

.. image:: ../assets/standard_e2e_scheme.png
   :alt: StandardE2E Architecture
   :align: center
   :width: 80%

|

Key Features
------------

✨ **Unified Data Format** - Single representation for all datasets

🔄 **Multimodal Support** - Camera, trajectories, detections, WIP LiDAR, maps, etc

⚙️ **Parametrizable Pipelines** - Configure programmatically or via YAML

🚀 **PyTorch Native** - Seamless DataLoader integration

📦 **Extensible** - Add new datasets, adapters, augmentations, etc.

Getting Started
---------------

From PyPI:

.. code-block:: bash

   pip install standard-e2e

Or for development:

.. code-block:: bash

   git clone https://github.com/stepankonev/StandardE2E.git
   cd StandardE2E
   conda create -n standard_e2e python=3.12
   conda activate standard_e2e
   pip install -e .

Refer to the :doc:`quickstart` guide for detailed usage.

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   quickstart
   overview
   user_guide

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   reference/api

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   📓 Introduction Tutorial <https://github.com/stepankonev/StandardE2E/blob/main/notebooks/intro_tutorial.ipynb>
   📓 Data Containers <https://github.com/stepankonev/StandardE2E/blob/main/notebooks/containers.ipynb>
   📓 Multi-Dataset Training <https://github.com/stepankonev/StandardE2E/blob/main/notebooks/multi_dataset_training_and_filtering.ipynb>
   📓 Custom Adapters <https://github.com/stepankonev/StandardE2E/blob/main/notebooks/creating_custom_adapter.ipynb>

.. toctree::
   :caption: Project Links
   :hidden:

   GitHub <https://github.com/stepankonev/StandardE2E>
   Discord <https://discord.gg/vJnQNcQGQ8>
   PyPI <https://pypi.org/project/standard-e2e/>

Community & Support
-------------------

Reach out to the maintainers via GitHub issues or Discord.

- **GitHub Issues**: https://github.com/stepankonev/StandardE2E/issues
- **Discord**: https://discord.gg/vJnQNcQGQ8

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
