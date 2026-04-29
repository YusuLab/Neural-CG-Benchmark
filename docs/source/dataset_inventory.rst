Dataset Inventory
=================

This page tracks the local research datasets currently available in the shared
workspace and maps them into the benchmark's domain-first taxonomy. Some
families are now backed by first-class loaders, while others remain cataloged
until their sample schema is normalized.

Dataset families
----------------

.. list-table::
   :header-rows: 1
   :widths: 24 18 14 14 30

   * - Dataset family
     - Domain category
     - Datapoint type
     - Status
     - Main local path
   * - Superblue Raw Circuit Data
     - ``asic_physical_design``
     - ``Graph``
     - cataloged only
     - ``/data/zhishang/data/superblue/raw_data``
   * - Superblue Processed Graph Features
     - ``asic_physical_design``
     - ``Graph``
     - cataloged only
     - ``/data/zhishang/data/superblue/2023-03-06_data``
   * - DEHNN Netlist Dataset
     - ``asic_physical_design``
     - ``Graph``
     - cataloged only
     - ``/data/zhishang/DEHNN/de_hnn``
   * - CircuitNet Design Graphs
     - ``asic_physical_design``
     - ``Graph``
     - cataloged only
     - ``/data/zhishang/CircuitNet/de_hnn_qm``
   * - DREAMPlace ASIC Benchmarks
     - ``asic_physical_design``
     - ``Graph``
     - cataloged only
     - ``/data/zhishang/DREAMPlace/benchmarks``
   * - DREAMPlaceFPGA Benchmarks
     - ``fpga_physical_design``
     - ``Graph``
     - cataloged only
     - ``/data/zhishang/DREAMPlaceFPGA/benchmarks``
   * - ChipDiffusion Graph Dataset
     - ``asic_physical_design``
     - ``Graph``
     - loader supported
     - ``/data/zhishang/chipdiffusion/datasets``
   * - UnifiedLearning ChipGen Logs
     - ``experiment_logs``
     - unmapped
     - logs only
     - ``/data/zhishang/UnifiedLearning/data/chipgen/test_logs``
   * - Shortest Paths Terrain Patches
     - ``terrain_shortest_path``
     - ``Graph``
     - loader supported
     - ``/data/zhishang/shortest-paths-nn/norway_patches``
   * - Synthetic Graph Benchmarks
     - ``synthetic_graph_reasoning``
     - ``Graph``
     - loader supported
     - ``/data/zhishang/to_merge/gnn_agop/data/datasets``
   * - Mirrored GitHub Research Datasets
     - ``mirrors``
     - unmapped
     - mirror only
     - ``/data/zhishang/github``

Category notes
--------------

``asic_physical_design``
   Real ASIC physical-design datasets, placement graphs, and benchmark inputs.
   ``ChipDiffusionPygGraph`` is the active v1 loader in this domain.

``fpga_physical_design``
   FPGA physical-design datasets and benchmarks. These are cataloged, but not
   yet normalized into a stable loader surface.

``terrain_shortest_path``
   Real terrain patch graphs used for shortest-path approximation. These are
   active loaders through the terrain patch dataset classes.

``synthetic_graph_reasoning``
   Synthetic graph-learning corpora used for reasoning and explainability
   experiments. These are active loaders through the single-graph dataset
   classes.

``experiment_logs``
   Run outputs and merged reports. Useful for provenance and analysis, but not a
   dataset family to register directly.

``mirrors``
   Secondary copies of repositories. These should not be treated as canonical
   dataset locations unless the primary path disappears.

Integration plan
----------------

The current v1 loader surface is:

* ``nugets.datasets.terrain_shortest_path``
* ``nugets.datasets.synthetic_graph_reasoning``
* ``nugets.datasets.asic_physical_design``

The FPGA and some additional ASIC corpora remain catalog-only until their
per-sample graph schema is stable enough for a clean loader.

Machine-readable source
-----------------------

The tracked Python manifest for this page lives in
``nugets.datasets.research_catalog``.
