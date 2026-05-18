Datasets: download and server-local access
==========================================

Dataset availability
--------------------

The benchmark datasets fall into two groups:

**Available on Hugging Face** (``luckyjackluo/Neural-CG-Benchmark``):

.. list-table::
   :header-rows: 1
   :widths: 38 40 22

   * - Entry slug
     - Description
     - Loaders
   * - ``chipdiffusion-graph-dataset``
     - ChipDiffusion ASIC graph (PyG format)
     - ``ChipDiffusionPygGraph``
   * - ``shortest-paths-terrain-patches``
     - Terrain shortest-path patches (Norway, Holland, Phil)
     - ``NorwayTerrainPatches``, ``HollandTerrainPatches``, ``PhilTerrainPatches``
   * - ``synthetic-graph-benchmarks``
     - Synthetic graph reasoning benchmarks (BA-shapes, tree-cycles, etc.)
     - ``BAShapes``, ``BACommunity``, ``TreeCycles``, ``TreeGrid``, ``ShortestPathSynthetic``
   * - ``dehnn-netlist-dataset``
     - DE-HNN MLCAD netlists (~43 GB) + ISPD16 netlists (~1.3 GB)
     - ``DEHNNMLCADNetlistGraphs``, ``DEHNNISPD16NetlistGraphs``
   * - ``circuitnet-design-graphs``
     - CircuitNet standardized instance graphs (~1.4 GB)
     - ``CircuitNetStandardizedGraphs``
   * - ``superblue-processed-graph-features``
     - Superblue processed node-feature and bipartite graphs (~18 GB)
     - ``SuperblueProcessedGraphs``

**Not on Hugging Face**:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Dataset
     - How to obtain
   * - Superblue raw circuit data (~6 GB)
     - Server-local only
   * - DREAMPlace ASIC benchmarks (~2 GB)
     - https://github.com/limbo018/DREAMPlace
   * - DREAMPlaceFPGA benchmarks (~8 GB)
     - https://github.com/zhilix/DREAMPlaceFPGA-MP

Downloading from Hugging Face
------------------------------

Use ``download_research_datasets.py`` to fetch datasets from the HF repo.
Files are saved to ``data/server-local/<entry-slug>/`` by default (this
directory is gitignored).

Install the dependency first::

   uv add huggingface_hub

List what is available::

   python download_research_datasets.py --list

Download everything at once::

   python download_research_datasets.py --all

Download a specific entry::

   python download_research_datasets.py --entry shortest-paths-terrain-patches

Download to a custom directory::

   python download_research_datasets.py --all --dest /path/to/my/datasets

Preview what would be downloaded without fetching::

   python download_research_datasets.py --all --dry-run

The script prints the ``root=`` argument to pass to each loader class after
the download finishes.  For example, after downloading
``chipdiffusion-graph-dataset`` into the default ``data/`` directory::

   from nugets.datasets import get_dataset_register
   register = get_dataset_register()
   ds = register["ChipDiffusionPygGraph"](
       root="data/server-local/chipdiffusion-graph-dataset/datasets/graph/pyg_graph.pt"
   )

Skipping HF login
^^^^^^^^^^^^^^^^^

Set the ``HF_TOKEN`` environment variable and pass ``--skip-login``::

   HF_TOKEN=hf_... python download_research_datasets.py --all --skip-login

Uploading to Hugging Face
--------------------------

Use ``upload_research_datasets.py`` to push server-local datasets to the HF
repo (requires write access).

Dry run::

   python upload_research_datasets.py \
       --repo-id luckyjackluo/Neural-CG-Benchmark \
       --all \
       --dry-run

Upload one entry::

   python upload_research_datasets.py \
       --repo-id luckyjackluo/Neural-CG-Benchmark \
       --entry shortest-paths-terrain-patches

Upload everything::

   python upload_research_datasets.py \
       --repo-id luckyjackluo/Neural-CG-Benchmark \
       --all

The uploader publishes each source folder as
``server-local/<entry-slug>/<folder-name>/`` inside the dataset repo.

Active loaders
--------------

These dataset classes are registered and ready to use:

*ASIC physical design*

* ``ChipDiffusionPygGraph``
* ``DEHNNMLCADNetlistGraphs``
* ``CircuitNetStandardizedGraphs``
* ``SuperblueProcessedGraphs``

*FPGA physical design*

* ``DEHNNISPD16NetlistGraphs``

*Terrain shortest-path*

* ``NorwayTerrainPatches``
* ``HollandTerrainPatches``
* ``PhilTerrainPatches``

*Synthetic graph reasoning*

* ``BAShapes``
* ``BACommunity``
* ``TreeCycles``
* ``TreeGrid``
* ``ShortestPathSynthetic``

Still catalog-only (no active loader yet):

* Superblue raw circuit data
* DREAMPlace ASIC benchmarks
* DREAMPlaceFPGA benchmarks
