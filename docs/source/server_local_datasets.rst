Server-local datasets and Hugging Face export
=============================================

How the benchmark currently works
---------------------------------

The current NUGETS codebase assumes two different remote storage patterns:

1. Raw datasets are often fetched from the internet by dataset loaders and then
   cached under ``workdir/datasets/raw``.
2. Processed task datasets are cached as tar files under
   ``workdir/datasets/processed`` and can optionally be downloaded from or
   uploaded to GCS through :mod:`nugets.tasks.task`.

This is a reasonable fit for public datasets such as ModelNet, ShapeNet, or the
terrain loaders, but it does not match the local research datasets in the shared
server workspace. Those datasets already exist on disk under ``/data/zhishang``
and should be treated as canonical local sources rather than something the
benchmark downloads itself.

What changed
------------

We added three tracked integration pieces:

* ``nugets.datasets.research_catalog``:
  a machine-readable manifest of the known server-local dataset families.
* ``upload_research_datasets.py``:
  a Hugging Face uploader that publishes selected local dataset folders with
  ``huggingface_hub.upload_folder``.
* domain-first local loaders under ``nugets.datasets``:
  active v1 support for terrain shortest-path graphs, synthetic graph
  reasoning datasets, and the main ChipDiffusion ASIC graph.

Why we do not upload from repo root
-----------------------------------

This snippet is directionally correct:

.. code-block:: python

   from huggingface_hub import login, upload_folder

   login()
   upload_folder(folder_path=".", repo_id="luckyjackluo/Neural-CG-Benchmark", repo_type="dataset")

But running it at the repository root would upload the benchmark source tree,
not the actual datasets stored elsewhere on the server. For our case, the
correct ``folder_path`` must point at the dataset directories themselves.

Uploader usage
--------------

Dry run:

.. code-block:: console

   python upload_research_datasets.py \
       --repo-id luckyjackluo/Neural-CG-Benchmark \
       --all \
       --dry-run

Upload one dataset family:

.. code-block:: console

   python upload_research_datasets.py \
       --repo-id luckyjackluo/Neural-CG-Benchmark \
       --entry shortest-paths-terrain-patches

Upload the catalog plus all primary datasets:

.. code-block:: console

   python upload_research_datasets.py \
       --repo-id luckyjackluo/Neural-CG-Benchmark \
       --all

What gets uploaded
------------------

The uploader does two things:

1. It uploads a generated metadata bundle to
   ``server-local/catalog`` inside the target dataset repo.
2. It uploads each selected local source folder to
   ``server-local/<entry-slug>/<source-folder-name>``.

For example, ``shortest-paths-terrain-patches`` uploads the actual directories
like:

* ``/data/zhishang/shortest-paths-nn/norway_patches``
* ``/data/zhishang/shortest-paths-nn/holland_patches``
* ``/data/zhishang/shortest-paths-nn/phil_patches``

Integration status
------------------

The benchmark now has:

* a tracked inventory of server-local datasets,
* a documented distinction between downloaded public datasets and local
  canonical datasets,
* a Hugging Face export path using ``login`` and ``upload_folder``,
* first-class dataset loaders for the stable local graph corpora.

Active v1 loaders:

* ``NorwayTerrainPatches``
* ``HollandTerrainPatches``
* ``PhilTerrainPatches``
* ``BAShapes``
* ``BACommunity``
* ``TreeCycles``
* ``TreeGrid``
* ``ShortestPathSynthetic``
* ``ChipDiffusionPygGraph``

Still catalog-only:

* FPGA datasets whose sample schema is not yet normalized
* DEHNN, CircuitNet, and Superblue variants that still need a canonical
  benchmark-facing graph format
