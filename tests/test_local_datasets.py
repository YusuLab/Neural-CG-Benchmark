from nugets.datasets import get_dataset_register
from nugets.datasets.datapoint_types import Graph_datapoint


def test_local_dataset_registry_contains_new_entries():
    register = get_dataset_register()
    for name in (
        "NorwayTerrainPatches",
        "HollandTerrainPatches",
        "PhilTerrainPatches",
        "BAShapes",
        "BACommunity",
        "TreeCycles",
        "TreeGrid",
        "ShortestPathSynthetic",
        "ChipDiffusionPygGraph",
    ):
        assert name in register


def test_norway_terrain_loader_shapes_and_split():
    register = get_dataset_register()
    dataset_type = register["NorwayTerrainPatches"]
    train = dataset_type(which="train", length=4, split_seed=7)
    val = dataset_type(which="val", length=4, split_seed=7)
    ood = dataset_type(which="ood", length=4, split_seed=7)

    assert len(train) > 0
    sample = train[0]
    assert isinstance(sample, Graph_datapoint)
    assert sample.pointset.ndim == 2
    assert sample.edges.ndim == 2
    assert sample.edges.shape[0] == 2

    train_names = {path.name for path in train.paths}
    val_names = {path.name for path in val.paths}
    assert train_names.isdisjoint(val_names)
    assert [path.name for path in val.paths] == [path.name for path in ood.paths]


def test_synthetic_graph_loader_shapes():
    register = get_dataset_register()
    dataset = register["BAShapes"](which="train")
    sample = dataset[0]

    assert isinstance(sample, Graph_datapoint)
    assert sample.pointset.ndim == 2
    assert sample.edges.ndim == 2
    assert sample.edges.shape[0] == 2


def test_chipdiffusion_loader_shape():
    register = get_dataset_register()
    dataset = register["ChipDiffusionPygGraph"](which="train")
    sample = dataset[0]

    assert isinstance(sample, Graph_datapoint)
    assert sample.pointset.ndim == 2
    assert sample.edges.ndim == 2
    assert sample.edges.shape[0] == 2
