from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
from pathlib import Path


def load_research_catalog_module():
    module_path = Path(__file__).resolve().parent / "nugets" / "datasets" / "research_catalog.py"
    spec = importlib.util.spec_from_file_location("nugets_research_catalog", module_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load research catalog module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


catalog_module = load_research_catalog_module()
LOCAL_DATASET_CATALOG = catalog_module.LOCAL_DATASET_CATALOG
DatasetCatalogEntry = catalog_module.DatasetCatalogEntry
get_catalog_map = catalog_module.get_catalog_map
get_uploadable_entries = catalog_module.get_uploadable_entries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upload server-local research datasets to a Hugging Face dataset repo."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target Hugging Face dataset repo, for example luckyjackluo/Neural-CG-Benchmark.",
    )
    parser.add_argument(
        "--entry",
        action="append",
        default=[],
        help="Catalog entry slug to upload. Can be passed multiple times.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Upload all cataloged datasets except logs and mirrors.",
    )
    parser.add_argument(
        "--include-logs",
        action="store_true",
        help="Include log-only entries in selection.",
    )
    parser.add_argument(
        "--include-mirrors",
        action="store_true",
        help="Include mirrored repo copies in selection.",
    )
    parser.add_argument(
        "--path-prefix",
        default="server-local",
        help="Prefix inside the Hugging Face dataset repo.",
    )
    parser.add_argument(
        "--skip-login",
        action="store_true",
        help="Skip interactive huggingface_hub.login(). Useful if HF_TOKEN is already configured.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned uploads without performing them.",
    )
    return parser


def select_entries(args: argparse.Namespace) -> tuple[DatasetCatalogEntry, ...]:
    catalog_map = get_catalog_map()
    if args.all:
        selected = list(
            get_uploadable_entries(
                include_logs=args.include_logs,
                include_mirrors=args.include_mirrors,
            )
        )
    else:
        selected = []
        for slug in args.entry:
            if slug not in catalog_map:
                valid = ", ".join(sorted(catalog_map))
                raise SystemExit(f"Unknown entry slug '{slug}'. Valid values: {valid}")
            selected.append(catalog_map[slug])

    if not selected:
        raise SystemExit("No dataset entries selected. Use --all or pass --entry.")
    return tuple(selected)


def upload_catalog_metadata(
    *,
    upload_folder,
    repo_id: str,
    path_prefix: str,
    dry_run: bool,
) -> None:
    manifest = []
    for entry in LOCAL_DATASET_CATALOG:
        manifest.append(
            {
                "name": entry.name,
                "slug": entry.slug(),
                "category": entry.category,
                "domain_module": entry.domain_module,
                "description": entry.description,
                "local_paths": list(entry.local_paths),
                "existing_local_paths": list(entry.existing_local_paths()),
                "formats": list(entry.formats),
                "suggested_datapoint_type": entry.suggested_datapoint_type,
                "status": entry.status,
            }
        )

    with tempfile.TemporaryDirectory(prefix="hf_dataset_catalog_") as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "catalog.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (tmp_path / "README.md").write_text(
            "# Neural-CG-Benchmark server-local dataset catalog\n\n"
            "This folder was generated from `nugets.datasets.research_catalog`.\n",
            encoding="utf-8",
        )
        if dry_run:
            print(f"[dry-run] metadata folder -> {repo_id}:{path_prefix}/catalog from {tmp_path}")
            return
        upload_folder(
            folder_path=str(tmp_path),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=f"{path_prefix}/catalog",
        )


def upload_entry(*, upload_folder, repo_id: str, path_prefix: str, entry: DatasetCatalogEntry, dry_run: bool) -> None:
    existing_paths = entry.existing_local_paths()
    if not existing_paths:
        print(f"[skip] {entry.slug()} has no existing local paths")
        return

    for source_path in existing_paths:
        src = Path(source_path)
        dest = f"{path_prefix}/{entry.slug()}/{src.name}"
        if dry_run:
            print(f"[dry-run] {src} -> {repo_id}:{dest}")
            continue
        upload_folder(
            folder_path=str(src),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=dest,
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        from huggingface_hub import login, upload_folder
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required for uploads. Install it first, for example with `uv add huggingface_hub`."
        ) from exc

    selected = select_entries(args)

    if not args.skip_login and not args.dry_run:
        login()

    upload_catalog_metadata(
        upload_folder=upload_folder,
        repo_id=args.repo_id,
        path_prefix=args.path_prefix,
        dry_run=args.dry_run,
    )

    for entry in selected:
        upload_entry(
            upload_folder=upload_folder,
            repo_id=args.repo_id,
            path_prefix=args.path_prefix,
            entry=entry,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
