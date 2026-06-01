from __future__ import annotations

import argparse
from pathlib import Path


HF_REPO_ID = "luckyjackluo/Neural-CG-Benchmark"
HF_PREFIX = "server-local"

# Entries that are published on the HuggingFace dataset repo.
# Each entry maps to the sub-folder name under server-local/ and carries
# loader hints that tell users what root= argument to pass after download.
HF_ENTRIES: dict[str, dict] = {
    "chipdiffusion-graph-dataset": {
        "description": "ChipDiffusion graph dataset (ASIC physical design)",
        "loader_hints": [
            (
                "ChipSyntheticDataset",
                "{hf_root}/chipdiffusion-graph-dataset",
            ),
        ],
    },
    "dehnn-netlist-dataset": {
        "description": "DE-HNN netlist graphs — MLCAD (ASIC, ~43 GB) and ISPD16 (FPGA, ~1.3 GB)",
        "loader_hints": [
            (
                "DEHNNMLCADCongestionDataset",
                "{hf_root}/dehnn-netlist-dataset/all_designs_netlist_data",
            ),
            (
                "DEHNNISPD16SiteUtilizationDataset",
                "{hf_root}/dehnn-netlist-dataset/ispd16_netlist_data",
            ),
        ],
    },
    "circuitnet-design-graphs": {
        "description": "CircuitNet standardized instance-congestion data (ASIC, ~1.4 GB)",
        "loader_hints": [
            (
                "CircuitNetCongestionDataset",
                "{hf_root}/circuitnet-design-graphs/processed_standardized",
            ),
        ],
    },
    "superblue-processed-graph-features": {
        "description": "Superblue processed congestion graph features and targets (ASIC, ~18 GB)",
        "loader_hints": [
            (
                "SuperblueCongestionDataset",
                "{hf_root}/superblue-processed-graph-features/2023-03-06_data",
            ),
        ],
    },
}

# Entries NOT on HuggingFace, with a note on where to get them.
NOT_ON_HF: dict[str, str] = {
    "superblue-raw-circuit-data": (
        "Superblue raw circuit data (~6 GB). Server-local only."
    ),
    "dreamplace-asic-benchmarks": (
        "DREAMPlace ASIC benchmarks (~2 GB). See https://github.com/limbo018/DREAMPlace"
    ),
    "dreamplacefpga-benchmarks": (
        "DREAMPlaceFPGA benchmarks (~8 GB). See https://github.com/zhilix/DREAMPlaceFPGA-MP"
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download research datasets from the Hugging Face dataset repo "
            f"{HF_REPO_ID} into a local directory."
        )
    )
    parser.add_argument(
        "--dest",
        default="data",
        help=(
            "Local directory to download into. Files land at "
            "<dest>/server-local/<entry-slug>/. Default: data/"
        ),
    )
    parser.add_argument(
        "--repo-id",
        default=HF_REPO_ID,
        help=f"Hugging Face dataset repo to download from. Default: {HF_REPO_ID}",
    )
    parser.add_argument(
        "--entry",
        action="append",
        default=[],
        metavar="SLUG",
        help="Entry slug to download. Can be passed multiple times.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all entries available on Hugging Face.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available entries and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without downloading anything.",
    )
    parser.add_argument(
        "--skip-login",
        action="store_true",
        help="Skip interactive huggingface_hub.login(). Set HF_TOKEN env var instead.",
    )
    return parser


def list_entries() -> None:
    print("Available on Hugging Face:")
    for slug, info in HF_ENTRIES.items():
        print(f"  {slug}")
        print(f"    {info['description']}")
    print()
    print("Not on Hugging Face (server-local or external source):")
    for slug, note in NOT_ON_HF.items():
        print(f"  {slug}")
        print(f"    {note}")


def select_entries(args: argparse.Namespace) -> list[str]:
    if args.all:
        return list(HF_ENTRIES)
    selected = []
    for slug in args.entry:
        if slug in HF_ENTRIES:
            selected.append(slug)
        elif slug in NOT_ON_HF:
            print(f"[skip] '{slug}' is not on Hugging Face: {NOT_ON_HF[slug]}")
        else:
            valid = ", ".join(sorted(HF_ENTRIES))
            raise SystemExit(f"Unknown entry slug '{slug}'. Available: {valid}")
    if not selected:
        raise SystemExit("No entries selected. Use --all or --entry <slug>. Run --list to see options.")
    return selected


def print_loader_hints(slug: str, hf_root: str) -> None:
    hints = HF_ENTRIES[slug]["loader_hints"]
    print(f"  Loader root paths for {slug}:")
    for cls_name, template in hints:
        root = template.format(hf_root=hf_root)
        print(f"    {cls_name}(root={root!r})")


def download_entry(*, snapshot_download, repo_id: str, slug: str, dest: Path, dry_run: bool) -> None:
    pattern = f"{HF_PREFIX}/{slug}/**"
    hf_root = str(dest / HF_PREFIX)

    if dry_run:
        print(f"[dry-run] would download {repo_id}:{HF_PREFIX}/{slug}/ -> {dest / HF_PREFIX / slug}/")
        print_loader_hints(slug, hf_root)
        return

    print(f"Downloading {slug} ...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[pattern],
        local_dir=str(dest),
        local_dir_use_symlinks=False,
    )
    print(f"  -> {dest / HF_PREFIX / slug}/")
    print_loader_hints(slug, hf_root)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        list_entries()
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install it with: uv add huggingface_hub"
        ) from exc

    if not args.skip_login and not args.dry_run:
        try:
            from huggingface_hub import login
            login()
        except Exception:
            pass

    selected = select_entries(args)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    for slug in selected:
        download_entry(
            snapshot_download=snapshot_download,
            repo_id=args.repo_id,
            slug=slug,
            dest=dest,
            dry_run=args.dry_run,
        )

    if not args.dry_run:
        print()
        print("Done. Pass the root= paths above to the corresponding dataset classes.")


if __name__ == "__main__":
    main()
