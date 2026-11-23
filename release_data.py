"""Load precomputed LearnSphere data from a reviewed public GitHub Release.

Only use public, explicitly approved, pseudonymized data. Never upload raw
training data, enrollment records, or private user identifiers here.
"""
from __future__ import annotations

import csv
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPOSITORY = "omni6361-spec/LearnSphere"
RELEASE_TAG = "learnsphere-data-v1"
EXPECTED_RECOMMENDATIONS = 177_247
SHARDS = [f"xai_batch_{i:03d}.sqlite" for i in range(1, 13)]
DOWNLOAD_LIMITS = {
    "recommendations.csv": 100_000_000,
    **{name: 150_000_000 for name in SHARDS},
    "public_analytics.json": 10_000_000,
}


class ReleaseDataError(RuntimeError):
    """A complete, valid public data release is not available."""


def _download(url: str, destination: Path, max_bytes: int) -> None:
    request = Request(url, headers={"User-Agent": "LearnSphere-Streamlit/1.0"})
    downloaded = 0
    with urlopen(request, timeout=120) as response, destination.open("wb") as output:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            downloaded += len(block)
            if downloaded > max_bytes:
                raise ReleaseDataError(f"Asset too large: {destination.name}")
            output.write(block)
    if downloaded == 0:
        raise ReleaseDataError(f"Empty release asset: {destination.name}")


def _check_recommendations(path: Path) -> set[tuple[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        needed = {"userid_DI", "course_id", "rank", "recommendation_score"}
        if not needed.issubset(reader.fieldnames or []):
            raise ReleaseDataError("Invalid recommendations.csv columns")
        keys: set[tuple[str, str]] = set()
        count = 0
        for record in reader:
            key = (record["userid_DI"], record["course_id"])
            if not key[0].startswith("Learner ") or not key[1]:
                raise ReleaseDataError("Invalid learner/course identifier")
            if key in keys:
                raise ReleaseDataError("Duplicate recommendation")
            keys.add(key)
            count += 1
            float(record["recommendation_score"])
            int(record["rank"])
    if count != EXPECTED_RECOMMENDATIONS:
        raise ReleaseDataError(
            f"Expected {EXPECTED_RECOMMENDATIONS:,} recommendations; got {count:,}"
        )
    return keys


def _check_shards(paths: list[Path], expected: set[tuple[str, str]]) -> None:
    found: set[tuple[str, str]] = set()
    for shard in paths:
        if not shard.is_file() or shard.stat().st_size < 4096:
            raise ReleaseDataError(f"Missing or empty SQLite shard: {shard.name}")
        with sqlite3.connect(shard.as_uri() + "?mode=ro", uri=True) as db:
            if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ReleaseDataError(f"Corrupt SQLite shard: {shard.name}")
            for learner, course in db.execute(
                "SELECT learner, course FROM explanations"
            ):
                key = (learner, course)
                if key not in expected:
                    raise ReleaseDataError(f"Unexpected key in {shard.name}")
                if key in found:
                    raise ReleaseDataError(f"Duplicate key across shards: {shard.name}")
                found.add(key)
    if found != expected:
        raise ReleaseDataError(
            f"Explanations missing for {len(expected - found):,} recommendations"
        )


def ensure_release_data(data_dir: Path) -> bool:
    """Download all 13 files once if recommendations.csv is missing.

    Publish an approved GitHub Release with the tag learnsphere-data-v1.
    Assets must be recommendations.csv and all 12 xai_batch_NNN.sqlite files.
    An optional public_analytics.json enables the aggregated dashboard.
    """
    data_dir = Path(data_dir)
    destination = data_dir / "local_xai"
    if (data_dir / "recommendations.csv").is_file() and all(
        (destination / name).is_file() for name in SHARDS
    ):
        return False
    api = f"https://api.github.com/repos/{REPOSITORY}/releases/tags/{RELEASE_TAG}"
    try:
        request = Request(
            api,
            headers={"User-Agent": "LearnSphere-Streamlit/1.0",
                     "Accept": "application/vnd.github+json"},
        )
        with urlopen(request, timeout=30) as response:
            release = json.load(response)
    except HTTPError as exc:
        if exc.code == 404:
            raise ReleaseDataError(
                f"Public data release '{RELEASE_TAG}' has not been published. "
                f"Upload the reviewed 13 assets at "
                f"https://github.com/{REPOSITORY}/releases/new"
            ) from exc
        raise ReleaseDataError(f"Release lookup failed: HTTP {exc.code}") from exc
    except (URLError, TimeoutError, ValueError) as exc:
        raise ReleaseDataError(f"Cannot contact GitHub release: {exc}") from exc

    assets = {
        item["name"]: item["browser_download_url"]
        for item in release.get("assets", [])
    }
    needed = ["recommendations.csv", *SHARDS]
    missing = [name for name in needed if name not in assets]
    if missing:
        raise ReleaseDataError(
            "Dataset release is incomplete; missing " + ", ".join(missing[:5])
        )
    prefix = (
        f"https://github.com/{REPOSITORY}/releases/download/{RELEASE_TAG}/"
    )
    for name in [*needed, "public_analytics.json"]:
        if name in assets and assets[name] != prefix + name:
            raise ReleaseDataError("Unexpected asset download URL")

    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="learnsphere-release-", dir=data_dir
    ) as temp:
        stage = Path(temp)
        for name in needed:
            try:
                _download(assets[name], stage / name, DOWNLOAD_LIMITS[name])
            except (HTTPError, URLError, TimeoutError, OSError) as exc:
                raise ReleaseDataError(f"Failed to download {name}: {exc}") from exc
        expected = _check_recommendations(stage / "recommendations.csv")
        _check_shards([stage / name for name in SHARDS], expected)

        if "public_analytics.json" in assets:
            try:
                _download(
                    assets["public_analytics.json"],
                    stage / "public_analytics.json",
                    DOWNLOAD_LIMITS["public_analytics.json"],
                )
                json.loads(
                    (stage / "public_analytics.json").read_text(encoding="utf-8")
                )
            except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
                raise ReleaseDataError(
                    f"Could not fetch aggregate metrics: {exc}"
                ) from exc

        destination.mkdir(parents=True, exist_ok=True)
        os.replace(stage / "recommendations.csv", data_dir / "recommendations.csv")
        for name in SHARDS:
            os.replace(stage / name, destination / name)
        if (stage / "public_analytics.json").exists():
            os.replace(
                stage / "public_analytics.json",
                data_dir / "public_analytics.json",
            )
    return True
