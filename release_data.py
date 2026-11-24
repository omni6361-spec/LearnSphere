"""Load precomputed LearnSphere data from a reviewed public GitHub Release.

Only use public, explicitly approved, pseudonymized data. Never upload raw
training data, enrollment records, or private user identifiers here.
"""
from __future__ import annotations

import csv
import hashlib
import json
import zipfile
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
BUNDLE_NAME = "LearnSphere_Data_One_Upload.zip"
BUNDLE_SHA256 = "16e5ceca7787960044240ccff44f7d4d4ca1c8bd3f4d9e8a8721302b3731cdf7"
PART_SHAS = {
    "LearnSphere_Data_Part_1.zip": "7af8902e8fd236fe7b1024f7fa6ae6119163080a1cd5abeca8126c764fa4f90b",
    "LearnSphere_Data_Part_2.zip": "ecec2d34ec606385ca2740bb193d7bdc266af09d3c5a64bd682e4ebac941bb99",
    "LearnSphere_Data_Part_3.zip": "12353e0af57701f5ba30db6ab378de8b30f1a277d294f118d03c5cf99bb7ea3f",
    "LearnSphere_Data_Part_4.zip": "f1535ecac6403b6ec7d8ba7b347688e1b396175603c8f96e6e1772e1e224bbae",
}
PART_CONTENTS = {
    "LearnSphere_Data_Part_1.zip": set(SHARDS[:3]),
    "LearnSphere_Data_Part_2.zip": set(SHARDS[3:6]),
    "LearnSphere_Data_Part_3.zip": set(SHARDS[6:9]),
    "LearnSphere_Data_Part_4.zip": {"recommendations.csv", *SHARDS[9:]},
}
DOWNLOAD_LIMITS = {
    BUNDLE_NAME: 110_000_000,
    **{name: 35_000_000 for name in PART_SHAS},
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
    """Download and verify precomputed data from an approved GitHub Release.

    The release accepts either one bundle, four small parts, or 13 original
    files. All prepackaged ZIP assets are pinned by SHA256 and checked before
    extraction; individual SQLite shards are integrity-checked after download.
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
    if BUNDLE_NAME in assets:
        packaging = "bundle"
        chosen = [BUNDLE_NAME]
    elif all(name in assets for name in PART_SHAS):
        packaging = "parts"
        chosen = list(PART_SHAS)
    else:
        packaging = "individual"
        chosen = needed
        missing = [name for name in needed if name not in assets]
        if missing:
            raise ReleaseDataError(
                "Upload LearnSphere_Data_One_Upload.zip OR all four "
                "LearnSphere_Data_Part_N.zip files. Missing: "
                + ", ".join(missing[:5])
            )
    prefix = f"https://github.com/{REPOSITORY}/releases/download/{RELEASE_TAG}/"
    for name in [*chosen, "public_analytics.json"]:
        if name in assets and assets[name] != prefix + name:
            raise ReleaseDataError("Unexpected asset download URL")

    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="learnsphere-release-", dir=data_dir
    ) as temp:
        stage = Path(temp)
        if packaging == "individual":
            for name in chosen:
                try:
                    _download(assets[name], stage / name, DOWNLOAD_LIMITS[name])
                except (HTTPError, URLError, TimeoutError, OSError) as exc:
                    raise ReleaseDataError(f"Failed to download {name}: {exc}") from exc
        else:
            for name in chosen:
                try:
                    archived = stage / name
                    _download(assets[name], archived, DOWNLOAD_LIMITS[name])
                    expected_sha = BUNDLE_SHA256 if packaging == "bundle" else PART_SHAS[name]
                    digest = hashlib.sha256()
                    with archived.open("rb") as handle:
                        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(chunk)
                    if digest.hexdigest() != expected_sha:
                        raise ReleaseDataError(
                            f"{name} failed SHA256 verification. Do not repack the archive."
                        )
                    with zipfile.ZipFile(archived, "r") as archive:
                        expected_names = (set(needed) if packaging == "bundle"
                                          else PART_CONTENTS[name])
                        entries = archive.infolist()
                        if (len(entries) != len(expected_names) or
                            {entry.filename for entry in entries} != expected_names or
                            sum(entry.file_size for entry in entries) > 600_000_000):
                            raise ReleaseDataError(f"Unexpected files in {name}")
                        for entry in entries:
                            if (entry.is_dir() or entry.file_size < 1_000 or
                                entry.file_size > DOWNLOAD_LIMITS[entry.filename]):
                                raise ReleaseDataError(f"Invalid entry in {name}")
                        archive.extractall(stage)
                    archived.unlink()
                except ReleaseDataError:
                    raise
                except (HTTPError, URLError, TimeoutError, OSError,
                        zipfile.BadZipFile, RuntimeError) as exc:
                    raise ReleaseDataError(
                        f"Could not download or unpack {name}: {exc}"
                    ) from exc

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



def ensure_release_history(data_dir: Path) -> bool:
    """Create a minimal pre-cutoff learner-course display from a Release history.csv.

    This never publishes raw histories in the repository and never displays
    activity, identifiers other than pseudonymous learner IDs, or future starts.
    Raw Release asset visibility must be reviewed separately by the data owner.
    """
    data_dir = Path(data_dir)
    output = data_dir / "history_display.csv"
    if output.is_file():
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
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        raise ReleaseDataError(f"Unable to find course-history release: {exc}") from exc

    sources = {
        item["name"]: item["browser_download_url"]
        for item in release.get("assets", [])
    }
    source_url = sources.get("history.csv")
    expected_url = (
        f"https://github.com/{REPOSITORY}/releases/download/"
        f"{RELEASE_TAG}/history.csv"
    )
    if not source_url:
        raise ReleaseDataError("history.csv is not attached to the data release")
    if source_url != expected_url:
        raise ReleaseDataError("Unexpected course-history asset URL")

    # Only a minimal, pre-2013-08-01 view is retained on the server.
    # Never copy full history.csv into the public site's data directory.
    import pandas as pd

    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="learnsphere-history-", dir=data_dir
    ) as temp:
        stage = Path(temp)
        raw = stage / "history.csv"
        try:
            _download(source_url, raw, 60_000_000)
            history = pd.read_csv(
                raw, dtype={"userid_DI": "string", "course_id": "string"},
                low_memory=False
            )
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            raise ReleaseDataError(
                f"Could not download or read course history: {exc}"
            ) from exc

        required = {"userid_DI", "course_id", "start_time_DI"}
        if not required.issubset(history.columns):
            raise ReleaseDataError(
                "Course-history columns missing: "
                + ", ".join(sorted(required - set(history.columns)))
            )

        dates = pd.to_datetime(
            history["start_time_DI"], errors="coerce", utc=True,
        )
        before_cutoff = dates < pd.Timestamp("2013-08-01", tz="UTC")
        display = (
            history.loc[before_cutoff, ["userid_DI", "course_id"]]
            .dropna()
            .drop_duplicates()
            .copy()
        )
        valid = display["userid_DI"].str.fullmatch(
            r"Learner \d+", na=False
        )
        if not valid.all():
            raise ReleaseDataError(
                "Course history has non-pseudonymous identifiers; display denied"
            )
        if not display["course_id"].str.strip().ne("").all():
            raise ReleaseDataError("Course history has blank course IDs")
        display.to_csv(stage / "history_display.csv", index=False)
        os.replace(stage / "history_display.csv", output)
    return True
