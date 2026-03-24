"""Download and unpack ATCF aid_public data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import gzip
import os
from pathlib import Path
import shutil
import tempfile

import requests

AID_PUBLIC_URL = "https://ftp.nhc.noaa.gov/atcf/aid_public/"
CHUNK_SIZE = 64 * 1024


def get_file_links(session: requests.Session) -> list[str]:
    """Parse the directory listing and return .dat.gz filenames."""
    from bs4 import BeautifulSoup

    response = session.get(AID_PUBLIC_URL, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    return [link["href"] for link in soup.find_all("a", href=True) if link["href"].endswith(".dat.gz")]


def get_file_modtime(session: requests.Session, filename: str) -> datetime | None:
    """Read Last-Modified from remote headers when available."""
    response = session.head(AID_PUBLIC_URL + filename, allow_redirects=True, timeout=20)
    if "Last-Modified" not in response.headers:
        return None

    try:
        dt = parsedate_to_datetime(response.headers["Last-Modified"])
    except Exception:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def download_file(
    session: requests.Session,
    *,
    url: str,
    dest_path: Path,
    chunk_size: int = CHUNK_SIZE,
) -> None:
    """Stream a download into a temp file and atomically replace."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_handle = None
    try:
        with session.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(dir=dest_path.parent, delete=False) as tmp_handle:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        tmp_handle.write(chunk)
                tmp_handle.flush()

        os.replace(tmp_handle.name, dest_path)
        tmp_handle = None
    finally:
        if tmp_handle is not None and os.path.exists(tmp_handle.name):
            try:
                os.remove(tmp_handle.name)
            except Exception:
                pass


def decompress_gz(*, src_path: Path, dest_path: Path, buffer_size: int = CHUNK_SIZE) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(src_path, "rb") as src, open(dest_path, "wb") as dest:
        shutil.copyfileobj(src, dest, length=buffer_size)


def sync_aid_public(
    *,
    output_dir: Path | str,
    download_dir: Path | str = ".",
    min_age_hours: float = 0.0,
) -> int:
    output_path = Path(output_dir).expanduser().resolve()
    download_path = Path(download_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    download_path.mkdir(parents=True, exist_ok=True)

    now = datetime.now(tz=timezone.utc)

    with requests.Session() as session:
        files = get_file_links(session)
        print(f"Found {len(files)} .dat.gz files")

        for filename in files:
            if min_age_hours > 0:
                modtime = get_file_modtime(session, filename)
                if modtime is not None:
                    age_hours = (now - modtime).total_seconds() / 3600.0
                    if age_hours < min_age_hours:
                        print(f"Skipping {filename}: age {age_hours:.2f}h < {min_age_hours:.2f}h")
                        continue

            download_target = download_path / filename
            decompressed_name = filename[:-3]
            decompressed_target = output_path / decompressed_name

            print(f"Downloading {filename}")
            download_file(session, url=AID_PUBLIC_URL + filename, dest_path=download_target)

            print(f"Decompressing {filename}")
            decompress_gz(src_path=download_target, dest_path=decompressed_target)

            try:
                download_target.unlink()
            except FileNotFoundError:
                pass

            print(f"Wrote {decompressed_target}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and unpack ATCF aid_public data")
    parser.add_argument("--output-dir", default="/data/gempak/atcf/", help="Directory for decompressed .dat files")
    parser.add_argument("--download-dir", default=".", help="Temporary directory for .dat.gz downloads")
    parser.add_argument("--min-age-hours", type=float, default=0.0, help="Skip files newer than this age")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return sync_aid_public(
        output_dir=args.output_dir,
        download_dir=args.download_dir,
        min_age_hours=args.min_age_hours,
    )


__all__ = [
    "AID_PUBLIC_URL",
    "build_parser",
    "decompress_gz",
    "download_file",
    "get_file_links",
    "get_file_modtime",
    "main",
    "sync_aid_public",
]


if __name__ == "__main__":
    raise SystemExit(main())
