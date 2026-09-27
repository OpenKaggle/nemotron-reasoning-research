from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import requests
from kagglesdk.kaggle_http_client import KaggleHttpClient
from kagglesdk.kernels.types.kernels_api_service import (
    ApiListKernelSessionOutputRequest,
    ApiListKernelSessionOutputResponse,
)


def list_output_files(kernel: str, attempts: int, sleep_seconds: float):
    owner, slug = kernel.split("/", 1)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with KaggleHttpClient() as client:
                request = ApiListKernelSessionOutputRequest()
                request.user_name = owner
                request.kernel_slug = slug
                response = client.call(
                    "kernels.KernelsApiService",
                    "ListKernelSessionOutput",
                    request,
                    ApiListKernelSessionOutputResponse,
                ).files or []
                return response
        except Exception as exc:  # Kaggle API is currently flaky; retry all network failures.
            last_error = exc
            print(f"list output attempt {attempt}/{attempts} failed: {exc}", flush=True)
            if attempt < attempts:
                time.sleep(sleep_seconds)
    raise RuntimeError("Could not list kernel outputs") from last_error


def find_output_url(kernel: str, filename: str, attempts: int, sleep_seconds: float) -> str:
    for item in list_output_files(kernel, attempts, sleep_seconds):
        if item.file_name == filename:
            return item.url
    raise FileNotFoundError(f"{filename!r} not found in outputs for {kernel}")


def remote_size_from_headers(headers: requests.structures.CaseInsensitiveDict) -> int | None:
    content_range = headers.get("Content-Range")
    if content_range:
        match = re.search(r"/(\d+)$", content_range)
        if match:
            return int(match.group(1))
    content_length = headers.get("Content-Length")
    if content_length:
        return int(content_length)
    return None


def download_with_resume(
    kernel: str,
    filename: str,
    output: Path,
    attempts: int,
    sleep_seconds: float,
    chunk_size: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    url: str | None = None
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            if url is None:
                url = find_output_url(kernel, filename, attempts=3, sleep_seconds=sleep_seconds)

            existing = output.stat().st_size if output.exists() else 0
            headers = {"Range": f"bytes={existing}-"} if existing else {}
            mode = "ab" if existing else "wb"
            print(
                f"download attempt {attempt}/{attempts}: "
                f"starting at byte {existing:,}",
                flush=True,
            )
            with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
                if response.status_code in (401, 403, 404):
                    # Signed URL may have expired; reacquire on the next attempt.
                    url = None
                    raise RuntimeError(
                        f"signed URL returned HTTP {response.status_code}"
                    )
                response.raise_for_status()

                if existing and response.status_code == 200:
                    # Server ignored Range; restart rather than append duplicate bytes.
                    mode = "wb"
                    existing = 0

                total_size = remote_size_from_headers(response.headers)
                bytes_done = existing
                next_report = bytes_done + 256 * 1024 * 1024
                with open(output, mode + ("" if "b" in mode else "b")) as handle:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        bytes_done += len(chunk)
                        if bytes_done >= next_report:
                            if total_size:
                                print(
                                    f"  downloaded {bytes_done:,}/{total_size:,} bytes",
                                    flush=True,
                                )
                            else:
                                print(f"  downloaded {bytes_done:,} bytes", flush=True)
                            next_report = bytes_done + 256 * 1024 * 1024

            final_size = output.stat().st_size
            if total_size is None or final_size >= total_size:
                print(f"download complete: {output} ({final_size:,} bytes)", flush=True)
                return
            raise RuntimeError(
                f"incomplete download: {final_size:,} of {total_size:,} bytes"
            )
        except Exception as exc:
            last_error = exc
            print(f"download attempt {attempt}/{attempts} failed: {exc}", flush=True)
            if attempt < attempts:
                time.sleep(sleep_seconds)

    raise RuntimeError(f"Could not download {filename}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--attempts", type=int, default=20)
    parser.add_argument("--sleep-seconds", type=float, default=20.0)
    parser.add_argument("--chunk-size", type=int, default=8 * 1024 * 1024)
    args = parser.parse_args()

    download_with_resume(
        kernel=args.kernel,
        filename=args.filename,
        output=args.output,
        attempts=args.attempts,
        sleep_seconds=args.sleep_seconds,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
