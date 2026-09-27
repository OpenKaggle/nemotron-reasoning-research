from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path


COMPETITION = "nvidia-nemotron-model-reasoning-challenge"


def retry(label: str, attempts: int, sleep_seconds: float, func):
    last_exc: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            print(f"[{label}] attempt {attempt}/{attempts}", flush=True)
            return func()
        except Exception as exc:  # Kaggle API raises several requests/http wrappers.
            last_exc = exc
            print(f"[{label}] failed: {exc}", flush=True)
            if attempt < attempts:
                time.sleep(sleep_seconds)
    raise RuntimeError(f"{label} failed after {attempts} attempts") from last_exc


def load_api():
    try:
        kaggle = importlib.import_module("kaggle")
        return kaggle.api
    except Exception:
        for name in list(sys.modules):
            if name == "kaggle" or name.startswith("kaggle.") or name.startswith("kagglesdk."):
                sys.modules.pop(name, None)
        raise


def create_dataset(path: str, public: bool) -> None:
    folder = str(Path(path).resolve())

    def _create():
        api = load_api()
        result = api.dataset_create_new(
            folder=folder,
            public=public,
            quiet=False,
            convert_to_csv=True,
            dir_mode="zip",
        )
        print(f"[dataset] status={result.status} url={result.url} error={result.error}")
        return result

    retry("dataset-create", 8, 8, _create)


def status(kernel: str) -> None:
    def _status():
        api = load_api()
        result = api.kernels_status(kernel)
        print(f"[status] {kernel}: {result.status}")
        if result.failure_message:
            print(f"[status] failure: {result.failure_message}")
        return result

    retry("kernel-status", 8, 8, _status)


def submit_kernel(kernel: str, version: int, file_name: str, message: str) -> None:
    def _submit():
        api = load_api()
        result = api.competition_submit_code(
            file_name=file_name,
            message=message,
            competition=COMPETITION,
            kernel=kernel,
            kernel_version=version,
            quiet=False,
        )
        print(f"[submit] {result.message}")
        return result

    retry("kernel-submit", 8, 8, _submit)


def push_kernel(path: str) -> None:
    folder = str(Path(path).resolve())

    def _push():
        api = load_api()
        result = api.kernels_push(folder)
        if result.error:
            raise RuntimeError(result.error)
        print(f"[push] version={result.versionNumber} url={result.url}")
        return result

    retry("kernel-push", 8, 8, _push)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    create_parser = subparsers.add_parser("create-dataset")
    create_parser.add_argument("path")
    create_parser.add_argument("--public", action="store_true")

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("kernel")

    submit_parser = subparsers.add_parser("submit-kernel")
    submit_parser.add_argument("kernel")
    submit_parser.add_argument("version", type=int)
    submit_parser.add_argument("--file", default="submission.zip")
    submit_parser.add_argument("--message", required=True)

    push_parser = subparsers.add_parser("push-kernel")
    push_parser.add_argument("path")

    args = parser.parse_args()
    if args.cmd == "create-dataset":
        create_dataset(args.path, args.public)
    elif args.cmd == "status":
        status(args.kernel)
    elif args.cmd == "submit-kernel":
        submit_kernel(args.kernel, args.version, args.file, args.message)
    elif args.cmd == "push-kernel":
        push_kernel(args.path)


if __name__ == "__main__":
    main()
