"""PEP 517 backend that builds Hygon aiter from the vendor sources.

The vendor publishes aiter only as a ``cp310`` wheel whose JIT extensions are
compiled against one exact torch build, so that wheel cannot serve a newer
interpreter. This backend clones the Hygon downstream repository, builds its
python package for the running interpreter, and republishes it under its own
distribution name: aiter compiles the kernels with its own JIT configuration
when the target environment first uses them.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

NAME = "aiter-dtk"
VERSION = "0.1.8"
REPOSITORY = "https://github.com/HYGON-AI/aiter-das"
REVISION = "5ff4e86fc46629c6d389f564d6b44fe813c9cba6"
REQUIRES = ("einops", "ninja==1.11.1", "pandas<3", "psutil", "pybind11>=3.0.1")
CHECKOUT = pathlib.Path("~/.cache/flaggems/src/aiter-das").expanduser()


def _log(message: str) -> None:
    print(f"aiter: {message}", file=sys.stderr, flush=True)


def _source_tree() -> pathlib.Path:
    """Reuse the checkout; the upstream flow builds in a fresh image anyway."""
    if (CHECKOUT / "setup.py").exists():
        _log(f"reusing {CHECKOUT}; delete it if that cache is dirty")
        return CHECKOUT
    CHECKOUT.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 9):
        _log(f"cloning {REPOSITORY} at {REVISION} (attempt {attempt}/8)")
        shutil.rmtree(CHECKOUT, ignore_errors=True)
        clone = subprocess.run(["git", "clone", "--quiet", REPOSITORY, str(CHECKOUT)])
        if clone.returncode == 0:
            checkout = subprocess.run(
                ["git", "checkout", "--quiet", REVISION], cwd=CHECKOUT
            )
            submodules = subprocess.run(
                ["git", "submodule", "update", "--init", "--recursive"],
                cwd=CHECKOUT,
            )
            if checkout.returncode == 0 and submodules.returncode == 0:
                return CHECKOUT
        if attempt == 8:
            break
        time.sleep(10 * attempt)
    shutil.rmtree(CHECKOUT, ignore_errors=True)
    raise RuntimeError(f"could not clone {REPOSITORY} at {REVISION}")


def _vendor_wheel(source: pathlib.Path) -> pathlib.Path:
    """Build the wheel the way the vendor's rebuild_aiter.sh does."""
    output = pathlib.Path(tempfile.mkdtemp(prefix="aiter-wheel-"))
    for stale in (
        source / "aiter_meta",
        source / "build",
        source / "aiter.egg-info",
        source / "aiter" / "jit" / "build",
        source / "aiter" / "install_mode",
        source / "aiter" / "jit" / "aiter_.so",
    ):
        if stale.is_dir():
            shutil.rmtree(stale, ignore_errors=True)
        elif stale.exists():
            stale.unlink()
    for module in (source / "aiter" / "jit").glob("*.so"):
        module.unlink()
    env = dict(os.environ)
    env["AITER_BUILD_MODE"] = "wheel"
    env["AITER_USE_SCM_VERSION"] = "0"
    env["AITER_FIXED_VERSION"] = VERSION
    env["PREBUILD_KERNELS"] = "0"
    subprocess.run(
        [
            sys.executable,
            "-u",
            "setup.py",
            "bdist_wheel",
            "--dist-dir",
            str(output),
        ],
        cwd=source,
        check=True,
        env=env,
    )
    wheels = sorted(output.glob("*.whl"))
    if not wheels:
        raise RuntimeError(f"no wheel produced in {output}")
    return wheels[0]


def _metadata() -> str:
    lines = [
        "Metadata-Version: 2.1",
        f"Name: {NAME}",
        f"Version: {VERSION}",
        "Summary: Hygon aiter built from the vendor sources",
        "Requires-Python: >=3.10",
    ]
    lines.extend(f"Requires-Dist: {requirement}" for requirement in REQUIRES)
    return "\n".join(lines) + "\n"


def _wheel(tags: list[str]) -> str:
    lines = [
        "Wheel-Version: 1.0",
        "Generator: aiter_wrapper",
        "Root-Is-Purelib: false",
    ]
    lines.extend(f"Tag: {tag}" for tag in tags)
    return "\n".join(lines) + "\n"


def _dist_info() -> str:
    return f"{NAME.replace('-', '_')}-{VERSION}.dist-info"


def _record_row(path: str, data: bytes) -> list[str]:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return [path, f"sha256={digest.decode()}", str(len(data))]


def _vendor_tags(vendor: pathlib.Path) -> list[str]:
    with zipfile.ZipFile(vendor) as archive:
        wheel_file = [
            name for name in archive.namelist() if name.endswith(".dist-info/WHEEL")
        ]
        text = archive.read(wheel_file[0]).decode("utf-8")
    return [
        line.split(": ", 1)[1].strip()
        for line in text.splitlines()
        if line.startswith("Tag: ")
    ]


def get_requires_for_build_wheel(config_settings=None):
    return []


def get_requires_for_build_editable(config_settings=None):
    return []


def get_requires_for_build_sdist(config_settings=None):
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    _log("hook: prepare_metadata_for_build_wheel")
    directory = pathlib.Path(metadata_directory) / _dist_info()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "METADATA").write_text(_metadata(), encoding="utf-8")
    (directory / "WHEEL").write_text(_wheel(["py3-none-any"]), encoding="utf-8")
    return _dist_info()


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _log("hook: build_wheel")
    vendor = _vendor_wheel(_source_tree())
    tags = _vendor_tags(vendor)
    target = (
        pathlib.Path(wheel_directory)
        / f"{NAME.replace('-', '_')}-{VERSION}-{tags[0]}.whl"
    )
    dist_info = _dist_info()
    record = io.StringIO()
    writer = csv.writer(record, lineterminator="\n")
    with (
        zipfile.ZipFile(vendor) as source,
        zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as out,
    ):
        for member in source.infolist():
            if ".dist-info/" in member.filename:
                continue
            data = source.read(member)
            out.writestr(member, data)
            writer.writerow(_record_row(member.filename, data))
        for name, text in (
            ("METADATA", _metadata()),
            ("WHEEL", _wheel(tags)),
            ("INSTALLER", "uv\n"),
        ):
            data = text.encode("utf-8")
            out.writestr(f"{dist_info}/{name}", data)
            writer.writerow(_record_row(f"{dist_info}/{name}", data))
        writer.writerow([f"{dist_info}/RECORD", "", ""])
        out.writestr(f"{dist_info}/RECORD", record.getvalue())
    return target.name


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    raise NotImplementedError("aiter_wrapper only distributes a wheel")


def build_sdist(sdist_directory, config_settings=None):
    raise NotImplementedError("aiter_wrapper only distributes a wheel")
