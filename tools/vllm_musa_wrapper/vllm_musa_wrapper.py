"""PEP 517 backend that builds the Moore Threads vLLM plugin in place.

The plugin has no wheel of its own and its build needs the MUSA torch stack, so
this backend keeps the distribution metadata static, clones the pinned vendor
source on demand, runs the vendor requirement passes, builds the plugin against
the torch already installed in the environment, and repacks it under its own
distribution name.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib
import io
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

NAME = "vllm-musa"
VERSION = "0.1.24"
REPOSITORY = "https://github.com/MooreThreads/vllm-musa"
REVISION = "9992297d59ebca36132ed6c59a2d3fb736f7a410"
REQUIRES = (
    "click==8.2.0",
    "einops",
    "mthreads-ml-py>=2.4.1",
    "openai>=2.24.0",
    "ray<=2.48.0,>=2.47.1",
    "torchada>=0.1.76",
)
CHECKOUT = pathlib.Path("~/.cache/flaggems/src/vllm-musa").expanduser()
# vllm-musa imports these while it generates its metadata. ``tvm_ffi`` is the
# Moore Threads build of apache-tvm-ffi, published only on the vendor index, so
# the index is passed to that single install instead of being declared globally
# where it would capture the package for every other vendor.
VENDOR_INDEX = "https://dl.mthreads.com/repo/api/pypi/pypi/simple"
SETUPTOOLS_VERSION = "80.10.2"

BUILD_PREREQS = {
    "torchada": ("torchada>=0.1.76", None),
    "tvm_ffi": ("apache-tvm-ffi==0.1.9.post3+musa.1", VENDOR_INDEX),
}


def _log(message: str) -> None:
    print(f"vllm_musa: {message}", file=sys.stderr, flush=True)


def _run(command: list[str], env: dict | None = None) -> None:
    for attempt in range(1, 9):
        result = subprocess.run(command, env=env)
        if result.returncode == 0:
            return
        if attempt == 8:
            raise RuntimeError(f"command failed after 8 attempts: {command}")
        _log(f"command failed (attempt {attempt}/8), retrying in {10 * attempt}s")
        time.sleep(10 * attempt)


def _git_output(*arguments: str) -> str:
    if not (CHECKOUT / ".git").exists():
        return ""
    result = subprocess.run(
        ["git", *arguments], cwd=CHECKOUT, capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _source_tree() -> pathlib.Path:
    """Reuse the checkout only while it is the pinned revision and untouched."""
    if CHECKOUT.exists():
        head = _git_output("rev-parse", "HEAD")
        status = _git_output(
            "status", "--porcelain", "--untracked-files=no", "--ignore-submodules=all"
        )
        if head == REVISION and not status:
            _log(f"reusing {CHECKOUT}")
            return CHECKOUT
        _log(
            f"removing {CHECKOUT}: head={head or '<none>'} status={status.splitlines()[:3]}"
        )
        shutil.rmtree(CHECKOUT, ignore_errors=True)
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


def _torchada_exclude_dirs(env: dict) -> None:
    """Tell torchada to skip header porting for the vendor torch package.

    The package path comes from ``find_spec``: importing torch_musa on a vendor
    image autoloads torch backend extensions that may be unusable there.
    """
    probe = (
        "import importlib.util, pathlib\n"
        "spec = importlib.util.find_spec('torch_musa')\n"
        "print(pathlib.Path(spec.origin).parent if spec and spec.origin else '')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env=env
    )
    package = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    if result.returncode != 0 or not package:
        tail = result.stderr.strip().splitlines()
        raise RuntimeError(
            f"cannot locate torch_musa: {tail[-1] if tail else 'not importable'}"
        )
    env["TORCHADA_EXCLUDE_DIRS"] = os.pathsep.join([package, "torch_musa"])
    _log(f"torchada exempts {package} from header porting")


def _cpp_toolchain() -> tuple[str, list[str], list[str]]:
    """Ask a C++ driver where its standard headers and libraries live."""
    for name in ("g++", "g++-14", "g++-13", "g++-12", "c++"):
        try:
            probe = subprocess.run(
                [name, "-E", "-x", "c++", "-", "-v"],
                input="",
                capture_output=True,
                text=True,
            )
        except OSError:
            continue
        if probe.returncode != 0:
            continue
        include, inside = [], False
        for line in probe.stderr.splitlines():
            if line.startswith("#include <...> search starts here:"):
                inside = True
            elif line.startswith("End of search list."):
                break
            elif inside:
                path = line.strip()
                if path.startswith("/") and os.path.isdir(path):
                    include.append(path)
        if not include:
            continue
        libraries = []
        search = subprocess.run(
            [name, "-print-search-dirs"], capture_output=True, text=True
        )
        for line in search.stdout.splitlines():
            if line.startswith("libraries: ="):
                libraries = [path for path in line.split("=", 1)[1].split(":") if path]
        return name, include, libraries
    fallback = sorted(pathlib.Path("/usr/include/c++").glob("*"))
    return "", [str(path) for path in fallback if path.is_dir()], []


def _cxx_toolchain_env(env: dict) -> None:
    """Hand clang the libstdc++ search paths it cannot discover by itself."""
    driver, include, libraries = _cpp_toolchain()
    if not include:
        _log(
            "no C++ standard headers found; install g++ and libstdc++ headers "
            "so a source build can include <cmath>"
        )
        return
    _log(f"C++ headers from {driver or 'fallback'}: {include[0]} (+{len(include) - 1})")
    env["CPLUS_INCLUDE_PATH"] = ":".join([*include, env.get("CPLUS_INCLUDE_PATH", "")])
    env["CPLUS_INCLUDE_PATH"] = env["CPLUS_INCLUDE_PATH"].strip(":")
    if libraries:
        env["LIBRARY_PATH"] = ":".join([*libraries, env.get("LIBRARY_PATH", "")]).strip(
            ":"
        )


def _ensure_build_prereqs() -> None:
    """Install what the vendored build imports; probe without importing it."""
    for module, (requirement, index) in BUILD_PREREQS.items():
        if importlib.util.find_spec(module) is not None:
            continue
        command = [sys.executable, "-m", "pip", "install", "--quiet"]
        if index is not None:
            command += ["--index-url", index]
        _log(f"installing {requirement}")
        _run([*command, requirement])


def _ensure_setuptools() -> None:
    """Pin the setuptools build used by the vendored extension build."""
    import importlib.metadata

    try:
        current = importlib.metadata.version("setuptools")
    except importlib.metadata.PackageNotFoundError:
        current = ""
    if current == SETUPTOOLS_VERSION:
        return
    _log(f"installing setuptools=={SETUPTOOLS_VERSION} (found {current or '<none>'})")
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            f"setuptools=={SETUPTOOLS_VERSION}",
        ]
    )


def _vendor_wheel() -> pathlib.Path:
    _ensure_setuptools()
    _ensure_build_prereqs()
    source = _source_tree()
    output = pathlib.Path(tempfile.mkdtemp(prefix="vllm-musa-wheel-"))
    env = dict(os.environ)
    # The vendored setup.py reads torch.musa, which only exists once the image's
    # torch autoloads torch_musa, so that switch is left alone.
    _cxx_toolchain_env(env)
    _torchada_exclude_dirs(env)
    _log(f"building {source}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-build-isolation",
            "--no-deps",
            str(source),
            "-w",
            str(output),
        ],
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
        "Summary: Moore Threads vLLM platform plugin",
        "Requires-Python: >=3.10",
    ]
    lines.extend(f"Requires-Dist: {requirement}" for requirement in REQUIRES)
    return "\n".join(lines) + "\n"


def _wheel(tags: list[str]) -> str:
    lines = [
        "Wheel-Version: 1.0",
        "Generator: vllm_musa_wrapper",
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
    vendor = _vendor_wheel()
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
    raise NotImplementedError("vllm_musa_wrapper only distributes a wheel")


def build_sdist(sdist_directory, config_settings=None):
    raise NotImplementedError("vllm_musa_wrapper only distributes a wheel")
