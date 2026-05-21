from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

CLIENT_FILES = (
    "package-lock.json",
    "package.json",
    "src/index.ts",
    "tsconfig.json",
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLIENT_DIR = ROOT / "client"
ASSET_DIR = pathlib.Path(__file__).resolve().parent / "client"
ASSETS = (
    ASSET_DIR / "body-models-viser.js",
    ASSET_DIR / "body-models-viser.wasm",
)


def ensure_client_is_built() -> None:
    inputs = [CLIENT_DIR / name for name in CLIENT_FILES]
    inputs.extend((ROOT / "Cargo.lock", ROOT / "Cargo.toml"))
    inputs.extend((ROOT / "src").glob("*.rs"))

    if not CLIENT_DIR.exists():
        for asset in ASSETS:
            asset.stat()
        return

    if all(path.exists() for path in ASSETS) and max(path.stat().st_mtime for path in inputs) < min(
        path.stat().st_mtime for path in ASSETS
    ):
        return

    node_bin_dir = _install_sandboxed_node()
    _build_client(node_bin_dir)


def _install_sandboxed_node() -> pathlib.Path:
    env_dir = CLIENT_DIR / ".nodeenv"
    node_bin_dir = env_dir / ("Scripts" if sys.platform == "win32" else "bin")
    npm_path = node_bin_dir / "npm"
    if sys.platform == "win32":
        npm_path = npm_path.with_suffix(".cmd")
    if npm_path.exists():
        return node_bin_dir
    if env_dir.exists():
        shutil.rmtree(env_dir)

    subprocess.run([sys.executable, "-m", "nodeenv", "--node=24.12.0", str(env_dir)], check=True)
    return node_bin_dir


def _build_client(node_bin_dir: pathlib.Path) -> None:
    npm_path = node_bin_dir / "npm"
    if sys.platform == "win32":
        npm_path = npm_path.with_suffix(".cmd")

    env = os.environ.copy()
    env["NODE_VIRTUAL_ENV"] = str(node_bin_dir.parent)
    env["PATH"] = str(node_bin_dir) + os.pathsep + env["PATH"]

    subprocess.run([str(npm_path), "ci"], cwd=CLIENT_DIR, env=env, check=True)
    subprocess.run([str(npm_path), "run", "build"], cwd=CLIENT_DIR, env=env, check=True)
