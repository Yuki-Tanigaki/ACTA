#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path

# 除外したいディレクトリ名
IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".vscode",
    ".idea",
    "dist",
    "build",
    ".egg-info",
}

# 除外したいファイル名
IGNORED_FILES = {
    ".DS_Store",
}


def print_tree(dir_path: Path, prefix: str = "") -> None:
    """dir_path 以下を、指定したプレフィクス付きでツリー表示する。"""

    children = []
    for p in dir_path.iterdir():
        # 隠しファイル・ディレクトリは基本除外
        if p.name.startswith(".") and p.name not in (".gitignore",):
            continue

        if p.is_dir() and p.name in IGNORED_DIRS:
            continue
        if p.is_file() and p.name in IGNORED_FILES:
            continue

        children.append(p)

    # ディレクトリ優先でソート
    children.sort(key=lambda p: (p.is_file(), p.name))

    for i, path in enumerate(children):
        connector = "└── " if i == len(children) - 1 else "├── "
        print(prefix + connector + path.name)

        if path.is_dir():
            extension = "    " if i == len(children) - 1 else "│   "
            print_tree(path, prefix + extension)


if __name__ == "__main__":
    # このスクリプトが scripts/ にある前提で、1つ上をプロジェクトルートとみなす
    root = Path(__file__).resolve().parent.parent
    print(f"Project tree for: {root}")
    print_tree(root)
