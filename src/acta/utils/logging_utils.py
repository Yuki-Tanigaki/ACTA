# src/acta/utils/logging_utils.py
from __future__ import annotations

from pathlib import Path
import logging
from typing import Optional


def setup_logging(
    output_dir: Path,
    run_name: str,
    *,
    enable_file: bool = True,
    log_level: int = logging.INFO,
    console_level: Optional[int] = None,
    file_level: Optional[int] = None,
    logger_name: str | None = None,
) -> logging.Logger:
    """
    ログ設定をまとめて行うユーティリティ。

    Parameters
    ----------
    output_dir : Path
        出力ディレクトリ (例: cfg.output_dir)
    run_name : str
        ログファイル名のベース (例: "toy_001_seed0000")
    enable_file : bool, default True
        True のときファイルにも出力する
    log_level : int, default logging.INFO
        全体の基本ログレベル
    console_level : int | None
        標準出力ハンドラのログレベル (None の場合 log_level と同じ)
    file_level : int | None
        ファイルハンドラのログレベル (None の場合 log_level と同じ)
    logger_name : str | None
        設定対象のロガー名。None の場合 root logger を設定。

    Returns
    -------
    logging.Logger
        設定されたロガー
    """
    logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
    logger.setLevel(log_level)

    # 既存ハンドラを全部外す（重複出力防止）
    logger.handlers.clear()

    # フォーマット
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )

    # --- コンソール出力 ---
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(console_level if console_level is not None else log_level)
    logger.addHandler(ch)

    # --- ファイル出力 ---
    if enable_file:
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        log_file = logs_dir / f"{run_name}.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        fh.setLevel(file_level if file_level is not None else log_level)
        logger.addHandler(fh)

        logger.info("Logging to file: %s", log_file)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """
    モジュール側から呼ぶ用のヘルパー。
    __name__ を渡して使う想定。
    """
    return logging.getLogger(name)
