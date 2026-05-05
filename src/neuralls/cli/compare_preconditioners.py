#!/usr/bin/env python3
"""Run every comparison profile declared in one case config."""

from __future__ import annotations

import os
from pathlib import Path

import typer
from loguru import logger

from neuralls.shared.constants import (
    EXIT_FAILURE,
    EXIT_KEYBOARD_INTERRUPT,
    SYMBOL_CHECKMARK,
)
from neuralls.composition.experiments.comparison_batch import run_comparison_batch
from neuralls.platform.config.settings import load_case_settings
from neuralls.composition.comparison.models import (
    ComparisonOutcome,
    ComparisonParams,
    ComparisonResult,
)

os.environ.setdefault("MPLBACKEND", "Agg")


def _log_comparison_results(result: ComparisonResult) -> None:
    preconditioners = result.preconditioners
    summary = result.summary

    logger.info(f"Available preconditioners: {preconditioners}")
    if summary:
        logger.info("=" * 60)
        logger.info(summary)
        logger.info("=" * 60)


def _log_outcomes(outcomes: list[ComparisonOutcome]) -> None:
    successful = [o for o in outcomes if o.success]
    failed = [o for o in outcomes if not o.success]
    for outcome in outcomes:
        label = outcome.comparison_display_name
        logger.info("=" * 80)
        logger.info(f"Comparison: {label}")
        logger.info("=" * 80)
        for warning in outcome.warnings:
            logger.warning(warning)
        if outcome.success and outcome.payload:
            _log_comparison_results(outcome.payload)
        else:
            logger.error(f"Comparison failed: {outcome.error}")

    logger.info("")
    logger.info("=" * 80)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total comparisons: {len(outcomes)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    if failed:
        for item in failed:
            logger.error(f"  ✗ {item.comparison_id}: {item.error}")
        raise typer.Exit(code=EXIT_FAILURE)
    logger.info(f"\n{SYMBOL_CHECKMARK} All comparisons completed successfully!")


def main(
    config: Path = typer.Argument(
        ...,
        help="Path to a case config TOML.",
    ),
    env_file: Path | None = typer.Option(
        None,
        help="Optional env file to load before config resolution.",
    ),
) -> None:
    """Benchmark classical and neural preconditioners for one case config."""
    params = ComparisonParams()

    try:
        settings = load_case_settings(config, env_file)
        outcomes = run_comparison_batch(config, params, settings)
    except (ValueError, KeyError) as exc:
        logger.error(str(exc))
        raise typer.Exit(code=EXIT_FAILURE) from exc

    _log_outcomes(outcomes)


def run() -> None:
    """Entry point for pyproject.toml script registration."""
    typer.run(main)


if __name__ == "__main__":
    try:
        typer.run(main)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        raise SystemExit(EXIT_KEYBOARD_INTERRUPT)
