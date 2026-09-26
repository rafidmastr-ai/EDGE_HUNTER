"""Chronological train/validation/OOS and walk-forward split construction."""

from __future__ import annotations

from app.optimization.models import SplitPlan, WalkForwardFold


def make_split_plan(
    total_bars: int,
    *,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> SplitPlan:
    train_end = max(1, int(total_bars * train_fraction))
    validation_end = max(train_end + 1, int(total_bars * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, total_bars - 1)
    if train_end >= validation_end:
        raise ValueError("insufficient bars for requested train/validation/OOS fractions")
    return SplitPlan(train_end, validation_end, total_bars)


def make_walk_forward_folds(
    total_bars: int,
    *,
    folds: int = 3,
    min_train_bars: int | None = None,
) -> tuple[WalkForwardFold, ...]:
    if total_bars < 6:
        return ()
    if folds < 1:
        raise ValueError("folds must be >= 1")
    folds = min(folds, total_bars - 2)
    test_size = max(1, total_bars // (folds + 2))
    if min_train_bars is None:
        min_train_bars = max(2, total_bars // 3)

    result: list[WalkForwardFold] = []
    for fold_id in range(folds):
        train_end = min_train_bars + fold_id * test_size
        test_start = train_end
        test_end = test_start + test_size
        if test_end > total_bars:
            break
        result.append(
            WalkForwardFold(
                fold_id=fold_id + 1,
                train_start=0,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
    return tuple(result)


__all__ = ["make_split_plan", "make_walk_forward_folds"]
