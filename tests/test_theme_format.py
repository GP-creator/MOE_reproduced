"""dashboard.theme SI formatter: fixed-point, never scientific notation."""

import pytest

theme = pytest.importorskip("dashboard.theme")


@pytest.mark.parametrize("x,digits,want", [
    (910_000, 2, "910k"), (9_100, 2, "9.1k"), (999_999, 3, "1M"), (12_345_678, 3, "12.3M"),
    (0.5, 3, "0.5"), (123, 3, "123"), (-2.5e9, 3, "-2.5G"), (0, 3, "0"),
])
def test_si(x, digits, want):
    assert theme._si(x, digits) == want


def test_fmt_bytes_no_exponent():
    assert theme.fmt_bytes(912_000) == "910kB"
    assert "e+" not in theme.fmt_bytes(123_456)


def test_fmt_bytes_digits():
    assert theme.fmt_bytes(912_000, digits=3) == "912kB"
    assert theme.fmt_bytes(75_049_000, digits=3) == "75MB"
    assert theme.fmt_bytes(float("nan"), digits=3) == theme.DASH
