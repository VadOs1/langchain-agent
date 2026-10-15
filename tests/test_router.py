import pytest

from app.graph import CONFIDENCE_THRESHOLD, route_after_review


def test_threshold_is_85():
    assert CONFIDENCE_THRESHOLD == 85


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [(0, "error"), (85, "error"), (86, "success"), (100, "success")],
)
def test_route_after_review(confidence, expected):
    assert route_after_review({"confidence": confidence}) == expected


def test_missing_confidence_routes_to_error():
    assert route_after_review({}) == "error"
