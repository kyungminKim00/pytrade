import inc_dec  # The code to test


def test_increment() -> None:
    """_summary_"""
    assert inc_dec.increment(3) == 4


def test_decrement() -> None:
    """_summary_"""
    assert inc_dec.decrement(3) == 2
