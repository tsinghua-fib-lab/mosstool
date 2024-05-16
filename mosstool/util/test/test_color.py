from ..color import hex_to_rgba


def test_hex_to_rgba():
    assert hex_to_rgba("#29A2FF", 0.5 * 255) == [41, 162, 255, 127]
    assert hex_to_rgba("#FFBE1A", 0.5 * 255) == [255, 190, 26, 127]
