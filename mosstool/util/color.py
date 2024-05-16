from typing import Union

__all__ = ["hex_to_rgba"]

def hex_to_rgba(hex: str, alpha: Union[float, int] = 255) -> list[int]:
    """
    Convert a hex color code to an rgba color code.

    Args:
    - hex: The hex color code to be converted.
    - alpha: The alpha value of the rgba color code.

    Returns:
    - The rgba color code [r, g, b, a].
    """
    hex = hex.lstrip("#")
    r, g, b = tuple(int(hex[i : i + 2], 16) for i in (0, 2, 4))
    return [r, g, b, int(alpha)]
