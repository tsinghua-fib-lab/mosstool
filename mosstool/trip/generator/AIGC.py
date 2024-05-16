import geopandas as gpd
from generate_od import generator

__all__ = ["AigcGenerator"]


class AigcGenerator:
    """
    Generate OD matrix inside the given area.
    """

    def __init__(self):
        self.generator = generator.Generator()

    def set_satetoken(self, satetoken: str):
        """
        Set the satetoken for the generator.
        """
        self.generator.set_satetoken(satetoken)

    def load_area(
        self,
        area: gpd.GeoDataFrame,
    ):
        """
        Load the area data.

        Args:
        - area (gpd.GeoDataFrame): The area data.
        """
        self.generator.load_area(area)

    def generate(self):
        """
        Generate the OD matrix.
        """
        return self.generator.generate()
