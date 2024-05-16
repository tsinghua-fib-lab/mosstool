import numpy as np
import geopandas as gpd
from mosstool.trip.generator import GravityGenerator



if __name__ == "__main__":

    # Initialize the gravity generator
    gravity_generator = GravityGenerator(
        Lambda=0.2,
        Alpha=0.5,
        Beta=0.5,
        Gamma=0.5
    )

    # Load the area data
    area = gpd.read_file("data/gravitygenerator/Beijing-shp/beijing.shp")
    print(type(area))

    pops = np.load("data/gravitygenerator/worldpop.npy")[:, 0]
    
    gravity_generator.load_area(area)

    # Generate the OD matrix
    od_matrix = gravity_generator.generate(pops)
    print(od_matrix)
    print(od_matrix.min(), od_matrix.max())
    print(od_matrix.sum())

    print(od_matrix.shape)