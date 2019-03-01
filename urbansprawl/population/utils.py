###############
# Repository: https://github.com/lgervasoni/urbansprawl
# MIT License
###############

import geopandas as gpd
import pandas as pd
import osmnx as ox
import numpy as np

from shapely.geometry import Polygon
from shapely.geometry import Point

from ..settings import storage_folder

# Format for load/save the geo-data ['geojson','shp']
geo_format = "geojson"  # 'shp'


def get_population_extract_filename(city_ref_file, data_source):
    """
        Get data population extract filename for input city

        Parameters
        ----------
        city_ref_file : string
                name of input city
        data_source : string
                desired population data source

        Returns
        ----------
        string
                returns the population extract filename

        """
    # Folder exists?
    import os

    if not (os.path.isdir(storage_folder + "/" + data_source)):
        os.makedirs(storage_folder + "/" + data_source)
    return (
        storage_folder
        + "/"
        + data_source
        + "/"
        + city_ref_file
        + "_population.shp"
    )


def get_population_urban_features_filename(city_ref_file, data_source):
    """
        Get population urban features extract filename for input city
        Force GeoJSON format: Shapefiles truncate column names

        Parameters
        ----------
        city_ref_file : string
                name of input city
        data_source : string
                desired population data source

        Returns
        ----------
        string
                returns the population extract filename

        """
    # Folder exists?
    import os

    if not (os.path.isdir(storage_folder + "/" + data_source)):
        os.makedirs(storage_folder + "/" + data_source)
    return (
        storage_folder
        + "/"
        + data_source
        + "/"
        + city_ref_file
        + "_urban_features."
        + geo_format
    )


def get_population_training_validating_filename(
    city_ref_file, data_source="training"
):
    """
        Get population normalized urban features extract and population densities filename for input city
        Stored in Numpy.Arrays

        Parameters
        ----------
        city_ref_file : string
                name of input city

        Returns
        ----------
        string
                returns the numpy stored/storing filename

        """
    # Folder exists?
    import os

    if not (os.path.isdir(storage_folder + "/" + data_source)):
        os.makedirs(storage_folder + "/" + data_source)
    return (
        storage_folder + "/" + data_source + "/" + city_ref_file + "_X_Y.npz"
    )


#################################################################


def get_aggregated_squares(df_insee, step=1000.0, conserve_squares_info=False):
    """Aggregates input population data in squares of 5x5

        Assumption: Input squares 200m by 200m

        INSEE data contains column 'idINSPIRE' which denotes in EPSG:3035 the
        northing/easting coordinates of the south-west box endpoint

    If conserve squares information is True, the information relative to each
        original square is kept
        Output: Aggregated squares of 1km by 1km

        Parameters
        ----------
        df_insee : geopandas.GeoDataFrame
                INSEE population data
        step : float
                sampling step (default of 1 kilometer)
        conserve_squares_info : bool
                determines if each aggregated square conserves the information of each smaller composing square

        Returns
        ----------
        geopandas.GeoDataFrame
                returns the aggregated population data
        """

    def get_northing_easting(x):  # Extract northing and easting coordinates
        try:
            north, east = x.idINSPIRE.split("N")[1].split("E")
            x["north"] = int(north)
            x["east"] = int(east)
        except (IndexError, ValueError):
            x["north"], x["east"] = np.nan, np.nan
        return x

    def index_square(x, df_insee, offset_index):
        squares = df_insee.cx[
            x.geometry.x - offset_index: x.geometry.x + offset_index,
            x.geometry.y - offset_index: x.geometry.y + offset_index,
        ]
        aggregated_polygon = Polygon()
        for geom in squares.geometry:
            aggregated_polygon = aggregated_polygon.union(geom)
        x["square_geometry"] = aggregated_polygon
        x["pop_count"] = squares.pop_count.sum()
        return x

    def index_square_conservative(x, df_insee, offset_index):
        squares = df_insee.cx[
            x.geometry.x - offset_index: x.geometry.x + offset_index,
            x.geometry.y - offset_index: x.geometry.y + offset_index,
        ]
        aggregated_polygon = Polygon()
        for geom in squares.geometry:
            aggregated_polygon = aggregated_polygon.union(geom)
        x["square_geometry"] = aggregated_polygon
        x["pop_count"] = squares.pop_count.sum()

        indices = []
        for y_diff in [-400, -200, 0, 200, 400]:
            for x_diff in [
                -400,
                -200,
                0,
                200,
                400,
            ]:  # First iterate over Easting
                # - 100 on each coordinate: Due to Northing Easting representing south-west point of box
                coords_match = (
                    "N"
                    + str(int(x.NE.y) + y_diff - 100)
                    + "E"
                    + str(int(x.NE.x) + x_diff - 100)
                )
                values = df_insee[
                    df_insee.idINSPIRE.str.contains(coords_match)
                ].index.values
                if len(values) == 0:
                    indices += [None]
                else:  # Cocatenate index value
                    indices += list(values)

        x["indices"] = indices
        return x

        # Get northing and easting coordinates

    coordinates = df_insee.apply(lambda x: get_northing_easting(x), axis=1)[
        ["north", "east"]
    ]

    if conserve_squares_info:  # +100 meters to obtain the centroid of each box
        coords_offset = 100.0
    else:  # +500 meters to obtain the centroid of the 5x5 squared-box
        coords_offset = 500.0

        # North, east coordinates denote the south-west box endpoint:
    north_min, north_max = (
        coordinates.north.min() + coords_offset,
        coordinates.north.max() + coords_offset,
    )
    east_min, east_max = (
        coordinates.east.min() + coords_offset,
        coordinates.east.max() + coords_offset,
    )

    # Create mesh grid: One point for each square's centroid: Each square has an extent of 1km by 1km
    xv, yv = np.meshgrid(
        np.arange(east_min, east_max, step),
        np.arange(north_min, north_max, step),
    )
    points = [Point(x, y) for x, y in zip(xv.ravel(), yv.ravel())]
    # Initialize GeoDataFrame
    df_squares = gpd.GeoDataFrame(
        points, columns=["geometry"], crs="+init=epsg:3035"
    )

    # Project
    df_squares = ox.project_gdf(df_squares, to_crs=df_insee.crs)

    # Save Northing-Easting original coordinates for its later reference
    df_squares["NE"] = points

    if conserve_squares_info:
        index_function = index_square_conservative
    else:
        index_function = index_square

        # Index, for each square centroid, +- 400 meters to achieve squares of 5 by 5
    df_squares = df_squares.apply(
        lambda x: index_function(x, df_insee, offset_index=400), axis=1
    )
    # Update geometry
    df_squares["geometry"] = df_squares.square_geometry

    # Drop useless columns
    df_squares.drop(["square_geometry", "NE"], axis=1, inplace=True)
    # Drop empty squares (rows)
    df_squares.drop(
        df_squares[df_squares.geometry.area == 0].index, axis=0, inplace=True
    )
    # Reset index
    df_squares.reset_index(drop=True, inplace=True)
    # Set CRS key-words
    df_squares.crs = df_insee.crs

    return df_squares


def population_downscaling_validation(df_osm_built, df_insee):
    """Validates the population downscaling estimation by means of aggregating
    the sum of buildings estimated population lying within each population
    square
        Allows to compare the real population count with the estimated
    population lying within each square

    Updates new column 'pop_estimation' for each square in the population data
    frame

        Parameters
        ----------
        df_osm_built : geopandas.GeoDataFrame
                input buildings with computed population count
        df_insee : geopandas.GeoDataFrame
                INSEE population data

        Returns
        ----------

        """
    df_osm_built["geom"] = df_osm_built.geometry
    df_osm_built_residential = df_osm_built[
        df_osm_built.apply(lambda x: x.landuses_m2["residential"] > 0, axis=1)
    ]
    df_insee.crs = df_osm_built_residential.crs

    # Intersecting gridded population - buildings
    sjoin = gpd.sjoin(df_insee, df_osm_built_residential, op="intersects")
    # Calculate area within square (percentage of building with the square)
    sjoin["pop_estimation"] = sjoin.apply(
        lambda x: x.population
        * (x.geom.intersection(x.geometry).area / x.geom.area),
        axis=1,
    )

    # Initialize
    df_insee["pop_estimation"] = np.nan
    sum_pop_per_square = sjoin.groupby(sjoin.index)["pop_estimation"].sum()

    df_insee.loc[
        sum_pop_per_square.index, "pop_estimation"
    ] = sum_pop_per_square.values
    # Drop unnecessary column
    df_osm_built.drop("geom", axis=1, inplace=True)
    # Set to 0 nan values
    df_insee.loc[df_insee.pop_estimation.isnull(), "pop_estimation"] = 0

    # Compute absolute and relative error
    df_insee["absolute_error"] = df_insee.apply(
        lambda x: abs(x.pop_count - x.pop_estimation), axis=1
    )
    df_insee["relative_error"] = df_insee.apply(
        lambda x: abs(x.pop_count - x.pop_estimation) / x.pop_count, axis=1
    )


def get_population_df_filled_empty_squares(df_insee):
    """
        Add empty squares as 0-population box-squares

        Parameters
        ----------
        df_insee : geopandas.GeoDataFrame
                INSEE population data

        Returns
        ----------

        """

    def get_northing_easting(x):  # Extract northing and easting coordinates
        north, east = x.idINSPIRE.split("N")[1].split("E")
        x["north"] = int(north)
        x["east"] = int(east)
        return x

        # Project data to its original projection coordinates

    df_insee_3035 = ox.project_gdf(df_insee, to_crs="+init=epsg:3035")

    # Get northing and easting coordinates
    coordinates = df_insee.apply(lambda x: get_northing_easting(x), axis=1)[
        ["north", "east"]
    ]

    # +100 meters to obtain the centroid of each box
    coords_offset = 100
    # Input data step
    step = 200.0

    # North, east coordinates denote the south-west box endpoint:
    north_min, north_max = (
        coordinates.north.min() + coords_offset,
        coordinates.north.max() + coords_offset,
    )
    east_min, east_max = (
        coordinates.east.min() + coords_offset,
        coordinates.east.max() + coords_offset,
    )

    # Create mesh grid: One point for each square's centroid: Each square has an extent of 1km by 1km
    xv, yv = np.meshgrid(
        np.arange(east_min, east_max, step),
        np.arange(north_min, north_max, step),
    )

    # For every given coordinate, if a box is not created (no population), make it with an initial population of 0
    empty_population_box = []

    for E, N in zip(xv.ravel(), yv.ravel()):  # Center-point
        point_df = gpd.GeoDataFrame(
            [Point(E, N)], columns=["geometry"], crs="+init=epsg:3035"
        )
        if gpd.sjoin(
            point_df, df_insee_3035
        ).empty:  # Does not intersect any existing square-box
            # Create new square
            empty_population_box.append(
                Polygon(
                    [
                        (E - 100.0, N - 100.0),
                        (E - 100.0, N + 100.0),
                        (E + 100.0, N + 100.0),
                        (E + 100.0, N - 100.0),
                        (E - 100.0, N - 100.0),
                    ]
                )
            )

            # Concatenate original data frame + Empty squares
    gdf_concat = pd.concat(
        [
            df_insee_3035,
            gpd.GeoDataFrame(
                {
                    "geometry": empty_population_box,
                    "pop_count": [0] * len(empty_population_box),
                },
                crs="+init=epsg:3035",
            ),
        ],
        ignore_index=True,
        sort=False,
    )

    # Remove added grid-cells outside the convex hull of the population data frame
    df_insee_convex_hull_3035 = df_insee_3035.unary_union.convex_hull
    gdf_concat = gdf_concat[
        gdf_concat.apply(
            lambda x: df_insee_convex_hull_3035.intersects(x.geometry), axis=1
        )
    ]
    gdf_concat.reset_index(drop=True, inplace=True)

    # Project (First project to latitude-longitude due to GeoPandas issue)
    return ox.project_gdf(ox.project_gdf(gdf_concat, to_latlong=True))
