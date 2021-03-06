###############
# Repository: https://github.com/lgervasoni/urbansprawl
# MIT License
###############

import geopandas as gpd
import pandas as pd
import numpy as np
import osmnx as ox
import os.path
import time

from osmnx import log

from .utils import (
    get_aggregated_squares,
    get_population_df_filled_empty_squares,
)

# Filenames
from .utils import (
    get_population_urban_features_filename,
    get_population_training_validating_filename,
)

# Sprawl indices
from ..sprawl.dispersion import compute_grid_dispersion
from ..sprawl.landusemix import compute_grid_landusemix

from shapely.geometry import Polygon


def compute_full_urban_features(
    city_ref,
    df_osm_built=None,
    df_osm_pois=None,
    pop_grid=None,
    data_source=None,
    landusemix_args={
        "walkable_distance": 600,
        "compute_activity_types_kde": True,
        "weighted_kde": True,
        "pois_weight": 9,
        "log_weighted": True,
    },
    dispersion_args={
        "radius_search": 750,
        "use_median": True,
        "K_nearest": 50,
    },
    kwargs={"max_dispersion": 15},
):
    """
        Computes a set of urban features for each square where population count
    data exists

        Parameters
        ----------
        city_ref : string
                city reference name
        df_osm_built : geopandas.GeoDataFrame
                input buildings
        df_osm_pois : geopandas.GeoDataFrame
                input points of interest
        pop_grid : geopandas.GeoDataFrame
                grid-cells with population count where urban features will be
        calculated
        data_source : str
                define the type of population data for its retrieval in case it
        was stored
        kwargs : dict
                keyword arguments to guide the process

        Returns
        ----------
        geopandas.GeoDataFrame
                geometry with updated urban features
        """

    # Population extract exists?
    if os.path.exists(
        get_population_urban_features_filename(city_ref, data_source)
    ):
        log(
            "Urban features from population gridded data exist for city: "
            + city_ref
        )
        # Read from GeoJSON (default projection coordinates)
        pop_features_4326 = gpd.read_file(
            get_population_urban_features_filename(city_ref, data_source)
        )
        # Project to UTM coordinates
        return ox.project_gdf(pop_features_4326)

        # Required arguments
    assert df_osm_built is not None
    assert df_osm_pois is not None
    assert pop_grid is not None

    # Get population count data with filled empty squares (null population)
    if data_source == "insee":
        pop_features = get_population_df_filled_empty_squares(
            pop_grid
        )
    elif data_source == "gpw":
        pop_features = pop_grid
    else:
        raise ValueError("Unknown data source.")
    # Set crs
    crs_proj = pop_grid.crs
    pop_features.crs = crs_proj

    ##################
    # Urban features
    ##################
    # Compute the urban features for each square
    log("Calculating urban features")
    start = time.time()

    # Conserve building geometries
    df_osm_built["geom_building"] = df_osm_built["geometry"]

    # Spatial join: grid-cell i - building j for all intersections
    pop_features = gpd.sjoin(
        pop_features, df_osm_built, op="intersects", how="left"
    )

    # When a grid-cell i does not intersect any building: NaN values
    null_idx = pop_features.loc[
        pop_features["geom_building"].isnull()
    ].index
    # Replace NaN for urban features calculation
    min_polygon = Polygon(
        [
            (0, 0),
            (0, np.finfo(float).eps),
            (np.finfo(float).eps, np.finfo(float).eps),
        ]
    )
    pop_features.loc[
        null_idx, "geom_building"
    ] = pop_features.loc[null_idx, "geom_building"].apply(
        lambda x: min_polygon
    )
    pop_features.loc[null_idx, "landuses_m2"] = len(null_idx) * [
        {"residential": 0, "activity": 0}
    ]
    pop_features.loc[null_idx, "building_levels"] = len(
        null_idx
    ) * [0]

    # Pre-calculation of urban features

    # Apply percentage of building presence within square:
    # 1 if fully contained, 0.5 if half the building contained, ...
    pop_features["building_ratio"] = pop_features.apply(
        lambda x: x.geom_building.intersection(x.geometry).area
        / x.geom_building.area,
        axis=1,
    )

    pop_features[
        "m2_total_residential"
    ] = pop_features.apply(
        lambda x: x.building_ratio * x.landuses_m2["residential"], axis=1
    )
    pop_features[
        "m2_total_activity"
    ] = pop_features.apply(
        lambda x: x.building_ratio * x.landuses_m2["activity"], axis=1
    )

    pop_features["m2_footprint_residential"] = 0
    pop_features.loc[
        pop_features.classification.isin(["residential"]),
        "m2_footprint_residential",
    ] = pop_features.loc[
        pop_features.classification.isin(["residential"])
    ].apply(
        lambda x: x.building_ratio * x.geom_building.area, axis=1
    )
    pop_features["m2_footprint_activity"] = 0
    pop_features.loc[
        pop_features.classification.isin(["activity"]),
        "m2_footprint_activity",
    ] = pop_features.loc[
        pop_features.classification.isin(["activity"])
    ].apply(
        lambda x: x.building_ratio * x.geom_building.area, axis=1
    )
    pop_features["m2_footprint_mixed"] = 0
    pop_features.loc[
        pop_features.classification.isin(["mixed"]),
        "m2_footprint_mixed",
    ] = pop_features.loc[
        pop_features.classification.isin(["mixed"])
    ].apply(
        lambda x: x.building_ratio * x.geom_building.area, axis=1
    )

    pop_features["num_built_activity"] = 0
    pop_features.loc[
        pop_features.classification.isin(["activity"]),
        "num_built_activity",
    ] = pop_features.loc[
        pop_features.classification.isin(["activity"])
    ].building_ratio
    pop_features["num_built_residential"] = 0
    pop_features.loc[
        pop_features.classification.isin(["residential"]),
        "num_built_residential",
    ] = pop_features.loc[
        pop_features.classification.isin(["residential"])
    ].building_ratio
    pop_features["num_built_mixed"] = 0
    pop_features.loc[
        pop_features.classification.isin(["mixed"]),
        "num_built_mixed",
    ] = pop_features.loc[
        pop_features.classification.isin(["mixed"])
    ].building_ratio

    pop_features["num_levels"] = pop_features.apply(
        lambda x: x.building_ratio * x.building_levels, axis=1
    )
    pop_features["num_buildings"] = pop_features[
        "building_ratio"
    ]

    pop_features["built_up_m2"] = pop_features.apply(
        lambda x: x.geom_building.area * x.building_ratio, axis=1
    )

    # Urban features aggregation functions
    urban_features_aggregation = {}
    if data_source == "insee":
        urban_features_aggregation["idINSPIRE"] = lambda x: x.head(1)
        urban_features_aggregation["pop_count"] = lambda x: x.head(1)
    elif data_source == "gpw":
        urban_features_aggregation["idx"] = lambda x: x.head(1)
    urban_features_aggregation["geometry"] = lambda x: x.head(1)

    urban_features_aggregation["m2_total_residential"] = "sum"
    urban_features_aggregation["m2_total_activity"] = "sum"

    urban_features_aggregation["m2_footprint_residential"] = "sum"
    urban_features_aggregation["m2_footprint_activity"] = "sum"
    urban_features_aggregation["m2_footprint_mixed"] = "sum"

    urban_features_aggregation["num_built_activity"] = "sum"
    urban_features_aggregation["num_built_residential"] = "sum"
    urban_features_aggregation["num_built_mixed"] = "sum"

    urban_features_aggregation["num_levels"] = "sum"
    urban_features_aggregation["num_buildings"] = "sum"

    urban_features_aggregation["built_up_m2"] = "sum"

    # Apply aggregate functions
    pop_features = pop_features.groupby(
        pop_features.index
    ).agg(urban_features_aggregation)

    # Calculate built up relation (relative to the area of the grid-cell geometry)
    pop_features[
        "built_up_relation"
    ] = pop_features.apply(
        lambda x: x.built_up_m2 / x.geometry.area, axis=1
    )
    pop_features.drop("built_up_m2", axis=1, inplace=True)

    # To geopandas.GeoDataFrame and set crs
    pop_features = gpd.GeoDataFrame(pop_features)
    pop_features.crs = crs_proj

    # POIs
    df_osm_pois_selection = df_osm_pois[
        df_osm_pois.classification.isin(["activity", "mixed"])
    ]
    gpd_intersection_pois = gpd.sjoin(
        pop_features,
        df_osm_pois_selection,
        op="intersects",
        how="left",
    )
    # Number of activity/mixed POIs
    pop_features[
        "num_activity_pois"
    ] = gpd_intersection_pois.groupby(gpd_intersection_pois.index).agg(
        {"osm_id": "count"}
    )

    ##################
    # Sprawling indices
    ##################
    pop_features["geometry_squares"] = pop_features.geometry
    pop_features["geometry"] = pop_features.geometry.centroid

    # Compute land uses mix + densities estimation
    compute_grid_landusemix(
        pop_features, df_osm_built, df_osm_pois, landusemix_args
    )
    # Dispersion indices
    compute_grid_dispersion(pop_features, df_osm_built, dispersion_args)

    # Set back original geometries
    pop_features["geometry"] = pop_features.geometry_squares
    pop_features.drop("geometry_squares", axis=1, inplace=True)

    if kwargs.get("max_dispersion"):  # Set max bounds for dispersion values
        pop_features.loc[
            pop_features.dispersion > kwargs.get("max_dispersion"),
            "dispersion",
        ] = kwargs.get("max_dispersion")

    # Fill NaN sprawl indices with 0
    pop_features.fillna(0, inplace=True)

    # Save to GeoJSON file (no projection conserved, then use EPSG 4326)
    ox.project_gdf(pop_features, to_latlong=True).to_file(
        get_population_urban_features_filename(city_ref, data_source),
        driver="GeoJSON",
    )

    elapsed_time = int(time.time() - start)
    log(
        "Done: Urban features calculation. Elapsed time (H:M:S): "
        + "{:02d}:{:02d}:{:02d}".format(
            elapsed_time // 3600,
            (elapsed_time % 3600 // 60),
            elapsed_time % 60,
        )
    )
    return pop_features


def get_training_testing_data(city_ref, pop_features=None):
    """
        Returns the Y and X arrays for training/testing population downscaling estimates.

        Y contains vectors with the correspondent population densities
        X contains vectors with normalized urban features
        X_columns columns referring to X values
        Numpy arrays are stored locally

        Parameters
        ----------
        city_ref : string
                city reference name
        pop_features : geopandas.GeoDataFrame
                grid-cells with population count data and calculated urban features

        Returns
        ----------
        np.array, np.array, np.array
                Y vector, X vector, X column names vector
        """
    # Population extract exists?
    if os.path.exists(get_population_training_validating_filename(city_ref)):
        log(
            "Urban population training+validation data/features exist for input city: "
            + city_ref
        )
        # Read from Numpy.Arrays
        data = np.load(get_population_training_validating_filename(city_ref))
        # Project to UTM coordinates
        return data["Y"], data["X"], data["X_columns"]

    log(
        "Calculating urban training+validation data/features for city: "
        + city_ref
    )
    start = time.time()

    # Select columns to normalize
    columns_to_normalise = [
        col
        for col in pop_features.columns
        if "num_" in col
        or "m2_" in col
        or "dispersion" in col
        or "accessibility" in col
    ]
    # Normalize selected columns
    pop_features.loc[
        :, columns_to_normalise
    ] = pop_features.loc[:, columns_to_normalise].apply(
        lambda x: x / x.max(), axis=0
    )

    # By default, idINSPIRE for created squares (0 population count) is 0:
    # Change for 'CRS' string: Coherent with squares aggregation procedure
    # (string matching)
    pop_features.loc[
        pop_features.idINSPIRE == "0", "idINSPIRE"
    ] = "CRS"

    # Aggregate 5x5 squares: Get all possible aggregations
    # (step of 200 meters = length of individual square)
    aggregated_pop_features = get_aggregated_squares(
        ox.project_gdf(pop_features, to_crs="+init=epsg:3035"),
        step=200.0,
        conserve_squares_info=True,
    )

    # X values: Vector <x1,x2, ... , xn> with normalized urban features
    X_values = []
    # Y values: Vector <y1, y2, ... , ym>
    # with normalized population densities. m=25
    Y_values = []

    # For each <Indices> combination, create a X and Y vector
    for idx in aggregated_pop_features.indices:
        # Extract the urban features in the given 'indices' order
        # (Fill to 0 for non-existent squares)
        square_info = pop_features.reindex(idx).fillna(0)
        # Y input (Ground truth): Population densities
        population_densities = (
            square_info["pop_count"] / square_info["pop_count"].sum()
        ).values

        if all(
            pd.isna(population_densities)
        ):  # If sum of population count is 0, remove (NaN values)
            continue

        # X input: Normalized urban features
        urban_features = square_info[
            [
                col
                for col in square_info.columns
                if col not in ["idINSPIRE", "geometry", "pop_count"]
            ]
        ].values

        # Append X, Y
        X_values.append(urban_features)
        Y_values.append(population_densities)

        # Get the columns order referenced in each X vector
    X_values_columns = pop_features[
        [
            col
            for col in square_info.columns
            if col not in ["idINSPIRE", "geometry", "pop_count"]
        ]
    ].columns
    X_values_columns = np.array(X_values_columns)

    # To Numpy Array
    X_values = np.array(X_values)
    Y_values = np.array(Y_values)

    # Save to file
    np.savez(
        get_population_training_validating_filename(city_ref),
        X=X_values,
        Y=Y_values,
        X_columns=X_values_columns,
    )

    log(
        "Done: urban training+validation data/features. Elapsed time (H:M:S): "
        + time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    )

    return Y_values, X_values, X_values_columns


def get_Y_X_features_population_data(cities_selection=None, cities_skip=None):
    """
        Returns the Y and X arrays for training/testing population downscaling estimates.
        It gathers either a selection of cities or all stored cities but a selected list to skip

        Y contains vectors with the correspondent population densities
        X contains vectors with normalized urban features
        X_columns columns referring to X values
        Numpy arrays are previously stored

        Parameters
        ----------
        cities_selection : string
                list of cities to select
        cities_skip : string
                list of cities to skip (retrieve the rest)

        Returns
        ----------
        np.array, np.array, np.array
                Y vector, X vector, X column names vector
        """
    arr_X, arr_Y = [], []

    # Get the complete training-testig dataset
    for Y_X_data_city in os.listdir("data/training"):
        # Only if it contains a valid extension
        if ".npz" not in Y_X_data_city:
            continue

        # Get city's name
        city_ref = Y_X_data_city.replace("_X_Y.npz", "")

        # Only retrieve data from cities_selection (if ever given)
        if (cities_selection is not None) and (
            city_ref not in cities_selection
        ):
            log("Skipping city: " + str(city_ref))
            continue

            # Skip cities data from from cities_skip (if ever given)
        if (cities_skip is not None) and (city_ref in cities_skip):
            log("Skipping city:", city_ref)
            continue

        log("Retrieving data for city: " + str(city_ref))

        # Get stored data
        city_Y, city_X, city_X_cols = get_training_testing_data(city_ref)
        # Append values
        arr_Y.append(city_Y)
        arr_X.append(city_X)

        # Assumption: All generated testing-training data contain the same X columns
    return np.concatenate(arr_Y), np.concatenate(arr_X), city_X_cols


def prepare_testing_data(city_ref, pop_features=None):
    """Return a X array for population downscaling inference, that contain
    normalized urban features

        X contains vectors with normalized urban features
        X_columns columns referring to X values
        Numpy arrays are stored locally

        Parameters
        ----------
        city_ref : string
                city reference name
        pop_features : geopandas.GeoDataFrame
                grid-cells with population count data and calculated urban features

        Returns
        ----------
        np.array, np.array, np.array
                Y vector, X vector, X column names vector
        """
    log(
        "Calculating urban testing data/features for city: " + city_ref
    )
    start = time.time()

    # Select columns to normalize
    columns_to_normalise = [
        col
        for col in pop_features.columns
        if "num_" in col
        or "m2_" in col
        or "dispersion" in col
        or "accessibility" in col
    ]
    # Normalize selected columns
    pop_features.loc[
        :, columns_to_normalise
    ] = pop_features.loc[:, columns_to_normalise].apply(
        lambda x: x / x.max(), axis=0
    )

    # X values: Vector <x1,x2, ... , xn> with normalized urban features
    X_values = []
    geom_values = []

    for idx in pop_features.idx.unique():
        square_info = pop_features[pop_features["idx"] == idx]
        urban_features = square_info[
            [col for col in square_info.columns
             if col not in ["geometry", "pop_count", "idx"]]
        ].values
        X_values.append(urban_features)
        geom = square_info["geometry"]
        geom_values.append(geom)

    # Get the columns order referenced in each X vector
    X_values_columns = pop_features[
        [
            col
            for col in square_info.columns
            if col not in ["geometry", "pop_count", "idx"]
        ]
    ].columns
    X_values_columns = np.array(X_values_columns)
    X_values = np.array(X_values)
    geom_values = np.array(geom_values)

    log(
        "Done: urban training+validation data/features. Elapsed time (H:M:S): "
        + time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    )

    return X_values, X_values_columns, geom_values
