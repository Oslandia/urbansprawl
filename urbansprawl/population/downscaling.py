###############
# Repository: https://github.com/lgervasoni/urbansprawl
# MIT License
###############

import geopandas as gpd
import osmnx as ox

from keras import callbacks, optimizers
from keras.models import Sequential
from keras.layers import Activation, Flatten, Conv1D


def proportional_population_downscaling(df_osm_built, df_insee):
    """
        Performs a proportional population downscaling considering the surface dedicated to residential land use
        Associates the estimated population to each building in column 'population'

        Parameters
        ----------
        df_osm_built : geopandas.GeoDataFrame
                input buildings with computed residential surface
        df_insee : geopandas.GeoDataFrame
                INSEE population data

        Returns
        ----------

        """
    if df_insee.crs != df_osm_built.crs:  # If projections do not match
        # First project to Lat-Long coordinates, then project to UTM coordinates
        df_insee = ox.project_gdf(ox.project_gdf(df_insee, to_latlong=True))

        # OSM Building geometries are already projected
        assert df_insee.crs == df_osm_built.crs

    df_osm_built["geom"] = df_osm_built.geometry
    df_osm_built_residential = df_osm_built[
        df_osm_built.apply(lambda x: x.landuses_m2["residential"] > 0, axis=1)
    ]

    # Loading/saving using geopandas loses the 'ellps' key
    df_insee.crs = df_osm_built_residential.crs

    # Intersecting gridded population - buildings
    sjoin = gpd.sjoin(df_insee, df_osm_built_residential, op="intersects")
    # Calculate area within square (percentage of building with the square)
    sjoin["residential_m2_within"] = sjoin.apply(
        lambda x: x.landuses_m2["residential"]
        * (x.geom.intersection(x.geometry).area / x.geom.area),
        axis=1,
    )
    # Initialize
    df_insee["residential_m2_within"] = 0
    # Sum residential area within square
    sum_m2_per_square = sjoin.groupby(sjoin.index)[
        "residential_m2_within"
    ].sum()
    # Assign total residential area within each square
    df_insee.loc[
        sum_m2_per_square.index, "residential_m2_within"
    ] = sum_m2_per_square.values
    # Get number of M^2 / person
    df_insee["m2_per_person"] = df_insee.apply(
        lambda x: x.residential_m2_within / x.pop_count, axis=1
    )

    def population_building(x, df_insee):
        # Sum of: For each square: M2 of building within square / M2 per person
        return (
            x.get("m2", []) / df_insee.loc[x.get("idx", [])].m2_per_person
        ).sum()
        # Index: Buildings , Values: idx:Indices of gridded square population, m2: M2 within that square

    buildings_square_m2_association = sjoin.groupby("index_right").apply(
        lambda x: {"idx": list(x.index), "m2": list(x.residential_m2_within)}
    )
    # Associate
    df_osm_built.loc[
        buildings_square_m2_association.index, "population"
    ] = buildings_square_m2_association.apply(
        lambda x: population_building(x, df_insee)
    )
    # Drop unnecessary column
    df_osm_built.drop("geom", axis=1, inplace=True)


def build_downscaling_cnn(input_shape):
    """
        """
    _, input_shape_pixels, input_shape_features = input_shape
    model = Sequential()
    model.add(
        Conv1D(
            filters=10,
            kernel_size=1,
            strides=1,
            input_shape=(input_shape_pixels, input_shape_features),
        )
    )
    model.add(Activation("relu"))
    model.add(Conv1D(filters=5, kernel_size=1, strides=1))
    model.add(Activation("relu"))
    model.add(Conv1D(filters=1, kernel_size=1, strides=1))
    model.add(Activation("relu"))
    model.add(Flatten())
    model.add(Activation("softmax"))
    return model


def train_population_downscaling_model(
    X_train, Y_train, X_val, Y_val, batch_size, epochs, checkpoint_filename
):
    """
        Performs a population downscaling by feeding a neural network with
                                          various OSM-related features

        Associates the estimated population to each building in column 'population'

        Parameters
        ----------
        X_train :
        Y_train :
        X_val :
        Y_val :
        batch_size : int
        epochs : int
        checkpoint_filenames : str

        Returns
        ----------
        keras.models.Sequential

        """
    model = build_downscaling_cnn(X_train.shape)
    opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="mean_absolute_error", optimizer=opt, metrics=["mae"])
    checkpoint = callbacks.ModelCheckpoint(
        checkpoint_filename,
        monitor="val_loss",
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        period=1,
    )
    history = model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_val, Y_val),
        callbacks=[checkpoint],
    )
    return history
