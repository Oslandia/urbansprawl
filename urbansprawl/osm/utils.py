###############
# Repository: https://github.com/lgervasoni/urbansprawl
# MIT License
###############

import osmnx as ox
import pandas as pd
import geopandas as gpd
import numpy as np

from .tags import height_tags

from ..settings import storage_folder

# Format for load/save the geo-data ['geojson','shp']
geo_format = "geojson"  # 'shp'
geo_driver = "GeoJSON"  # 'ESRI Shapefile'

###################################################
# I/O utils
###################################################


def get_dataframes_filenames(city_ref_file):
    """
        Get data frame file names for input city

        Parameters
        ----------
        city_ref_file : string
                name of input city

        Returns
        ----------
        [ string, string, string ]
                returns filenames for buildings, building parts, and points of
        interest
        """
    import os

    if not (os.path.isdir(storage_folder)):
        os.makedirs(storage_folder)
    geo_poly_file = (
        storage_folder + "/" + city_ref_file + "_buildings." + geo_format
    )
    geo_poly_parts_file = (
        storage_folder + "/" + city_ref_file + "_building_parts." + geo_format
    )
    geo_point_file = (
        storage_folder + "/" + city_ref_file + "_poi." + geo_format
    )
    return geo_poly_file, geo_poly_parts_file, geo_point_file


def load_geodataframe(geo_filename):
    """
        Load input GeoDataFrame

        Parameters
        ----------
        geo_filename : string
                input GeoDataFrame filename

        Returns
        ----------
        geopandas.GeoDataFrame
                loaded data

        """
    # Load using geopandas
    df_osm_data = gpd.read_file(geo_filename)
    # Set None as NaN
    df_osm_data.fillna(value=np.nan, inplace=True)
    # Replace empty string (Json NULL sometimes read as '') for NaN
    df_osm_data.replace("", np.nan, inplace=True)

    def list_int_from_string(
        x
    ):  # List of integers given input in string format
        return [int(id_) for id_ in x.split(",")]

    def list_str_from_string(
        x
    ):  # List of strings given input in string format
        return x.split(",")

        # Recover list

    if "activity_category" in df_osm_data.columns:
        df_osm_data["activity_category"] = df_osm_data.activity_category.apply(
            lambda x: list_str_from_string(x) if pd.notnull(x) else np.nan
        )
    if "containing_parts" in df_osm_data.columns:
        df_osm_data["containing_parts"] = df_osm_data.containing_parts.apply(
            lambda x: list_int_from_string(x) if pd.notnull(x) else np.nan
        )
    if "containing_poi" in df_osm_data.columns:
        df_osm_data["containing_poi"] = df_osm_data.containing_poi.apply(
            lambda x: list_int_from_string(x) if pd.notnull(x) else np.nan
        )

        # To UTM coordinates
    return ox.project_gdf(df_osm_data)


def store_geodataframe(df_osm_data, geo_filename):
    """
        Store input GeoDataFrame

        Parameters
        ----------
        df_osm_data : geopandas.GeoDataFrame
                input OSM data frame
        geo_filename : string
                filename for GeoDataFrame storage

        Returns
        ----------

        """
    # To EPSG 4326 (GeoJSON does not store projection information)
    df_osm_data = ox.project_gdf(df_osm_data, to_latlong=True)

    # Lists to string (needed to save GeoJSON files)
    if "activity_category" in df_osm_data.columns:
        df_osm_data.activity_category = df_osm_data.activity_category.apply(
            lambda x: ",".join(str(e) for e in x)
            if isinstance(x, list)
            else np.nan
        )
    if "containing_parts" in df_osm_data.columns:
        df_osm_data.containing_parts = df_osm_data.containing_parts.apply(
            lambda x: ",".join(str(e) for e in x)
            if isinstance(x, list)
            else np.nan
        )
    if "containing_poi" in df_osm_data.columns:
        df_osm_data.containing_poi = df_osm_data.containing_poi.apply(
            lambda x: ",".join(str(e) for e in x)
            if isinstance(x, list)
            else np.nan
        )

        # Save to file
    df_osm_data.to_file(geo_filename, driver=geo_driver)


###################################################
# GeoDataFrame processing utils
###################################################


def sanity_check_height_tags(df_osm):
    """
        Compute a sanity check for all height tags

        If incorrectly tagged, try to replace with the correct tag

        Any meter or level related string are replaced, and heights using the
        imperial units are converted to the metric system

        Parameters
        ----------
        df_osm : geopandas.GeoDataFrame
                input OSM data frame

        Returns
        ----------

        """

    def sanity_check(value):
        # Sanity check for height tags (sometimes wrongly-tagged)
        if not ((value is np.nan) or (value is None) or (value == "")):  # Non-null value
            try:  # Can be read as float?
                return float(value)
            except ValueError:
                try:  # Try removing incorrectly tagged information: meters/levels
                    return float(
                        value.replace("meters", "")
                        .replace("meter", "")
                        .replace("m", "")
                        .replace("levels", "")
                        .replace("level", "")
                        .replace("l", "")
                    )
                except ValueError:
                    try:  # Feet and inch values? e.g.: 4'7''
                        split_value = value.split("'")
                        feet, inches = split_value[0], split_value[1]
                        if inches is "":  # Non existent inches
                            inches = "0"
                        tot_inches = float(feet) * 12 + float(inches)
                        # Return meters equivalent
                        return tot_inches * 0.0254
                    except TypeError:  # None. Incorrect tag
                        return None
        return value

        # Available height tags

    available_height_tags = [
        col for col in height_tags if col in df_osm.columns
    ]
    # Apply-map sanity check
    df_osm[available_height_tags] = df_osm[available_height_tags].applymap(
        sanity_check
    )


def associate_structures(
    df_osm_encompassing_structures,
    df_osm_structures,
    operation="contains",
    column="containing_",
):
    """
        Associate input structure geometries to its encompassing structures
        Structures are associated using the operation 'contains' or 'intersects'
        A new column in the encompassing data frame is added, incorporating the indices of the containing structures

        Parameters
        ----------
        df_osm_encompassing_structures : geopandas.GeoDataFrame
                encompassing data frame
        df_osm_structures : geopandas.GeoDataFrame
                structures data frame
        operation : string
                spatial join operation to associate structures
        column : string
                name of the column to add in encompassing data frame

        Returns
        ----------

        """
    # Find, for each geometry, all containing structures
    sjoin = gpd.sjoin(
        df_osm_encompassing_structures[["geometry"]],
        df_osm_structures[["geometry"]],
        op=operation,
        rsuffix="cont",
    )
    # Group by: polygon_index -> list of containing points indices
    group_indices = sjoin.groupby(sjoin.index, as_index=True)[
        "index_cont"
    ].apply(list)
    # Create new column
    df_osm_encompassing_structures.loc[
        group_indices.index, column
    ] = group_indices.values
    # Reset indices
    df_osm_encompassing_structures.index.rename("", inplace=True)
    df_osm_structures.index.rename("", inplace=True)
