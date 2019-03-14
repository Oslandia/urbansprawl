"""This module aims at recovering OpenStreetMap data through Overpass API

To accomplish a task, the following command must be run on the terminal:

```
python -m luigi --local-scheduler --module urbansprawl.tasks <Task> <params>
```

with `Task` one of the class defined below, and `params` the corresponding
parameters.

This computation is done locally (because of the ̀--local-scheduler` option). It
can be done on a server, by first launching an instance of the luigi daemon :
```
luigid
̀``

and then by running the previous command without the `--local-scheduler`
option. The task dependency graph and some miscellaneous information about the
tasks are visible at `localhost:8082` URL address.
"""

from configparser import ConfigParser
from datetime import date, datetime as dt
import json
import os
import requests
import sys
import zipfile

import geopandas as gpd
import luigi
from luigi.format import MixedUnicodeBytes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import numpy as np
import osmnx
from osgeo import gdal, ogr, osr
import pandas as pd
import sh
from shapely.geometry import Point, Polygon

from urbansprawl.osm.overpass import (
    create_buildings_gdf,
    create_building_parts_gdf,
    create_pois_gdf,
    create_landuse_gdf,
    retrieve_route_graph,
)
from urbansprawl.osm.utils import (
    sanity_check_height_tags,
    associate_structures,
)
from urbansprawl.osm.classification import (
    classify_tag,
    classify_activity_category,
    compute_landuse_inference,
)
from urbansprawl.osm.surface import compute_landuses_m2
from urbansprawl.sprawl.core import get_indices_grid_from_bbox
from urbansprawl.sprawl.landusemix import compute_grid_landusemix
from urbansprawl.sprawl.accessibility import compute_grid_accessibility
from urbansprawl.sprawl.dispersion import compute_grid_dispersion
from urbansprawl.population.data_extract import get_extract_population_data
from urbansprawl.population.urban_features import (
    compute_full_urban_features,
    get_training_testing_data,
    get_Y_X_features_population_data,
    prepare_testing_data
)
from urbansprawl.population.downscaling import (
    train_population_downscaling_model,
    build_downscaling_cnn,
)
from urbansprawl import geometries

config = ConfigParser()
if os.path.isfile("config.ini"):
    config.read("config.ini")
else:
    sys.exit(1)


# Columns of interest corresponding to OSM keys
OSM_TAG_COLUMNS = [
    "amenity",
    "landuse",
    "leisure",
    "shop",
    "man_made",
    "building",
    "building:use",
    "building:part",
]
COLUMNS_OF_INTEREST = OSM_TAG_COLUMNS + ["osm_id", "geometry", "height_tags"]
COLUMNS_OF_INTEREST_POIS = OSM_TAG_COLUMNS + ["osm_id", "geometry"]
COLUMNS_OF_INTEREST_LANDUSES = ["osm_id", "geometry", "landuse"]
HEIGHT_TAGS = [
    "min_height",
    "height",
    "min_level",
    "levels",
    "building:min_height",
    "building:height",
    "building:min_level",
    "building:levels",
    "building:levels:underground",
]
BUILDING_PARTS_TO_FILTER = ["no", "roof"]
MINIMUM_M2_BUILDING_AREA = 9.0


def define_filename(description, city, date, datapath, extension):
    """Build a distinctive filename regarding a given `description`, `city`,
    `date` (ISO-formatted), ̀datapath` and a `extension` for the file extension

    Parameters
    ----------
    description : str
        Describe the file content in one word
    city : str
        City of interest, used for the queries to Overpass API
    date : str
        Date of the Overpass query, in ISO format
    datapath : str
        Path of the file on the file system
    extension : str
        File extension, *i.e.* GeoJSON

    Returns
    -------
    str
        Full path name on the file system
    """
    os.makedirs(datapath, exist_ok=True)
    filename = "{}-{}.{}".format(description, date, extension)
    return os.path.join(datapath, city, filename)


def set_list_as_str(l):
    """Small utility function to transform list in string

    Parameters
    ----------
    l : list
        Input list

    Returns
    -------
    str
        Stringified version of the input list, with items separated with a comma
    """
    if type(l) == list:
        return ",".join(str(e) for e in l)


def clean_list_in_geodataframe_column(gdf, column):
    """Stringify items of `column` within ̀gdf`, in order to allow its
    serialization

    Parameters
    ----------
    gdf : GeoDataFrame
        Input data structure
    column : str
        Column to modify

    Returns
    -------
    GeoDataFrame
        Modified input structure, with a fixed `column` (contains stringified items)
    """
    if column in gdf.columns:
        gdf[column] = gdf[column].apply(lambda x: set_list_as_str(x))
    return gdf


class GetBoundingBox(luigi.Task):
    """Extract the bounding box around a given `city`

    Example:
    ```
    python -m luigi --local-scheduler --module urbansprawl.tasks GetBoundingBox
    --city valence-drome
    ```

    Attributes
    ----------
    city : str
        City of interest
    datapath : str
        Indicates the folder where the task result has to be serialized
    (default: `./data`)
    """
    city = luigi.Parameter()
    datapath = luigi.Parameter("./data")

    def output(self):
        """Indicates the task result destination onto the file system
        """
        path = os.path.join(self.datapath, self.city)
        os.makedirs(path, exist_ok=True)
        return luigi.LocalTarget(os.path.join(path, "bounding_box.geojson"))

    def run(self):
        """Main operations of the Luigi task
        """
        city_gdf = osmnx.gdf_from_place(self.city, which_result=1)
        city_gdf.to_file(self.output().path, driver="GeoJSON")


class GetData(luigi.Task):
    """Give a raw version of OpenStreetMap items through an Overpass API query
    (buildlings, building parts, POIs or land uses)

    Example:
    ```
    python -m luigi --local-scheduler --module urbansprawl.tasks GetData
    --city valence-drome --date-query 2017-01-01T1200 --table buildings
    ```

    Attributes
    ----------
    city : str
        City of interest
    datapath : str
        Indicates the folder where the task result has to be serialized
    geoformat : str
        Output file extension (by default: `GeoJSON`)
    date_query : str
        Date to which the OpenStreetMap data must be recovered (format:
    AAAA-MM-DDThhmm)
    table : str
        Type of data to retrieve (either `buildings`, `building-parts`, `pois`
    or `land-uses`)

    """

    city = luigi.Parameter()
    datapath = luigi.Parameter("./data")
    geoformat = luigi.Parameter("geojson")
    date_query = luigi.DateMinuteParameter(default=date.today())
    table = luigi.Parameter("buildings")

    def requires(self):
        """Gives the task(s) that are needed to accomplish the current one. It
        refers implicitely to the project dependency graph.
        """
        return GetBoundingBox(self.city, self.datapath)

    def output(self):
        output_path = define_filename(
            "raw-" + self.table,
            self.city,
            dt.date(self.date_query).isoformat(),
            self.datapath,
            self.geoformat,
        )
        return luigi.LocalTarget(output_path)

    def run(self):
        city_gdf = gpd.read_file(self.input().path)
        north, south, east, west = city_gdf.loc[
            0, ["bbox_north", "bbox_south", "bbox_east", "bbox_west"]
        ]
        date = "[date:'" + str(self.date_query) + "']"
        if self.table == "buildings":
            gdf = create_buildings_gdf(
                date=date, north=north, south=south, east=east, west=west
            )
            gdf.drop(["nodes"], axis=1, inplace=True)
        elif self.table == "building-parts":
            gdf = create_building_parts_gdf(
                date=date, north=north, south=south, east=east, west=west
            )
            if "building" in gdf.columns:
                gdf = gdf[
                    (~gdf["building:part"].isin(BUILDING_PARTS_TO_FILTER))
                    & (~gdf["building:part"].isnull())
                    & (gdf["building"].isnull())
                ]
            else:
                gdf = gdf[
                    (~gdf["building:part"].isin(BUILDING_PARTS_TO_FILTER))
                    & (~gdf["building:part"].isnull())
                ]
            if "nodes" in gdf.columns:
                gdf.drop(["nodes"], axis=1, inplace=True)
            gdf["osm_id"] = gdf.index
            gdf.reset_index(drop=True, inplace=True)
        elif self.table == "pois":
            gdf = create_pois_gdf(
                date=date, north=north, south=south, east=east, west=west
            )
            columns_to_drop = [
                col
                for col in list(gdf.columns)
                if col not in COLUMNS_OF_INTEREST_POIS
            ]
            gdf.drop(columns_to_drop, axis=1, inplace=True)
            gdf["osm_id"] = gdf.index
            gdf.reset_index(drop=True, inplace=True)
        elif self.table == "land-uses":
            gdf = create_landuse_gdf(
                date=date, north=north, south=south, east=east, west=west
            )
            gdf = gdf[["landuse", "geometry"]]
            gdf["osm_id"] = gdf.index
            columns_to_drop = [
                col
                for col in list(gdf.columns)
                if col not in COLUMNS_OF_INTEREST_LANDUSES
            ]
            gdf.drop(columns_to_drop, axis=1, inplace=True)
            gdf.reset_index(drop=True, inplace=True)
        else:
            raise ValueError(
                (
                    "Please provide a valid table name (either "
                    "'buildings', 'building-parts', 'pois' "
                    "or 'land-uses')."
                )
            )
        gdf.to_file(self.output().path, driver="GeoJSON")


class SanityCheck(luigi.Task):
    """Check buildings and building parts GeoDataFrames, especially their
    height tags

    Example:
    ```
    python -m luigi --local-scheduler --module urbansprawl.tasks SanityCheck
    --city valence-drome --table buildings
    ```

    Attributes
    ----------
    city : str
        City of interest
    datapath : str
        Indicates the folder where the task result has to be serialized
    geoformat : str
        Output file extension (by default: `GeoJSON`)
    date_query : str
        Date to which the OpenStreetMap data must be recovered (format:
    AAAA-MM-DDThhmm)
    table : str
        Structure to check, either `buildings` or `building-parts`
    """

    city = luigi.Parameter()
    datapath = luigi.Parameter("./data")
    geoformat = luigi.Parameter("geojson")
    date_query = luigi.DateMinuteParameter(default=date.today())
    table = luigi.Parameter(default="buildings")

    def requires(self):
        if self.table in ["buildings", "building-parts"]:
            return GetData(
                self.city,
                self.datapath,
                self.geoformat,
                self.date_query,
                self.table,
            )
        else:
            raise ValueError(
                (
                    "Please provide a valid table name (either "
                    "'buildings' or 'building-parts')."
                )
            )

    def output(self):
        output_path = define_filename(
            "checked-" + self.table,
            self.city,
            dt.date(self.date_query).isoformat(),
            self.datapath,
            self.geoformat,
        )
        return luigi.LocalTarget(output_path)

    def run(self):
        gdf = gpd.read_file(self.input().path)
        sanity_check_height_tags(gdf)

        def remove_nan_dict(x):
            """Remove entries with nan values
            """
            return {k: v for k, v in x.items() if pd.notnull(v)}

        gdf["height_tags"] = gdf[
            [c for c in HEIGHT_TAGS if c in gdf.columns]
        ].apply(lambda x: remove_nan_dict(x.to_dict()), axis=1)
        columns_to_drop = [
            col for col in list(gdf.columns) if col not in COLUMNS_OF_INTEREST
        ]
        gdf.drop(columns_to_drop, axis=1, inplace=True)
        gdf["osm_id"] = gdf.index
        gdf.reset_index(drop=True, inplace=True)
        gdf.to_file(self.output().path, driver="GeoJSON")


class GetClassifiedInfo(luigi.Task):
    """Classify each building, building part or POI record as "residential",
    "activity" or "mixed" according to the associated tags

    Example:
    ```
    python -m luigi --local-scheduler --module urbansprawl.tasks GetClassifiedInfo
    --city valence-drome --table buildings
    ```

    Attributes
    ----------
    city : str
        City of interest
    datapath : str
        Indicates the folder where the task result has to be serialized
    geoformat : str
        Output file extension (by default: `GeoJSON`)
    date_query : str
        Date to which the OpenStreetMap data must be recovered (format:
    AAAA-MM-DDThhmm)
    table : str
        Structure of interest, either `buildings`, `building-parts` or `pois`
    """

    city = luigi.Parameter()
    datapath = luigi.Parameter("./data")
    geoformat = luigi.Parameter("geojson")
    date_query = luigi.DateMinuteParameter(default=date.today())
    table = luigi.Parameter(default="buildings")

    def requires(self):
        if self.table in ["buildings", "building-parts"]:
            return SanityCheck(
                self.city,
                self.datapath,
                self.geoformat,
                self.date_query,
                self.table,
            )
        elif self.table == "pois":
            return GetData(
                self.city,
                self.datapath,
                self.geoformat,
                self.date_query,
                self.table,
            )
        else:
            raise ValueError(
                (
                    "Please provide a valid table name (either "
                    "'buildings', 'building-parts', 'pois')."
                )
            )

    def output(self):
        output_path = define_filename(
            "classified-" + self.table,
            self.city,
            dt.date(self.date_query).isoformat(),
            self.datapath,
            self.geoformat,
        )
        return luigi.LocalTarget(output_path)

    def run(self):
        gdf = gpd.read_file(self.input().path)
        gdf["classification"], gdf["key_value"] = list(
            zip(*gdf.apply(classify_tag, axis=1))
        )
        if self.table == "buildings":
            gdf.drop(gdf[gdf.classification.isnull()].index, inplace=True)
            gdf.reset_index(inplace=True, drop=True)
        elif self.table == "building-parts":
            # Building parts will acquire its containing building land use
            # if it is not available
            gdf.loc[
                gdf.classification.isin(["infer", "other"]), "classification"
            ] = None
        elif self.table == "pois":
            gdf.drop(
                gdf[
                    gdf.classification.isin(["infer", "other"])
                    | gdf.classification.isnull()
                ].index,
                inplace=True,
            )
            gdf.reset_index(inplace=True, drop=True)
        else:
            raise ValueError(
                (
                    "Please provide a valid table name (either "
                    "'buildings', 'building-parts', 'pois')."
                )
            )
        # Drop tag-related columns
        columns_to_drop = [
            col for col in OSM_TAG_COLUMNS if col in gdf.columns
        ]
        gdf.drop(columns_to_drop, axis=1, inplace=True)
        gdf.to_file(self.output().path, driver="GeoJSON")


class SetupProjection(luigi.Task):
    """Fix the GeoDataFrames projections, so as to ensure that every
    GeoDataFrames has the same projection

    Example:
    ```
    python -m luigi --local-scheduler --module urbansprawl.tasks SetupProjection
    --city valence-drome --table buildings-parts
    ```

    Attributes
    ----------
    city : str
        City of interest
    datapath : str
        Indicates the folder where the task result has to be serialized
    geoformat : str
        Output file extension (by default: `GeoJSON`)
    date_query : str
        Date to which the OpenStreetMap data must be recovered (format:
    AAAA-MM-DDThhmm)
    table : str
        Structure of interest, either `buildings`, `building-parts` or `pois`
    """

    city = luigi.Parameter()
    datapath = luigi.Parameter("./data")
    geoformat = luigi.Parameter("geojson")
    date_query = luigi.DateMinuteParameter(default=date.today())
    table = luigi.Parameter(default="buildings")

    def requires(self):
        if self.table in ["buildings", "building-parts", "pois"]:
            return GetClassifiedInfo(
                self.city,
                self.datapath,
                self.geoformat,
                self.date_query,
                self.table,
            )
        elif self.table == "land-uses":
            return GetData(
                self.city,
                self.datapath,
                self.geoformat,
                self.date_query,
                self.table,
            )
        else:
            raise ValueError(
                (
                    "Please provide a valid table name (either "
                    "'buildings', 'building-parts', "
                    "'pois' or 'land-uses')."
                )
            )

    def output(self):
        output_path = define_filename(
            "reprojected-" + self.table,
            self.city,
            dt.date(self.date_query).isoformat(),
            self.datapath,
            self.geoformat,
        )
        return luigi.LocalTarget(output_path)

    def run(self):
        gdf = gpd.read_file(self.input().path)
        # Project to UTM coordinates within the same zone
        gdf = osmnx.project_gdf(gdf)
        if self.table == "buildings":
            gdf.drop(
                gdf[gdf.geometry.area < MINIMUM_M2_BUILDING_AREA].index,
                inplace=True,
            )
        proj_path = os.path.join(
            self.datapath, self.city, "utm_projection.json"
        )
        if not os.path.isfile(proj_path):
            with open(proj_path, "w") as fobj:
                json.dump(gdf.crs, fobj)
        gdf.to_file(self.output().path, driver="GeoJSON")


class InferLandUse(luigi.Task):
    """Infer land use of each OpenStreetMap building thanks to land use
    information

    Example:
    ```
    python -m luigi --local-scheduler --module urbansprawl.tasks InferLandUse
    --city valence-drome --date-query 2017-01-01T1200
    ```

    Attributes
    ----------
    city : str
        City of interest
    datapath : str
        Indicates the folder where the task result has to be serialized
    geoformat : str
        Output file extension (by default: `GeoJSON`)
    date_query : str
        Date to which the OpenStreetMap data must be recovered (format:
    AAAA-MM-DDThhmm)
    """

    city = luigi.Parameter()
    datapath = luigi.Parameter("./data")
    geoformat = luigi.Parameter("geojson")
    date_query = luigi.DateMinuteParameter(default=date.today())

    def requires(self):
        return {
            "buildings": SetupProjection(
                self.city,
                self.datapath,
                self.geoformat,
                self.date_query,
                "buildings",
            ),
            "land-uses": SetupProjection(
                self.city,
                self.datapath,
                self.geoformat,
                self.date_query,
                "land-uses",
            ),
        }

    def output(self):
        output_path = define_filename(
            "infered-buildings",
            self.city,
            dt.date(self.date_query).isoformat(),
            self.datapath,
            self.geoformat,
        )
        return luigi.LocalTarget(output_path)

    def run(self):
        buildings = gpd.read_file(self.input()["buildings"].path)
        land_uses = gpd.read_file(self.input()["land-uses"].path)
        proj_path = os.path.join(
            self.datapath, self.city, "utm_projection.json"
        )
        with open(proj_path) as fobj:
            utm_proj = json.load(fobj)
            buildings.crs = utm_proj
            land_uses.crs = utm_proj
        compute_landuse_inference(buildings, land_uses)
        assert (
            len(buildings[buildings.key_value == {"inferred": "other"}]) == 0
        )
        assert len(buildings[buildings.classification.isnull()]) == 0
        buildings.to_file(self.output().path, driver="GeoJSON")


class ComputeLandUse(luigi.Task):
    """Compute land use per building type (residential, activity or mixed)

    Example:
    ```
    python -m luigi --local-scheduler --module urbansprawl.tasks ComputeLandUse
    --city valence-drome --default-heights 6 --meters-per-level 2
    ```

    Attributes
    ----------
    city : str
        City of interest
    datapath : str
        Indicates the folder where the task result has to be serialized
    geoformat : str
        Output file extension (by default: `GeoJSON`)
    date_query : str
        Date to which the OpenStreetMap data must be recovered (format:
    AAAA-MM-DDThhmm)
    default_height : int
        Default building height, in meters (default: 3 meters)
    meters_per_level : int
        Default height per level, in meter (default: 3 meters)
    """

    city = luigi.Parameter()
    datapath = luigi.Parameter("./data")
    geoformat = luigi.Parameter("geojson")
    date_query = luigi.DateMinuteParameter(default=date.today())
    default_height = luigi.IntParameter(3)
    meters_per_level = luigi.IntParameter(3)

    def requires(self):
        return {
            "buildings": InferLandUse(
                self.city, self.datapath, self.geoformat, self.date_query
            ),
            "building-parts": SetupProjection(
                self.city,
                self.datapath,
                self.geoformat,
                self.date_query,
                "building-parts",
            ),
            "pois": SetupProjection(
                self.city,
                self.datapath,
                self.geoformat,
                self.date_query,
                "pois",
            ),
        }

    def output(self):
        output_path = define_filename(
            "buildings-with-computed-land-use",
            self.city,
            dt.date(self.date_query).isoformat(),
            self.datapath,
            self.geoformat,
        )
        return luigi.LocalTarget(output_path)

    def run(self):
        proj_path = os.path.join(
            self.datapath, self.city, "utm_projection.json"
        )
        buildings = gpd.read_file(self.input()["buildings"].path)
        building_parts = gpd.read_file(self.input()["building-parts"].path)
        pois = gpd.read_file(self.input()["pois"].path)
        with open(proj_path) as fobj:
            utm_proj = json.load(fobj)
            buildings.crs = utm_proj
            building_parts.crs = utm_proj
            pois.crs = utm_proj
        associate_structures(
            buildings,
            building_parts,
            operation="contains",
            column="containing_parts",
        )
        associate_structures(
            buildings, pois, operation="intersects", column="containing_poi"
        )
        buildings["activity_category"] = buildings.apply(
            lambda x: classify_activity_category(x.key_value), axis=1
        )
        building_parts["activity_category"] = building_parts.apply(
            lambda x: classify_activity_category(x.key_value), axis=1
        )
        pois["activity_category"] = pois.apply(
            lambda x: classify_activity_category(x.key_value), axis=1
        )
        compute_landuses_m2(
            buildings,
            building_parts,
            pois,
            default_height=self.default_height,
            meters_per_level=self.meters_per_level,
            mixed_building_first_floor_activity=True,
        )
        buildings.loc[
            buildings.activity_category.apply(lambda x: len(x) == 0),
            "activity_category",
        ] = np.nan
        building_parts.loc[
            building_parts.activity_category.apply(lambda x: len(x) == 0),
            "activity_category",
        ] = np.nan
        pois.loc[
            pois.activity_category.apply(lambda x: len(x) == 0),
            "activity_category",
        ] = np.nan
        # Set the composed classification given, for each building,
        # its containing Points of Interest and building parts classification
        buildings.loc[
            buildings.apply(
                lambda x: x.landuses_m2["activity"] > 0
                and x.landuses_m2["residential"] > 0,
                axis=1,
            ),
            "classification",
        ] = "mixed"
        clean_list_in_geodataframe_column(buildings, "containing_parts")
        clean_list_in_geodataframe_column(buildings, "containing_poi")
        clean_list_in_geodataframe_column(buildings, "activity_category")
        buildings.to_file(self.output().path, driver="GeoJSON")


class GetRouteGraph(luigi.Task):
    """Retrieve routing graph for the given city, through its encompassing
    bounding box

    Example:
    ```
    python -m luigi --local-scheduler --module urbansprawl.tasks GetRouteGraph
    --city valence-drome
    ```

    Attributes
    ----------
    city : str
        City of interest
    datapath : str
        Indicates the folder where the task result has to be serialized
    geoformat : str
        Output file extension (by default: `GeoJSON`)
    date_query : str
        Date to which the OpenStreetMap data must be recovered (format:
    AAAA-MM-DDThhmm)
    """

    city = luigi.Parameter()
    datapath = luigi.Parameter("./data")
    geoformat = luigi.Parameter("geojson")
    date_query = luigi.DateMinuteParameter(default=date.today())

    def requires(self):
        return GetBoundingBox(self.city, self.datapath)

    def output(self):
        output_path = os.path.join(
            self.datapath, self.city + "_network.graphml"
        )
        return luigi.LocalTarget(output_path)

    def run(self):
        city_gdf = gpd.read_file(self.input().path)
        north, south, east, west = city_gdf.loc[
            0, ["bbox_north", "bbox_south", "bbox_east", "bbox_west"]
        ]
        date = "[date:'" + str(self.date_query) + "']"
        retrieve_route_graph(
            self.city,
            date=date,
            north=north,
            south=south,
            east=east,
            west=west,
        )


class GetIndiceGrid(luigi.Task):
    """Get the indice grid, regarding the area of interest

    Example
    -------
    ```
    python -m luigi --local-scheduler --module urbansprawl.tasks MasterTask
    --city valence-drome --date-query 2017-01-01T1200
    ̀``

    Attributes
    ----------
    city : str
        City of interest
    datapath : str
        Indicates the folder where the task result has to be serialized
    geoformat : str
        Output file extension (by default: `GeoJSON`)
    AAAA-MM-DDThhmm)
    step : int
        Distance (in meters) between each grid structuring points

    """

    city = luigi.Parameter()
    datapath = luigi.Parameter("./data")
    geoformat = luigi.Parameter("geojson")
    step = luigi.IntParameter(default=400)

    def requires(self):
        return GetBoundingBox(self.city, self.datapath)

    def output(self):
        output_path = define_filename(
            "indice-grid", self.city, self.step, self.datapath, self.geoformat
        )
        return luigi.LocalTarget(output_path)

    def run(self):
        city_gdf = gpd.read_file(self.input().path)
        bbox = osmnx.project_gdf(city_gdf).total_bounds
        indices = get_indices_grid_from_bbox(bbox, self.step)
        indices.to_file(self.output().path, driver="GeoJSON")


class ComputeGridLandUseMix(luigi.Task):
    """Compute land use mix indice values for a specific city on a set of
    grided points

    Attributes
    ----------
    city : str
        City of interest
    datapath : str
        Indicates the folder where the task result has to be serialized
    geoformat : str
        Output file extension (by default: `GeoJSON`)
    date_query : str
        Date to which the OpenStreetMap data must be recovered (format:
    AAAA-MM-DDThhmm)
    step : int
        Distance (in meters) between each grid structuring points
    walkable_distance : int
            the bandwidth assumption for Kernel Density Estimation calculations
    (meters)
    compute_activity_types_kde : bool
            determines if the densities for each activity type should be
    computed
    weighted_kde : bool
            use Weighted Kernel Density Estimation or classic version
    pois_weight : int
            Points of interest weight equivalence with buildings (squared
    meter)
    log_weighted : bool
            apply natural logarithmic function to surface weights

    """

    city = luigi.Parameter()
    datapath = luigi.Parameter("./data")
    geoformat = luigi.Parameter("geojson")
    date_query = luigi.DateMinuteParameter(default=date.today())
    step = luigi.IntParameter(default=400)
    default_height = luigi.IntParameter(3)
    meters_per_level = luigi.IntParameter(3)
    walkable_distance = luigi.IntParameter(600)
    compute_activity_types_kd = luigi.BoolParameter()
    weighted_kde = luigi.BoolParameter()
    pois_weights = luigi.IntParameter(9)
    log_weighted = luigi.BoolParameter()

    def requires(self):
        return {
            "grid": GetIndiceGrid(
                self.city, self.datapath, self.geoformat, self.step
            ),
            "buildings": ComputeLandUse(
                self.city,
                self.datapath,
                self.geoformat,
                self.date_query,
                self.default_height,
                self.meters_per_level,
            ),
            "pois": SetupProjection(
                self.city,
                self.datapath,
                self.geoformat,
                self.date_query,
                "pois",
            ),
        }

    def output(self):
        data_ident = (
            str(self.step) + "-" + dt.date(self.date_query).isoformat()
        )
        output_path = define_filename(
            "gridded-indices-land-use",
            self.city,
            data_ident,
            self.datapath,
            self.geoformat,
        )
        return luigi.LocalTarget(output_path)

    def run(self):
        grid = gpd.read_file(self.input()["grid"].path)
        buildings = gpd.read_file(self.input()["buildings"].path)
        pois = gpd.read_file(self.input()["pois"].path)
        landusemix_args = {
            "walkable_distance": self.walkable_distance,
            "compute_activity_types_kde": self.compute_activity_types_kd,
            "weighted_kde": self.weighted_kde,
            "pois_weight": self.pois_weights,
            "log_weighted": self.log_weighted,
        }
        compute_grid_landusemix(grid, buildings, pois, landusemix_args)
        grid.to_file(self.output().path, driver="GeoJSON")


class PlotLandUseMix(luigi.Task):
    """Plot land use mix indices, depending on the gridded land use mix
    analysis undertaken in `city` area, at date `date_query`

    Attributes
    ----------
    city : str
        City of interest
    datapath : str
        Indicates the folder where the task result has to be serialized
    geoformat : str
        Output file extension (by default: `GeoJSON`)
    date_query : str
        Date to which the OpenStreetMap data must be recovered (format:
    AAAA-MM-DDThhmm)
    step : int
        Distance (in meters) between each grid structuring points
    walkable_distance : int
            the bandwidth assumption for Kernel Density Estimation calculations
    (meters)
    compute_activity_types_kde : bool
            determines if the densities for each activity type should be
    computed
    weighted_kde : bool
            use Weighted Kernel Density Estimation or classic version
    pois_weight : int
            Points of interest weight equivalence with buildings (squared
    meter)
    log_weighted : bool
            apply natural logarithmic function to surface weights
    plotted_feature : str
        Dimension to plot, either `activity_pdf`, `residential_pdf`,
    `landusemix` or `landuse_intensity`. May also be
    `commercial/industrial_pdf`, `shop_pdf` or `leisure/amenity_pdf` if
    detailed activity have been required (`compute_activity_types_kd` is True).

    """

    city = luigi.Parameter()
    datapath = luigi.Parameter("./data")
    geoformat = luigi.Parameter("geojson")
    date_query = luigi.DateMinuteParameter(default=date.today())
    step = luigi.IntParameter(default=400)
    default_height = luigi.IntParameter(3)
    meters_per_level = luigi.IntParameter(3)
    walkable_distance = luigi.IntParameter(600)
    compute_activity_types_kd = luigi.BoolParameter()
    weighted_kde = luigi.BoolParameter()
    pois_weights = luigi.IntParameter(9)
    log_weighted = luigi.BoolParameter()
    plotted_feature = luigi.Parameter()
    figsize = luigi.IntParameter(8)

    def requires(self):
        return {
            "grid": ComputeGridLandUseMix(
                self.city,
                self.datapath,
                self.geoformat,
                self.date_query,
                self.step,
                self.default_height,
                self.meters_per_level,
                self.walkable_distance,
                self.compute_activity_types_kd,
                self.weighted_kde,
                self.pois_weights,
                self.log_weighted,
            ),
            "graph": GetRouteGraph(
                self.city, self.datapath, self.geoformat, self.date_query
            ),
        }

    def output(self):
        data_ident = (
            str(self.step) + "-" + dt.date(self.date_query).isoformat()
        )
        output_path = define_filename(
            "gridded-indices-land-use-" + f"{self.plotted_feature}",
            self.city,
            data_ident,
            self.datapath,
            "png",
        )
        return luigi.LocalTarget(output_path)

    def run(self):
        grid_land_use = gpd.read_file(self.input()["grid"].path)
        valid_features = grid_land_use.columns.tolist()
        valid_features.remove("geometry")
        if self.plotted_feature not in valid_features:
            raise ValueError(
                "Choose a valid feature to plot amongst" f" {valid_features}"
            )
        graph = osmnx.load_graphml(self.input()["graph"].path, folder="")
        fig, ax = osmnx.plot_graph(
            graph,
            fig_height=self.figsize,
            fig_width=self.figsize,
            close=False,
            show=False,
            edge_color="black",
            edge_alpha=0.3,
            node_alpha=0.1,
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        ax.set_title(
            f"{self.plotted_feature} kernel density (0: low, 1: high)"
        )
        grid_land_use.plot(
            self.plotted_feature,
            cmap="YlOrRd",
            ax=ax,
            cax=cax,
            legend=True,
            vmin=0,
            vmax=1,
        )
        fig.tight_layout()
        fig.savefig(self.output().path)


class ComputeGridAccessibility(luigi.Task):
    """

    Attributes
    ----------
    city : str
        City of interest
    datapath : str
        Indicates the folder where the task result has to be serialized
    geoformat : str
        Output file extension (by default: `GeoJSON`)
    date_query : str
        Date to which the OpenStreetMap data must be recovered (format:
    AAAA-MM-DDThhmm)
    step : int
        Distance (in meters) between each grid structuring points
    walkable_distance : int
            the bandwidth assumption for Kernel Density Estimation calculations
    (meters)
    compute_activity_types_kde : bool
            determines if the densities for each activity type should be
    computed
    weighted_kde : bool
            use Weighted Kernel Density Estimation or classic version
    pois_weight : int
            Points of interest weight equivalence with buildings (squared
    meter)
    log_weighted : bool
            apply natural logarithmic function to surface weights

    """

    city = luigi.Parameter()
    datapath = luigi.Parameter("./data")
    geoformat = luigi.Parameter("geojson")
    date_query = luigi.DateMinuteParameter(default=date.today())
    step = luigi.IntParameter(default=400)
    default_height = luigi.IntParameter(3)
    meters_per_level = luigi.IntParameter(3)
    fixed_distance = luigi.BoolParameter()  # True
    fixed_activities = luigi.BoolParameter()  # False
    max_edge_length = luigi.IntParameter(200)
    max_node_distance = luigi.IntParameter(250)
    fixed_distance_max_travel_distance = luigi.IntParameter(2000)
    fixed_distance_max_num_activities = luigi.IntParameter(250)
    fixed_activities_min_number = luigi.IntParameter(20)

    def requires(self):
        return {
            "grid": GetIndiceGrid(
                self.city, self.datapath, self.geoformat, self.step
            ),
            "buildings": ComputeLandUse(
                self.city,
                self.datapath,
                self.geoformat,
                self.date_query,
                self.default_height,
                self.meters_per_level,
            ),
            "pois": SetupProjection(
                self.city,
                self.datapath,
                self.geoformat,
                self.date_query,
                "pois",
            ),
            "graph": GetRouteGraph(
                self.city, self.datapath, self.geoformat, self.date_query
            ),
        }

    def output(self):
        data_ident = (
            str(self.step) + "-" + dt.date(self.date_query).isoformat()
        )
        output_path = define_filename(
            "gridded-indices-accessibility",
            self.city,
            data_ident,
            self.datapath,
            self.geoformat,
        )
        return luigi.LocalTarget(output_path)

    def run(self):
        grid = gpd.read_file(self.input()["grid"].path)
        graph = osmnx.load_graphml(self.input()["graph"].path, folder="")
        buildings = gpd.read_file(self.input()["buildings"].path)
        pois = gpd.read_file(self.input()["pois"].path)
        accessibility_args = {
            "fixed_distance": self.fixed_distance,
            "fixed_activities": self.fixed_activities,
            "max_edge_length": self.max_edge_length,
            "max_node_distance": self.max_node_distance,
            "fixed_distance_max_travel_distance": self.fixed_distance_max_travel_distance,
            "fixed_distance_max_num_activities": self.fixed_distance_max_num_activities,
            "fixed_activities_min_number": self.fixed_activities_min_number,
        }
        compute_grid_accessibility(
            grid, graph, buildings, pois, accessibility_args
        )
        grid.to_file(self.output().path, driver="GeoJSON")


class PlotAccessibility(luigi.Task):
    """Plot an accessibility indice, depending on the gridded land use mix
    analysis undertaken in `city` area, at date `date_query`

    Attributes
    ----------
    city : str
        City of interest
    datapath : str
        Indicates the folder where the task result has to be serialized
    geoformat : str
        Output file extension (by default: `GeoJSON`)
    date_query : str
        Date to which the OpenStreetMap data must be recovered (format:
    AAAA-MM-DDThhmm)
    step : int
        Distance (in meters) between each grid structuring points
    walkable_distance : int
            the bandwidth assumption for Kernel Density Estimation calculations
    (meters)
    compute_activity_types_kde : bool
            determines if the densities for each activity type should be
    computed
    weighted_kde : bool
            use Weighted Kernel Density Estimation or classic version
    pois_weight : int
            Points of interest weight equivalence with buildings (squared
    meter)
    log_weighted : bool
            apply natural logarithmic function to surface weights

    """

    city = luigi.Parameter()
    datapath = luigi.Parameter("./data")
    geoformat = luigi.Parameter("geojson")
    date_query = luigi.DateMinuteParameter(default=date.today())
    step = luigi.IntParameter(default=400)
    default_height = luigi.IntParameter(3)
    meters_per_level = luigi.IntParameter(3)
    fixed_distance = luigi.BoolParameter()  # True
    fixed_activities = luigi.BoolParameter()  # False
    max_edge_length = luigi.IntParameter(200)
    max_node_distance = luigi.IntParameter(250)
    fixed_distance_max_travel_distance = luigi.IntParameter(2000)
    fixed_distance_max_num_activities = luigi.IntParameter(250)
    fixed_activities_min_number = luigi.IntParameter(20)
    figsize = luigi.IntParameter(8)

    def requires(self):
        return {
            "grid": ComputeGridAccessibility(
                self.city,
                self.datapath,
                self.geoformat,
                self.date_query,
                self.step,
                self.default_height,
                self.meters_per_level,
                self.fixed_distance,
                self.fixed_activities,
                self.max_edge_length,
                self.max_node_distance,
                self.fixed_distance_max_travel_distance,
                self.fixed_distance_max_num_activities,
                self.fixed_activities_min_number,
            ),
            "graph": GetRouteGraph(
                self.city, self.datapath, self.geoformat, self.date_query
            ),
        }

    def output(self):
        data_ident = (
            str(self.step) + "-" + dt.date(self.date_query).isoformat()
        )
        output_path = define_filename(
            "gridded-indices-accessibility",
            self.city,
            data_ident,
            self.datapath,
            "png",
        )
        return luigi.LocalTarget(output_path)

    def run(self):
        grid_accessibility = gpd.read_file(self.input()["grid"].path)
        graph = osmnx.load_graphml(self.input()["graph"].path, folder="")
        fig, ax = osmnx.plot_graph(
            graph,
            fig_height=self.figsize,
            fig_width=self.figsize,
            close=False,
            show=False,
            edge_color="black",
            edge_alpha=0.3,
            node_alpha=0.1,
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        ax.set_title(
            "Accessibility (measured as the number of accessible POIs)"
        )
        grid_accessibility.plot(
            "accessibility",
            cmap="YlOrRd",
            ax=ax,
            cax=cax,
            legend=True,
            vmin=0,
            vmax=self.fixed_distance_max_num_activities,
        )
        fig.tight_layout()
        fig.savefig(self.output().path)


class ComputeGridDispersion(luigi.Task):
    """Compute land use mix indice values for a specific city on a set of
    grided points

    Attributes
    ----------
    city : str
        City of interest
    datapath : str
        Indicates the folder where the task result has to be serialized
    geoformat : str
        Output file extension (by default: `GeoJSON`)
    date_query : str
        Date to which the OpenStreetMap data must be recovered (format:
    AAAA-MM-DDThhmm)
    step : int
        Distance (in meters) between each grid structuring points
    walkable_distance : int
            the bandwidth assumption for Kernel Density Estimation calculations
    (meters)
    compute_activity_types_kde : bool
            determines if the densities for each activity type should be
    computed
    weighted_kde : bool
            use Weighted Kernel Density Estimation or classic version
    pois_weight : int
            Points of interest weight equivalence with buildings (squared
    meter)
    log_weighted : bool
            apply natural logarithmic function to surface weights

    """

    city = luigi.Parameter()
    datapath = luigi.Parameter("./data")
    geoformat = luigi.Parameter("geojson")
    date_query = luigi.DateMinuteParameter(default=date.today())
    step = luigi.IntParameter(default=400)
    default_height = luigi.IntParameter(3)
    meters_per_level = luigi.IntParameter(3)
    radius_search = luigi.IntParameter(750)
    use_median = luigi.BoolParameter()  # False
    K_nearest = luigi.IntParameter(50)

    def requires(self):
        return {
            "grid": GetIndiceGrid(
                self.city, self.datapath, self.geoformat, self.step
            ),
            "buildings": ComputeLandUse(
                self.city,
                self.datapath,
                self.geoformat,
                self.date_query,
                self.default_height,
                self.meters_per_level,
            ),
        }

    def output(self):
        data_ident = (
            str(self.step) + "-" + dt.date(self.date_query).isoformat()
        )
        output_path = define_filename(
            "gridded-indices-dispersion",
            self.city,
            data_ident,
            self.datapath,
            self.geoformat,
        )
        return luigi.LocalTarget(output_path)

    def run(self):
        grid = gpd.read_file(self.input()["grid"].path)
        buildings = gpd.read_file(self.input()["buildings"].path)
        dispersion_args = {
            "radius_search": self.radius_search,
            "use_median": self.use_median,
            "K_nearest": self.K_nearest,
        }
        compute_grid_dispersion(grid, buildings, dispersion_args)
        grid.to_file(self.output().path, driver="GeoJSON")


class PlotDispersion(luigi.Task):
    """Plot a dispersion indice, depending on the gridded land use mix
    analysis undertaken in `city` area, at date `date_query`

    Attributes
    ----------
    city : str
        City of interest
    datapath : str
        Indicates the folder where the task result has to be serialized
    geoformat : str
        Output file extension (by default: `GeoJSON`)
    date_query : str
        Date to which the OpenStreetMap data must be recovered (format:
    AAAA-MM-DDThhmm)
    step : int
        Distance (in meters) between each grid structuring points
    walkable_distance : int
            the bandwidth assumption for Kernel Density Estimation calculations
    (meters)
    compute_activity_types_kde : bool
            determines if the densities for each activity type should be
    computed
    weighted_kde : bool
            use Weighted Kernel Density Estimation or classic version
    pois_weight : int
            Points of interest weight equivalence with buildings (squared
    meter)
    log_weighted : bool
            apply natural logarithmic function to surface weights

    """

    city = luigi.Parameter()
    datapath = luigi.Parameter("./data")
    geoformat = luigi.Parameter("geojson")
    date_query = luigi.DateMinuteParameter(default=date.today())
    step = luigi.IntParameter(default=400)
    default_height = luigi.IntParameter(3)
    meters_per_level = luigi.IntParameter(3)
    radius_search = luigi.IntParameter(750)
    use_median = luigi.BoolParameter()  # False
    K_nearest = luigi.IntParameter(50)
    figsize = luigi.IntParameter(8)

    def requires(self):
        return {
            "grid": ComputeGridDispersion(
                self.city,
                self.datapath,
                self.geoformat,
                self.date_query,
                self.step,
                self.default_height,
                self.meters_per_level,
                self.radius_search,
                self.use_median,
                self.K_nearest,
            ),
            "graph": GetRouteGraph(
                self.city, self.datapath, self.geoformat, self.date_query
            ),
        }

    def output(self):
        data_ident = (
            str(self.step) + "-" + dt.date(self.date_query).isoformat()
        )
        output_path = define_filename(
            "gridded-indices-dispersion",
            self.city,
            data_ident,
            self.datapath,
            "png",
        )
        return luigi.LocalTarget(output_path)

    def run(self):
        grid_dispersion = gpd.read_file(self.input()["grid"].path)
        graph = osmnx.load_graphml(self.input()["graph"].path, folder="")
        fig, ax = osmnx.plot_graph(
            graph,
            fig_height=self.figsize,
            fig_width=self.figsize,
            close=False,
            show=False,
            edge_color="black",
            edge_alpha=0.3,
            node_alpha=0.1,
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        ax.set_title(
            "Dispersion (averaged distance between buildings, in meters)"
        )
        grid_dispersion.plot(
            "dispersion",
            cmap="YlOrRd",
            ax=ax,
            cax=cax,
            legend=True,
            vmin=0,
            vmax=10,
        )
        fig.tight_layout()
        fig.savefig(self.output().path)


class GetINSEEData(luigi.Task):
    """Download INSEE carroyed data (200m*200m resolution)

    INSEE data access:

    https://www.insee.fr/fr/statistiques/fichier/2520034/200m-carreaux-metropole.zip

    Attributes
    ----------
    datapath : str
        Path to the data folder on the file system
    """

    datapath = luigi.Parameter("./data")

    @property
    def path(self):
        return os.path.join(
            self.datapath, "insee", "200m-carreaux-metropole.zip"
        )

    @property
    def url(self):
        return ("https://www.insee.fr/fr/statistiques" +
                "/fichier/2520034/200m-carreaux-metropole.zip")

    def output(self):
        return luigi.LocalTarget(self.path, format=MixedUnicodeBytes)

    def run(self):
        with self.output().open("w") as fobj:
            resp = requests.get(self.url)
            resp.raise_for_status()
            fobj.write(resp.content)


class GetGPWData(luigi.Task):
    """Download GPW data from NASA Earthdata website. As there is an
    authentification step, a redirection must be undertaken.

    GPW data access:

    http://sedac.ciesin.columbia.edu\
    /downloads/data/gpw-v4/gpw-v4-population-count-rev10\
    /gpw-v4-population-count-rev10_2015_30_sec_tif.zip

    Attributes
    ----------
    datapath : str
        Path to the data folder on the file system
    """
    datapath = luigi.Parameter("./data")

    @property
    def filename(self):
        return "gpw_v4_population_count_rev10_2015_30_sec"

    @property
    def path(self):
        return os.path.join(self.datapath, "gpw", self.filename + ".zip")

    @property
    def url(self):
        return ("http://sedac.ciesin.columbia.edu" +
                "/downloads/data/gpw-v4/gpw-v4-population-count-rev10" +
                "/gpw-v4-population-count-rev10_2015_30_sec_tif.zip")

    def output(self):
        return luigi.LocalTarget(self.path, format=MixedUnicodeBytes)

    def run(self):
        with requests.Session() as session:
            r1 = session.request("get", self.url)  # Handle redirection
            login = config.get("credentials", "login")
            password = config.get("credentials", "pw")
            resp = requests.get(r1.url, auth=(login, password))
            resp.raise_for_status()
            with self.output().open("w") as fobj:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    fobj.write(chunk)


class UnzipData(luigi.Task):
    """Task dedicated to unzip file

    To get trace that the task has be done, the task creates a text file with
    the same same of the input zip file with the '.done' suffix. This generated
    file contains the path of the zipfile and all extracted files.

    Attributes
    ----------
    datapath : str
        Path to the data folder on the file system
    datasource : str
        Name of the data source, either 'insee' or 'gpw'
    """
    datapath = luigi.Parameter("./data")
    datasource = luigi.Parameter("insee")

    @property
    def gpw_filename(self):
        return "gpw_v4_population_count_rev10_2015_30_sec"

    @property
    def path(self):
        if self.datasource == "insee":
            return os.path.join(
                self.datapath, self.datasource, "200m-carreaux-metropole.zip"
            )
        elif self.datasource == "gpw":
            return os.path.join(
                self.datapath, self.datasource, self.gpw_filename + ".zip"
            )
        else:
            raise ValueError("Unknown data source, choose 'insee' or 'gpw'.")

    def requires(self):
        if self.datasource == "insee":
            return GetINSEEData(self.datapath)
        elif self.datasource == "gpw":
            return GetGPWData(self.datapath)
        else:
            raise ValueError("Unknown data source, choose 'insee' or 'gpw'.")

    def output(self):
        filepath = os.path.join(self.datapath, self.datasource, "unzip.done")
        return luigi.LocalTarget(filepath)

    def run(self):
        with self.output().open("w") as fobj:
            zip_ref = zipfile.ZipFile(self.path)
            fobj.write("\n".join(elt.filename for elt in zip_ref.filelist))
            fobj.write("\n")
            zip_ref.extractall(os.path.dirname(self.input().path))
            zip_ref.close()


class StoreINSEEGridAsShapefile(luigi.Task):
    """Store image labels to a database, considering that the input format is
    MapInfo. We use `ogr2ogr` program, and consider the task as accomplished
    after saving a `txt` file within insee data folder.

    Attributes
    ----------
    datapath : str
        Path towards the data on the file system

    """

    datapath = luigi.Parameter("./data")

    def requires(self):
        return UnzipData(self.datapath, "insee")

    def output(self):
        filepath = os.path.join(
            self.datapath, "insee", "200m-carreaux-metropole", "insee_car.shp"
        )
        return luigi.LocalTarget(filepath)

    def run(self):
        input_filename = os.path.join(
            self.datapath, "insee", "200m-carreaux-metropole", "car_m.mif"
        )
        ogr2ogr_args = [
            "-f",
            "ESRI Shapefile",
            self.output().path,
            input_filename,
            "-s_srs",
            "EPSG:27572",
            "-t_srs",
            "EPSG:4326",
        ]
        sh.ogr2ogr(ogr2ogr_args)


class ExtractLocalINSEEData(luigi.Task):
    """Extract INSEE data for a local area defined by the city of interest

    Attributes
    ----------
    city : str
        City of interest
    datapath : str
        Indicates the folder where the task result has to be serialized
    geoformat : str
        Output file extension (by default: `GeoJSON`)
    date_query : str
        Date to which the OpenStreetMap data must be recovered (format:
    AAAA-MM-DDThhmm)
    default_height : int
        Default building height, in meters (default: 3 meters)
    meters_per_level : int
        Default height per level, in meter (default: 3 meters)
    """

    city = luigi.Parameter()
    datapath = luigi.Parameter("./data")
    geoformat = luigi.Parameter("geojson")
    date_query = luigi.DateMinuteParameter(default=date.today())
    default_height = luigi.IntParameter(3)
    meters_per_level = luigi.IntParameter(3)

    def requires(self):
        return {
            "data": UnzipData(self.datapath, "insee"),
            "grid": StoreINSEEGridAsShapefile(self.datapath),
            "buildings": ComputeLandUse(
                self.city,
                self.datapath,
                self.geoformat,
                self.date_query,
                self.default_height,
                self.meters_per_level,
            ),
        }

    def output(self):
        filepath = os.path.join(
            self.datapath, self.city, "insee_population.geojson"
        )
        return luigi.LocalTarget(filepath)

    def run(self):
        buildings = gpd.read_file(self.input()["buildings"].path)
        proj_path = os.path.join(
            self.datapath, self.city, "utm_projection.json"
        )
        insee_data_filename = self.input()["grid"].path
        population_count_filename = os.path.join(
            self.datapath, "insee", "200m-carreaux-metropole", "car_m.dbf"
        )
        with open(proj_path) as fobj:
            buildings.crs = json.load(fobj)
        local_insee_pop = get_extract_population_data(
            city_ref=self.city,
            data_source="insee",
            pop_shapefile=insee_data_filename,
            pop_data_file=population_count_filename,
            to_crs=buildings.crs,
            polygons_gdf=buildings,
        )
        local_insee_pop.to_file(self.output().path, driver="GeoJSON")


class PlotINSEEData(luigi.Task):
    """Plot the INSEE local data for "city" on a GeoPandas choropleth; this
    data has a 200m*200m resolution

    Attributes
    ----------
    city : str
        City of interest
    datapath : str
        Indicates the folder where the task result has to be serialized
    geoformat : str
        Output file extension (by default: `GeoJSON`)
    date_query : str
        Date to which the OpenStreetMap data must be recovered (format:
    AAAA-MM-DDThhmm)
    default_height : int
        Default building height, in meters (default: 3 meters)
    meters_per_level : int
        Default height per level, in meter (default: 3 meters)
    """

    city = luigi.Parameter()
    datapath = luigi.Parameter("./data")
    geoformat = luigi.Parameter("geojson")
    date_query = luigi.DateMinuteParameter(default=date.today())
    default_height = luigi.IntParameter(3)
    meters_per_level = luigi.IntParameter(3)
    figsize = luigi.IntParameter(8)

    def requires(self):
        return {
            "population": ExtractLocalINSEEData(
                self.city,
                self.datapath,
                self.geoformat,
                self.date_query,
                self.default_height,
                self.meters_per_level,
            ),
            "graph": GetRouteGraph(
                self.city, self.datapath, self.geoformat, self.date_query
            ),
        }

    def output(self):
        filepath = os.path.join(
            self.datapath, self.city, "insee_population.png"
        )
        return luigi.LocalTarget(filepath)

    def run(self):
        population = gpd.read_file(self.input()["population"].path)
        graph = osmnx.load_graphml(self.input()["graph"].path, folder="")
        proj_path = os.path.join(
            self.datapath, self.city, "utm_projection.json"
        )
        with open(proj_path) as fobj:
            population.crs = json.load(fobj)
        fig, ax = osmnx.plot_graph(
            graph,
            fig_height=self.figsize,
            fig_width=self.figsize,
            close=False,
            show=False,
            edge_color="black",
            edge_alpha=0.15,
            node_alpha=0.05,
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        population.plot(
            "pop_count", ax=ax, cax=cax, cmap="YlOrRd", legend=True, vmin=0
        )
        ax.set_title("INSEE gridded population (in inhabitants)", fontsize=15)
        fig.tight_layout()
        fig.savefig(self.output().path)


class ExtractLocalGPWData(luigi.Task):
    """The raw GPW data is a tif image representing the world population; this
    class uses gdal_translate Python binding for reducing the tif only to the
    area of interest, following a provided set of coordinates

    Attributes
    ----------
    datapath : str
        Indicates the folder where the task result has to be serialized
    city : str
        Place of interest
    """
    datapath = luigi.Parameter("./data")
    city = luigi.Parameter()

    @property
    def filename(self):
        return "gpw_v4_population_count_rev10_2015_30_sec"

    def requires(self):
        return {
            "gpw": UnzipData(self.datapath, "gpw"),
            "bbox": GetBoundingBox(self.city, self.datapath),
        }

    def output(self):
        filepath = os.path.join(
            self.datapath, self.city, "gpw_raster.tif"
        )
        return luigi.LocalTarget(filepath)

    def run(self):
        city_gdf = gpd.read_file(self.input()["bbox"].path)
        north, south, east, west = city_gdf.loc[
            0, ["bbox_north", "bbox_south", "bbox_east", "bbox_west"]
        ]
        gdal.Translate(
            self.output().path,
            os.path.join(self.datapath, "gpw", self.filename + ".tif"),
            projWin=[west, north, east, south],
        )


class GridGPW(luigi.Task):
    """Extract a gridded version of local GPW data, that relies on pixel
    centroids

    Each pixel of the input raster is represented by its centroid, hence the
    corresponding population data is associated to Point in a vectorized
    version of the GPW data, that is lighter than the Polygon rasterization

    The output points are expressed in the same projection than the input data
    (GPW), hence EPSG:4326.

    Attributes
    ----------
    city : str
        City of interest
    datapath : str
        Path of the data on the file system
    """
    datapath = luigi.Parameter("./data")
    city = luigi.Parameter()

    def requires(self):
        return ExtractLocalGPWData(self.datapath, self.city)

    def output(self):
        filepath = os.path.join(
            self.datapath, self.city, "gpw_grid.geojson"
        )
        return luigi.LocalTarget(filepath)

    def run(self):
        raster = gdal.Open(self.input().path)
        raster_data = raster.ReadAsArray()
        geotransform = raster.GetGeoTransform()
        points = []
        xoffset, yoffset = geotransform[1], geotransform[5]
        for x in range(raster.RasterXSize):
            for y in range(raster.RasterYSize):
                p = Point(
                    (
                        geotransform[0] + (x+0.5) * xoffset,
                        geotransform[3] + (y+0.5) * yoffset
                    )
                )
                points.append(
                    {"pop_count": raster_data[y, x], "geometry": p}
                )
        gpd.GeoDataFrame(points).to_file(self.output().path, driver="GeoJSON")


class ExpandGrid(luigi.Task):
    """Expand a simple grid of georeferenced points by an integer factor

    For instance, with a factor 2, each point will be replaced by 2*2=4 new
    points spread over the initial point area.

    Each new point is linked with the initial point index for further
    identification purpose.

    Attributes
    ----------
    city : str
        City of interest
    datapath : str
        Path of the data on the file system
    factor : int
        Expansion factor, that gives the number of points in a single cell when
    squared
    """
    datapath = luigi.Parameter("./data")
    city = luigi.Parameter()
    factor = luigi.IntParameter(5)

    def requires(self):
        return {
            "grid": GridGPW(self.datapath, self.city),
            "raster": ExtractLocalGPWData(self.datapath, self.city)
        }

    def output(self):
        os.makedirs(os.path.join(self.datapath, "inference"), exist_ok=True)
        filepath = os.path.join(
            self.datapath, self.city,
            "gpw_expanded_grid_" + str(self.factor) + ".geojson"
        )
        return luigi.LocalTarget(filepath)

    def run(self):
        grid = gpd.read_file(self.input()["grid"].path)
        raster = gdal.Open(self.input()["raster"].path)
        geotransform = raster.GetGeoTransform()
        xoffset, yoffset = geotransform[1], geotransform[5]
        points = []
        for idx, point in grid.iterrows():
            new_points = geometries.expand_grid_point(
                point["geometry"], self.factor, xoffset, yoffset
            )
            for newp in new_points:
                points.append({"idx": idx, "geometry": newp})
        gdf_points = gpd.GeoDataFrame(points)
        gdf_points.to_file(self.output().path, driver="GeoJSON")


class CreateInferenceGrid(luigi.Task):
    """Create a set of georeferenced points starting from the GPW data where to
    compute the urbansprawl features. Such data may be taken as the input for
    the convolutional neural network that predicts population estimates.

    Attributes
    ----------
    datapath : str
        Path of the data on the file system
    city : str
        City of interest
    factor : int
        Expansion factor, that gives the number of points in a single cell when
    squared
    """
    datapath = luigi.Parameter("./data")
    city = luigi.Parameter()
    factor = luigi.IntParameter(5)

    def requires(self):
        return {
            "grid": ExpandGrid(self.datapath, self.city, self.factor),
            "raster": ExtractLocalGPWData(self.datapath, self.city)
         }

    def output(self):
        os.makedirs(os.path.join(self.datapath, "inference"), exist_ok=True)
        filepath = os.path.join(
            self.datapath, self.city,
            "gpw_squares_" + str(self.factor) + ".geojson"
        )
        return luigi.LocalTarget(filepath)

    def run(self):
        grid = gpd.read_file(self.input()["grid"].path)
        raster = gdal.Open(self.input()["raster"].path)
        xoffset = raster.GetGeoTransform()[1] / self.factor
        yoffset = raster.GetGeoTransform()[5] / self.factor
        cells = []
        for idx, point in grid.iterrows():
            px, py = point["geometry"].x, point["geometry"].y
            cell = Polygon(
                ((px - 0.5 * xoffset, py - 0.5 * yoffset),
                 (px + 0.5 * xoffset, py - 0.5 * yoffset),
                 (px + 0.5 * xoffset, py + 0.5 * yoffset),
                 (px - 0.5 * xoffset, py + 0.5 * yoffset),
                 (px - 0.5 * xoffset, py - 0.5 * yoffset)
                )
            )
            cells.append({"idx": point["idx"], "geometry": cell})
        cells = gpd.GeoDataFrame(cells)
        cells.crs = {"init": "EPSG:4326"}
        proj_path = os.path.join(
            self.datapath, self.city, "utm_projection.json"
        )
        with open(proj_path) as f:
            cells.to_crs(json.load(f), inplace=True)
        cells.to_file(self.output().path, driver="GeoJSON")


class VectorizeLocalGPWData(luigi.Task):
    """Transform local GPW data as a geojson file in order to process the data

    See GDAL documentation at:

    https://pcjericks.github.io/py-gdalogr-cookbook/\
    raster_layers.html#polygonize-a-raster-band

    Attributes
    ----------
    datapath : str
        Indicates the folder where the task result has to be serialized
    city : str
        Place of interest
    """

    datapath = luigi.Parameter("./data")
    city = luigi.Parameter()

    def requires(self):
        return ExtractLocalGPWData(self.datapath, self.city)

    def output(self):
        filepath = os.path.join(
            self.datapath, self.city, "gpw_vectorized.geojson"
        )
        return luigi.LocalTarget(filepath)

    def run(self):
        src_ds = gdal.Open(self.input().path)
        band = src_ds.GetRasterBand(1)
        maskband = band.GetMaskBand()
        driver = ogr.GetDriverByName("GeoJSON")
        dst_ds = driver.CreateDataSource(self.output().path)
        srs = osr.SpatialReference()
        srs.ImportFromWkt(src_ds.GetProjectionRef())
        dst_layer = dst_ds.CreateLayer(
            "out_layer", geom_type=ogr.wkbPolygon, srs=srs
        )
        fd = ogr.FieldDefn("pop_count", ogr.OFTInteger)
        dst_layer.CreateField(fd)
        dst_field = 0
        gdal.Polygonize(
            band,
            maskband,
            dst_layer,
            dst_field,
            [],
            callback=gdal.TermProgress,
        )
        band = maskband = src_ds = dst_ds = None


class PlotGPWData(luigi.Task):
    """Plot the GPW local data for "city" on a GeoPandas choropleth; this
    data has a 1km*1km resolution

    Attributes
    ----------
    city : str
        City of interest
    datapath : str
        Indicates the folder where the task result has to be serialized
    geoformat : str
        Output file extension (by default: `GeoJSON`)
    date_query : str
        Date to which the OpenStreetMap data must be recovered (format:
    AAAA-MM-DDThhmm)
    """

    city = luigi.Parameter()
    datapath = luigi.Parameter("./data")
    geoformat = luigi.Parameter("geojson")
    date_query = luigi.DateMinuteParameter(default=date.today())
    figsize = luigi.IntParameter(8)
    scale = luigi.ChoiceParameter(choices=["coarse", "fine"])
    training_cities = luigi.ListParameter([])
    validation_cities = luigi.ListParameter([])
    default_height = luigi.IntParameter(3)
    meters_per_level = luigi.IntParameter(3)
    walkable_distance = luigi.IntParameter(600)
    compute_activity_types_kd = luigi.BoolParameter()
    weighted_kde = luigi.BoolParameter()
    pois_weights = luigi.IntParameter(9)
    log_weighted = luigi.BoolParameter()
    radius_search = luigi.IntParameter(750)
    use_median = luigi.BoolParameter()  # False
    K_nearest = luigi.IntParameter(50)
    batch_size = luigi.IntParameter(32)
    epochs = luigi.IntParameter(50)

    def requires(self):
        if self.scale == "coarse":
            pop_data = VectorizeLocalGPWData(self.datapath, self.city)
        elif self.scale == "fine":
            pop_data = DownscaleGPWPopulationEstimates(
                self.city,
                self.training_cities,
                self.validation_cities,
                self.datapath,
                self.geoformat,
                self.date_query,
                self.default_height,
                self.meters_per_level,
                self.walkable_distance,
                self.compute_activity_types_kd,
                self.weighted_kde,
                self.pois_weights,
                self.log_weighted,
                self.radius_search,
                self.use_median,
                self.K_nearest
            )
        else:
            raise ValueError("Unknown scale, enter either 'coarse' or 'fine'.")
        return {
            "population": pop_data,
            "graph": GetRouteGraph(
                self.city, self.datapath, self.geoformat, self.date_query
            ),
            "proj": SetupProjection(
                self.city,
                self.datapath,
                self.geoformat,
                self.date_query,
                "pois",
            ),
        }

    def output(self):
        filepath = os.path.join(
            self.datapath, self.city, "gpw_population_" + self.scale + ".png"
        )
        return luigi.LocalTarget(filepath)

    def run(self):
        proj_path = os.path.join(
            self.datapath, self.city, "utm_projection.json"
        )
        with open(proj_path) as f:
            utm_proj = json.load(f)
        population = gpd.read_file(self.input()["population"].path)
        population.columns = ["pop_count", "geometry"]
        if self.scale == "coarse":
            population.crs = {"init": "epsg:4326"}
            population.to_crs(utm_proj, inplace=True)
        elif self.scale == "fine":
            population.crs = utm_proj
        else:
            raise ValueError("Unknown scale, choose either 'coarse' or 'fine'")
        graph = osmnx.load_graphml(self.input()["graph"].path, folder="")
        fig, ax = osmnx.plot_graph(
            graph,
            fig_height=self.figsize,
            fig_width=self.figsize,
            close=False,
            show=False,
            edge_color="black",
            edge_alpha=0.15,
            node_alpha=0.05,
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        population.plot(
            "pop_count", ax=ax, cax=cax, cmap="YlOrRd", legend=True, vmin=0
        )
        ax.set_title("GPW gridded population (in inhabitants)", fontsize=15)
        fig.tight_layout()
        fig.savefig(self.output().path)


class ComputePopulationFeatures(luigi.Task):
    """Compute population downscaling training features for "city" area,
    knowing a set of urbansprawl parameters.

    The features are stored as a new geojson file into the corresponding
    repository.

    Attributes
    ----------
    city : str
        City of interest
    datapath : str
        Indicates the folder where the task result has to be serialized
    geoformat : str
        Output file extension (by default: `GeoJSON`)
    date_query : str
        Date to which the OpenStreetMap data must be recovered (format:
    AAAA-MM-DDThhmm)
    default_height : int
        Default building height, in meters (default: 3 meters)
    meters_per_level : int
        Default height per level, in meter (default: 3 meters)
    walkable_distance : int
            the bandwidth assumption for Kernel Density Estimation calculations
    (meters)
    compute_activity_types_kde : bool
            determines if the densities for each activity type should be
    computed
    weighted_kde : bool
            use Weighted Kernel Density Estimation or classic version
    pois_weight : int
            Points of interest weight equivalence with buildings (squared
    meter)
    log_weighted : bool
            apply natural logarithmic function to surface weights
    radius_search : int
    use_median : bool
    K_nearest : int
    """

    city = luigi.Parameter()
    datapath = luigi.Parameter("./data")
    geoformat = luigi.Parameter("geojson")
    date_query = luigi.DateMinuteParameter(default=date.today())
    data_source = luigi.ChoiceParameter(choices=["insee", "gpw"])
    default_height = luigi.IntParameter(3)
    meters_per_level = luigi.IntParameter(3)
    walkable_distance = luigi.IntParameter(600)
    compute_activity_types_kd = luigi.BoolParameter()
    weighted_kde = luigi.BoolParameter()
    pois_weights = luigi.IntParameter(9)
    log_weighted = luigi.BoolParameter()
    radius_search = luigi.IntParameter(750)
    use_median = luigi.BoolParameter()  # False
    K_nearest = luigi.IntParameter(50)

    def requires(self):
        if self.data_source == "insee":
            grid_task = ExtractLocalINSEEData(
                self.city,
                self.datapath,
                self.geoformat,
                self.date_query,
                self.default_height,
                self.meters_per_level,
            )
        elif self.data_source == "gpw":
            grid_task = CreateInferenceGrid(self.datapath, self.city)
        else:
            raise ValueError("Unknown data source.")
        return {
            "buildings": ComputeLandUse(
                self.city,
                self.datapath,
                self.geoformat,
                self.date_query,
                self.default_height,
                self.meters_per_level,
            ),
            "pois": SetupProjection(
                self.city,
                self.datapath,
                self.geoformat,
                self.date_query,
                "pois",
            ),
            "grid": grid_task
        }

    def output(self):
        filepath = os.path.join(
            self.datapath, self.city,
            self.data_source + "_population_features.geojson"
        )
        return luigi.LocalTarget(filepath)

    def run(self):
        buildings = gpd.read_file(self.input()["buildings"].path)
        pois = gpd.read_file(self.input()["pois"].path)
        gridded_pop = gpd.read_file(self.input()["grid"].path)
        proj_path = os.path.join(
            self.datapath, self.city, "utm_projection.json"
        )
        with open(proj_path) as fobj:
            utm_proj = json.load(fobj)
            buildings.crs = utm_proj
            pois.crs = utm_proj
            gridded_pop.crs = utm_proj
        landusemix_args = {
            "walkable_distance": self.walkable_distance,
            "compute_activity_types_kde": self.compute_activity_types_kd,
            "weighted_kde": self.weighted_kde,
            "pois_weight": self.pois_weights,
            "log_weighted": self.log_weighted,
        }
        dispersion_args = {
            "radius_search": self.radius_search,
            "use_median": self.use_median,
            "K_nearest": self.K_nearest,
        }
        gdf = compute_full_urban_features(
            self.city,
            buildings,
            pois,
            gridded_pop,
            self.data_source,
            landusemix_args,
            dispersion_args,
        )
        gdf.to_file(self.output().path, driver="GeoJSON")


class SplitPopulationFeatures(luigi.Task):
    """Prepare population features in order to train a downscaling model;
    i.e. use data structure that allow to stores the inputs on the one hand
    (population density at a coarse-grained scale, urbansprawl features) and
    the output on the other hand (population density at a fine-grained scale).

    The resulting data is stored in "<datapath>/training/" so as to anticipate
    the training process.

    Attributes
    ----------
    city : str
        City of interest
    datapath : str
        Indicates the folder where the task result has to be serialized
    geoformat : str
        Output file extension (by default: `GeoJSON`)
    date_query : str
        Date to which the OpenStreetMap data must be recovered (format:
    AAAA-MM-DDThhmm)
    default_height : int
        Default building height, in meters (default: 3 meters)
    meters_per_level : int
        Default height per level, in meter (default: 3 meters)
    walkable_distance : int
            the bandwidth assumption for Kernel Density Estimation calculations
    (meters)
    compute_activity_types_kde : bool
            determines if the densities for each activity type should be
    computed
    weighted_kde : bool
            use Weighted Kernel Density Estimation or classic version
    pois_weight : int
            Points of interest weight equivalence with buildings (squared
    meter)
    log_weighted : bool
            apply natural logarithmic function to surface weights
    radius_search : int
    use_median : bool
    K_nearest : int
    """

    city = luigi.Parameter()
    datapath = luigi.Parameter("./data")
    geoformat = luigi.Parameter("geojson")
    date_query = luigi.DateMinuteParameter(default=date.today())
    data_source = luigi.ChoiceParameter(choices=["insee", "gpw"])
    default_height = luigi.IntParameter(3)
    meters_per_level = luigi.IntParameter(3)
    walkable_distance = luigi.IntParameter(600)
    compute_activity_types_kd = luigi.BoolParameter()
    weighted_kde = luigi.BoolParameter()
    pois_weights = luigi.IntParameter(9)
    log_weighted = luigi.BoolParameter()
    radius_search = luigi.IntParameter(750)
    use_median = luigi.BoolParameter()  # False
    K_nearest = luigi.IntParameter(50)

    def requires(self):
        return ComputePopulationFeatures(
            self.city,
            self.datapath,
            self.geoformat,
            self.date_query,
            self.data_source,
            self.default_height,
            self.meters_per_level,
            self.walkable_distance,
            self.compute_activity_types_kd,
            self.weighted_kde,
            self.pois_weights,
            self.log_weighted,
            self.radius_search,
            self.use_median,
            self.K_nearest,
        )

    def output(self):
        if self.data_source == "insee":
            filepath = os.path.join(
                self.datapath, "training", self.city + "_X_Y.npz"
            )
        elif self.data_source == "gpw":
            filepath = os.path.join(
                self.datapath, "inference",
                self.city + "_" + self.data_source + "_X.npz"
            )
        else:
            ValueError("Unknown data source.")
        return luigi.LocalTarget(filepath)

    def run(self):
        population_features = gpd.read_file(self.input().path)
        proj_path = os.path.join(
            self.datapath, self.city, "utm_projection.json"
        )
        with open(proj_path) as fobj:
            population_features.crs = json.load(fobj)
        if self.data_source == "insee":
            get_training_testing_data(self.city, population_features)
        elif self.data_source == "gpw":
            X, columns, geometries = prepare_testing_data(
                self.city, population_features
            )
            np.savez(
                self.output().path,
                X=X,
                X_columns=columns,
                geom=geometries
            )
        else:
            raise ValueError("Unknown data source.")


class TrainPopulationDownscalingModel(luigi.Task):
    """Train a population downscaling model so as automatically determine the
    population density at a fine-grained scale.

    The model is a deep convolutional neural network that use urbansprawl
    features and population density at 1km*1km scale. It predicts population
    density at 200m*200m scale.

    Every grid cell of "training_cities" is used for training, whilst every
    grid cell of "validation_cities" is used for validation. The user is
    expected to pass significant data in both sets.

    Attributes
    ----------
    city : str
        City of interest
    datapath : str
        Indicates the folder where the task result has to be serialized
    geoformat : str
        Output file extension (by default: `GeoJSON`)
    date_query : str
        Date to which the OpenStreetMap data must be recovered (format:
    AAAA-MM-DDThhmm)
    default_height : int
        Default building height, in meters (default: 3 meters)
    meters_per_level : int
        Default height per level, in meter (default: 3 meters)
    walkable_distance : int
            the bandwidth assumption for Kernel Density Estimation calculations
    (meters)
    compute_activity_types_kde : bool
            determines if the densities for each activity type should be
    computed
    weighted_kde : bool
            use Weighted Kernel Density Estimation or classic version
    pois_weight : int
            Points of interest weight equivalence with buildings (squared
    meter)
    log_weighted : bool
            apply natural logarithmic function to surface weights
    radius_search : int
    use_median : bool
    K_nearest : int
    batch_size : int
        Number of gridded sample to consider in each batch of data (must be
    small if limited computing resources)
    epochs : int
        Number of times the data are exploited during training process
    """

    training_cities = luigi.ListParameter()
    validation_cities = luigi.ListParameter()
    datapath = luigi.Parameter("./data")
    geoformat = luigi.Parameter("geojson")
    date_query = luigi.DateMinuteParameter(default=date.today())
    default_height = luigi.IntParameter(3)
    meters_per_level = luigi.IntParameter(3)
    walkable_distance = luigi.IntParameter(600)
    compute_activity_types_kd = luigi.BoolParameter()
    weighted_kde = luigi.BoolParameter()
    pois_weights = luigi.IntParameter(9)
    log_weighted = luigi.BoolParameter()
    radius_search = luigi.IntParameter(750)
    use_median = luigi.BoolParameter()  # False
    K_nearest = luigi.IntParameter(50)
    batch_size = luigi.IntParameter(32)
    epochs = luigi.IntParameter(50)

    def requires(self):
        for city in self.training_cities + self.validation_cities:
            yield SplitPopulationFeatures(
                city,
                self.datapath,
                self.geoformat,
                self.date_query,
                "insee",
                self.default_height,
                self.meters_per_level,
                self.walkable_distance,
                self.compute_activity_types_kd,
                self.weighted_kde,
                self.pois_weights,
                self.log_weighted,
                self.radius_search,
                self.use_median,
                self.K_nearest,
            )

    def output(self):
        isodate = dt.date(self.date_query).isoformat()
        filepath = os.path.join(
            self.datapath,
            "training",
            "checkpoint-step" + str(self.epochs) + "-" + isodate + ".h5",
        )
        return luigi.LocalTarget(filepath)

    def run(self):
        Y_train, X_train, _ = get_Y_X_features_population_data(
            cities_selection=self.training_cities
        )
        Y_val, X_val, _ = get_Y_X_features_population_data(
            cities_selection=self.validation_cities
        )
        hist = train_population_downscaling_model(
            X_train,
            Y_train,
            X_val,
            Y_val,
            self.batch_size,
            self.epochs,
            self.output().path,
        )
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for k in hist.history.keys():
            if "mean_absolute_error" not in k:
                continue
            label = "Validation error" if "val" in k else "Training error"
            ax.plot(hist.history[k], label=label)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(self.output().path.replace(".h5", ".png"))


class InferINSEEPopulationDownscaling(luigi.Task):
    """Estimates the population density at a fine-grained scale (200m*200m) by
    starting with coarse-grained data (1km*1km).

    Valid for INSEE data, it estimates 200m*200m gridded population. As we
    already have this information, one may compute accuracy metrics
    (e.g. RMSE).

    Attributes
    ----------
    city : str
        City of interest
    datapath : str
        Indicates the folder where the task result has to be serialized
    geoformat : str
        Output file extension (by default: `GeoJSON`)
    date_query : str
        Date to which the OpenStreetMap data must be recovered (format:
    AAAA-MM-DDThhmm)
    default_height : int
        Default building height, in meters (default: 3 meters)
    meters_per_level : int
        Default height per level, in meter (default: 3 meters)
    walkable_distance : int
            the bandwidth assumption for Kernel Density Estimation calculations
    (meters)
    compute_activity_types_kde : bool
            determines if the densities for each activity type should be
    computed
    weighted_kde : bool
            use Weighted Kernel Density Estimation or classic version
    pois_weight : int
            Points of interest weight equivalence with buildings (squared
    meter)
    log_weighted : bool
            apply natural logarithmic function to surface weights
    radius_search : int
    use_median : bool
    K_nearest : int
    batch_size : int
        Number of gridded sample to consider in each batch of data (must be
    small if limited computing resources)
    epochs : int
        Number of times the data are exploited during training process
    """

    city = luigi.Parameter()
    training_cities = luigi.ListParameter()
    validation_cities = luigi.ListParameter()
    datapath = luigi.Parameter("./data")
    geoformat = luigi.Parameter("geojson")
    date_query = luigi.DateMinuteParameter(default=date.today())
    default_height = luigi.IntParameter(3)
    meters_per_level = luigi.IntParameter(3)
    walkable_distance = luigi.IntParameter(600)
    compute_activity_types_kd = luigi.BoolParameter()
    weighted_kde = luigi.BoolParameter()
    pois_weights = luigi.IntParameter(9)
    log_weighted = luigi.BoolParameter()
    radius_search = luigi.IntParameter(750)
    use_median = luigi.BoolParameter()  # False
    K_nearest = luigi.IntParameter(50)
    batch_size = luigi.IntParameter(32)
    epochs = luigi.IntParameter(50)

    def requires(self):
        return {
            "model": TrainPopulationDownscalingModel(
                self.training_cities,
                self.validation_cities,
                self.datapath,
                self.geoformat,
                self.date_query,
                self.default_height,
                self.meters_per_level,
                self.walkable_distance,
                self.compute_activity_types_kd,
                self.weighted_kde,
                self.pois_weights,
                self.log_weighted,
                self.radius_search,
                self.use_median,
                self.K_nearest,
            ),
            "features": SplitPopulationFeatures(
                self.city,
                self.datapath,
                self.geoformat,
                self.date_query,
                "insee",
                self.default_height,
                self.meters_per_level,
                self.walkable_distance,
                self.compute_activity_types_kd,
                self.weighted_kde,
                self.pois_weights,
                self.log_weighted,
                self.radius_search,
                self.use_median,
                self.K_nearest,
            ),
        }

    def output(self):
        os.makedirs(os.path.join(self.datapath, "inference"), exist_ok=True)
        filepath = os.path.join(
            self.datapath, "inference", self.city + "_insee_Ypred.npz"
        )
        return luigi.LocalTarget(filepath)

    def run(self):
        _, X_train, _ = get_Y_X_features_population_data(
            cities_selection=self.training_cities
        )
        model = build_downscaling_cnn(X_train.shape)
        model.load_weights(self.input()["model"].path)
        data = np.load(self.input()["features"].path)
        y_pred = model.predict(data["X"])
        np.savez(self.output().path, y_pred=y_pred)


class InferGPWPopulationDownscaling(luigi.Task):
    """Estimates the population density at a fine-grained scale (200m*200m) by
    starting with coarse-grained data (1km*1km).

    Valid for GPW data, it estimates 200m*200m gridded population, however we
    do not have any ground-truth data for such an input.

    Attributes
    ----------
    city : str
        City of interest
    datapath : str
        Indicates the folder where the task result has to be serialized
    geoformat : str
        Output file extension (by default: `GeoJSON`)
    date_query : str
        Date to which the OpenStreetMap data must be recovered (format:
    AAAA-MM-DDThhmm)
    default_height : int
        Default building height, in meters (default: 3 meters)
    meters_per_level : int
        Default height per level, in meter (default: 3 meters)
    walkable_distance : int
            the bandwidth assumption for Kernel Density Estimation calculations
    (meters)
    compute_activity_types_kde : bool
            determines if the densities for each activity type should be
    computed
    weighted_kde : bool
            use Weighted Kernel Density Estimation or classic version
    pois_weight : int
            Points of interest weight equivalence with buildings (squared
    meter)
    log_weighted : bool
            apply natural logarithmic function to surface weights
    radius_search : int
    use_median : bool
    K_nearest : int
    batch_size : int
        Number of gridded sample to consider in each batch of data (must be
    small if limited computing resources)
    epochs : int
        Number of times the data are exploited during training process
    """
    city = luigi.Parameter()
    training_cities = luigi.ListParameter()
    validation_cities = luigi.ListParameter()
    datapath = luigi.Parameter("./data")
    geoformat = luigi.Parameter("geojson")
    date_query = luigi.DateMinuteParameter(default=date.today())
    default_height = luigi.IntParameter(3)
    meters_per_level = luigi.IntParameter(3)
    walkable_distance = luigi.IntParameter(600)
    compute_activity_types_kd = luigi.BoolParameter()
    weighted_kde = luigi.BoolParameter()
    pois_weights = luigi.IntParameter(9)
    log_weighted = luigi.BoolParameter()
    radius_search = luigi.IntParameter(750)
    use_median = luigi.BoolParameter()  # False
    K_nearest = luigi.IntParameter(50)
    batch_size = luigi.IntParameter(32)
    epochs = luigi.IntParameter(50)

    def requires(self):
        return {
            "model": TrainPopulationDownscalingModel(
                self.training_cities,
                self.validation_cities,
                self.datapath,
                self.geoformat,
                self.date_query,
                self.default_height,
                self.meters_per_level,
                self.walkable_distance,
                self.compute_activity_types_kd,
                self.weighted_kde,
                self.pois_weights,
                self.log_weighted,
                self.radius_search,
                self.use_median,
                self.K_nearest,
            ),
            "features": SplitPopulationFeatures(
                self.city,
                self.datapath,
                self.geoformat,
                self.date_query,
                "gpw",
                self.default_height,
                self.meters_per_level,
                self.walkable_distance,
                self.compute_activity_types_kd,
                self.weighted_kde,
                self.pois_weights,
                self.log_weighted,
                self.radius_search,
                self.use_median,
                self.K_nearest,
            ),
        }

    def output(self):
        os.makedirs(os.path.join(self.datapath, "inference"), exist_ok=True)
        filepath = os.path.join(
            self.datapath, "inference", self.city + "_gpw_Ypred.npz"
        )
        return luigi.LocalTarget(filepath)

    def run(self):
        _, X_train, _ = get_Y_X_features_population_data(
            cities_selection=self.training_cities
        )
        model = build_downscaling_cnn(X_train.shape)
        model.load_weights(self.input()["model"].path)
        data = np.load(self.input()["features"].path)
        y_pred = model.predict(data["X"])
        np.savez(self.output().path, y_pred=y_pred, geom=data["geom"])


class DownscaleGPWPopulationEstimates(luigi.Task):
    """Associate the GPW coarse-grained population estimates with the predicted
    population repartition given by the convolutional neural network

    Attributes
    ----------
    city : str
        City of interest
    datapath : str
        Indicates the folder where the task result has to be serialized
    geoformat : str
        Output file extension (by default: `GeoJSON`)
    date_query : str
        Date to which the OpenStreetMap data must be recovered (format:
    AAAA-MM-DDThhmm)
    default_height : int
        Default building height, in meters (default: 3 meters)
    meters_per_level : int
        Default height per level, in meter (default: 3 meters)
    walkable_distance : int
            the bandwidth assumption for Kernel Density Estimation calculations
    (meters)
    compute_activity_types_kde : bool
            determines if the densities for each activity type should be
    computed
    weighted_kde : bool
            use Weighted Kernel Density Estimation or classic version
    pois_weight : int
            Points of interest weight equivalence with buildings (squared
    meter)
    log_weighted : bool
            apply natural logarithmic function to surface weights
    radius_search : int
    use_median : bool
    K_nearest : int
    batch_size : int
        Number of gridded sample to consider in each batch of data (must be
    small if limited computing resources)
    epochs : int
        Number of times the data are exploited during training process
    """
    city = luigi.Parameter()
    training_cities = luigi.ListParameter()
    validation_cities = luigi.ListParameter()
    datapath = luigi.Parameter("./data")
    geoformat = luigi.Parameter("geojson")
    date_query = luigi.DateMinuteParameter(default=date.today())
    default_height = luigi.IntParameter(3)
    meters_per_level = luigi.IntParameter(3)
    walkable_distance = luigi.IntParameter(600)
    compute_activity_types_kd = luigi.BoolParameter()
    weighted_kde = luigi.BoolParameter()
    pois_weights = luigi.IntParameter(9)
    log_weighted = luigi.BoolParameter()
    radius_search = luigi.IntParameter(750)
    use_median = luigi.BoolParameter()  # False
    K_nearest = luigi.IntParameter(50)
    batch_size = luigi.IntParameter(32)
    epochs = luigi.IntParameter(50)

    def requires(self):
        return {"predictions": InferGPWPopulationDownscaling(
            self.city,
            self.training_cities,
            self.validation_cities,
            self.datapath,
            self.geoformat,
            self.date_query,
            self.default_height,
            self.meters_per_level,
            self.walkable_distance,
            self.compute_activity_types_kd,
            self.weighted_kde,
            self.pois_weights,
            self.log_weighted,
            self.radius_search,
            self.use_median,
            self.K_nearest
        ),
                "grid": GridGPW(self.datapath, self.city),
                "inference_grid": CreateInferenceGrid(self.datapath, self.city)
                }
    def output(self):
        os.makedirs(os.path.join(self.datapath, "inference"), exist_ok=True)
        filepath = os.path.join(
            self.datapath, self.city, "gpw_pred_population.geojson"
        )
        return luigi.LocalTarget(filepath)

    def run(self):
        predictions = np.load(self.input()["predictions"].path)
        proj_path = os.path.join(
            self.datapath, self.city, "utm_projection.json"
        )
        with open(proj_path) as fobj:
            utm_proj = json.load(fobj)
        grid = gpd.read_file(self.input()["grid"].path).reset_index()
        grid.to_crs(utm_proj, inplace=True)
        inference_grid = gpd.read_file(self.input()["inference_grid"].path)
        inference_grid.crs = utm_proj
        grid.drop("geometry", axis=1, inplace=True)
        grid.columns = ["idx", "coarse_pop"]
        pop_grid = pd.merge(grid, inference_grid, on="idx")
        y_preds = np.reshape(predictions["y_pred"], (-1))
        geoms = np.reshape(predictions["geom"], (-1))
        gdf = gpd.GeoDataFrame(
            {"pred": y_preds, "geometry": geoms}, crs=utm_proj
        )
        gdf_output = gpd.sjoin(gdf, pop_grid, op="contains")
        gdf_output["pop_count"] = gdf_output["coarse_pop"] * gdf_output["pred"]
        gdf_output[["geometry", "pop_count"]].to_file(
            self.output().path, driver="GeoJSON"
        )
