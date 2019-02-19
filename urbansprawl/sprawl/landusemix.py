###################################################################################################
# Repository: https://github.com/lgervasoni/urbansprawl
# MIT License
###################################################################################################

import math
import numpy as np
import pandas as pd
import time

from sklearn.neighbors.kde import KernelDensity
from .utils import WeightedKernelDensityEstimation

from osmnx import log

##############################################################
### Land use mix indices methods
##############################################################


def metric_phi_entropy(x, y):
    """ 
	Shannon's entropy metric
	Based on article "Comparing measures of urban land use mix, 2013"

	Parameters
	----------
	x : float
		probability related to land use X
	y : float
		probability related to land use Y

	Returns
	----------
	float
		entropy value
	"""
    # Undefined for negative values
    if x < 0 or y < 0:
        return np.nan
    # Entropy Index not defined for any input value equal to zero (due to logarithm)
    if x == 0 or y == 0:
        return 0
    # Sum = 1
    x_, y_ = x / (x + y), y / (x + y)
    phi_value = -((x_ * math.log(x_)) + (y_ * math.log(y_))) / math.log(2)
    return phi_value


#### Assign land use mix method
_land_use_mix = metric_phi_entropy

##############################################################
### Land use mix indices calculation
##############################################################


def compute_grid_landusemix(
    df_indices,
    df_osm_built,
    df_osm_pois,
    kw_args={
        "walkable_distance": 600,
        "compute_activity_types_kde": True,
        "weighted_kde": True,
        "pois_weight": 9,
        "log_weighted": True,
    },
):
    """ 
	Calculate land use mix indices on input grid

	Parameters
	----------
	df_indices : geopandas.GeoDataFrame
		data frame containing the (x,y) reference points to calculate indices
	df_osm_built : geopandas.GeoDataFrame
		data frame containing the building's geometries
	df_osm_pois : geopandas.GeoDataFrame
		data frame containing the points' of interest geometries
	kw_args: dict
		additional keyword arguments for the indices calculation
			walkable_distance : int
				the bandwidth assumption for Kernel Density Estimation calculations (meters)
			compute_activity_types_kde : bool
				determines if the densities for each activity type should be computed
			weighted_kde : bool
				use Weighted Kernel Density Estimation or classic version
			pois_weight : int
				Points of interest weight equivalence with buildings (squared meter)
			log_weighted : bool
				apply natural logarithmic function to surface weights

	Returns
	----------
	pandas.DataFrame
		land use mix indices
	"""
    log("Calculating land use mix indices")
    start = time.time()

    # Get the bandwidth, related to 'walkable distances'
    bandwidth = kw_args["walkable_distance"]
    # Compute a weighted KDE?
    weighted_kde = kw_args["weighted_kde"]
    X_weights = None

    # Get full list of contained POIs
    contained_pois = list(
        set(
            [
                element
                for list_ in df_osm_built.containing_poi[
                    df_osm_built.containing_poi.notnull()
                ]
                for element in list_
            ]
        )
    )
    # Get the POIs not contained by any building
    df_osm_pois_not_contained = df_osm_pois[
        ~df_osm_pois.index.isin(contained_pois)
    ]

    ############
    ### Calculate land use density estimations
    ############

    ####
    # Residential
    ####
    df_osm_built_indexed = df_osm_built[
        df_osm_built.classification.isin(["residential", "mixed"])
    ]
    if weighted_kde:
        X_weights = df_osm_built_indexed.landuses_m2.apply(
            lambda x: x["residential"]
        )

    df_indices["residential_pdf"] = calculate_kde(
        df_indices.geometry,
        df_osm_built_indexed,
        None,
        bandwidth,
        X_weights,
        kw_args["pois_weight"],
        kw_args["log_weighted"],
    )
    log("Residential density estimation done")

    ####
    # Activities
    ####
    df_osm_built_indexed = df_osm_built[
        df_osm_built.classification.isin(["activity", "mixed"])
    ]
    df_osm_pois_not_cont_indexed = df_osm_pois_not_contained[
        df_osm_pois_not_contained.classification.isin(["activity", "mixed"])
    ]
    if weighted_kde:
        X_weights = df_osm_built_indexed.landuses_m2.apply(
            lambda x: x["activity"]
        )

    df_indices["activity_pdf"] = calculate_kde(
        df_indices.geometry,
        df_osm_built_indexed,
        df_osm_pois_not_cont_indexed,
        bandwidth,
        X_weights,
        kw_args["pois_weight"],
        kw_args["log_weighted"],
    )
    log("Activity density estimation done")

    ####
    # Compute activity types densities
    ####
    if kw_args["compute_activity_types_kde"]:
        assert "activity_category" in df_osm_built.columns

        # Get unique category values
        unique_categories_built = [
            list(x)
            for x in set(
                tuple(x)
                for x in df_osm_built.activity_category.values
                if isinstance(x, list)
            )
        ]
        unique_categories_pois = [
            list(x)
            for x in set(
                tuple(x)
                for x in df_osm_pois_not_cont_indexed.activity_category.values
                if isinstance(x, list)
            )
        ]
        flat_list = [
            item
            for sublist in unique_categories_built + unique_categories_pois
            for item in sublist
        ]
        categories = list(set(flat_list))

        for cat in categories:  # Get data frame selection of input category
            # Buildings and POIs within that category
            df_built_category = df_osm_built_indexed[
                df_osm_built_indexed.activity_category.apply(
                    lambda x: (isinstance(x, list)) and (cat in x)
                )
            ]
            df_pois_category = df_osm_pois_not_cont_indexed[
                df_osm_pois_not_cont_indexed.activity_category.apply(
                    lambda x: (isinstance(x, list)) and (cat in x)
                )
            ]
            if weighted_kde:
                X_weights = df_built_category.landuses_m2.apply(
                    lambda x: x[cat]
                )

            df_indices[cat + "_pdf"] = calculate_kde(
                df_indices.geometry,
                df_built_category,
                df_pois_category,
                bandwidth,
                X_weights,
                kw_args["pois_weight"],
                kw_args["log_weighted"],
            )

        log("Activity grouped by types density estimation done")

        # Compute land use mix indices
    index_column = "landusemix"
    df_indices[index_column] = df_indices.apply(
        lambda x: _land_use_mix(x.activity_pdf, x.residential_pdf), axis=1
    )
    df_indices["landuse_intensity"] = df_indices.apply(
        lambda x: (x.activity_pdf + x.residential_pdf) / 2.0, axis=1
    )

    log(
        "Done: Land use mix indices. Elapsed time (H:M:S): "
        + time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
    )


####


def calculate_kde(
    points,
    df_osm_built,
    df_osm_pois=None,
    bandwidth=400,
    X_weights=None,
    pois_weight=9,
    log_weight=True,
):
    """
	Evaluate the probability density function using Kernel Density Estimation of input geo-localized data
	KDE's bandwidth stands for walkable distances
	If input weights are given, a Weighted Kernel Density Estimation is carried out

	Parameters
	----------
	points : geopandas.GeoSeries
		reference points to calculate indices
	df_osm_built : geopandas.GeoDataFrame
		data frame containing the building's geometries
	df_osm_pois : geopandas.GeoDataFrame
		data frame containing the points' of interest geometries
	bandwidth: int
		bandwidth value to be employed on the Kernel Density Estimation
	X_weights : pandas.Series
		indicates the weight for each input building (e.g. surface)
	pois_weight : int
		weight assigned to points of interest
	log_weight : bool
		if indicated, applies a log transformation to input weight values

	Returns
	----------
	pandas.Series
		
	"""
    # X_b : Buildings array
    X_b = [[p.x, p.y] for p in df_osm_built.geometry.centroid.values]

    # X_p : Points array
    if df_osm_pois is None:
        X_p = []
    else:
        X_p = [[p.x, p.y] for p in df_osm_pois.geometry.centroid.values]

    # X : Full array
    X = np.array(X_b + X_p)

    # Points where the probability density function will be evaluated
    Y = np.array([[p.x, p.y] for p in points.values])

    if not (X_weights is None):  # Weighted Kernel Density Estimation
        # Building's weight + POIs weight
        X_W = np.concatenate(
            [X_weights.values, np.repeat([pois_weight], len(X_p))]
        )

        if log_weight:  # Apply logarithm
            X_W = np.log(X_W)

        PDF = WeightedKernelDensityEstimation(X, X_W, bandwidth, Y)
        return pd.Series(PDF / PDF.max())
    else:  # Kernel Density Estimation
        # Sklearn
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(X)
        # Sklearn returns the results in the form log(density)
        PDF = np.exp(kde.score_samples(Y))
        return pd.Series(PDF / PDF.max())
