"""Google Earth Engine initialisation for paper1.

Reads configuration from environment variables and initialises the Earth Engine
client.  Import this module before any GEE API call.

Environment variables
---------------------
EE_PROJECT : str
    GEE Cloud project name.
EE_ROI_ASSET : str, optional
    GEE asset ID for the study boundary (e.g. projects/<project>/assets/<name>).
    Only used when ``EE_ROI_LOCAL`` is not set.
EE_ROI_LOCAL : str, optional
    Path to a local GeoJSON study boundary.
"""

from __future__ import annotations

import os

import ee


EE_PROJECT      = os.getenv("EE_PROJECT", "ee-geophilo")
GEE_DRIVE_FOLDER = os.getenv("GEE_DRIVE_FOLDER", "bbtmk_paper1")
EE_ROI_LOCAL    = os.getenv("EE_ROI_LOCAL", None)
EE_ROI_ASSET    = os.getenv("EE_ROI_ASSET", None)


try:
    ee.Initialize(project=EE_PROJECT)
except ee.EEException:
    ee.Initialize()


def get_study_area():
    """Return the study-area ``ee.Geometry``.

    Uses a local GeoJSON if ``EE_ROI_LOCAL`` is set, otherwise requires
    ``EE_ROI_ASSET`` to be defined in the environment.
    """
    if EE_ROI_LOCAL:
        import json
        with open(EE_ROI_LOCAL, encoding="utf-8") as fh:
            geojson = json.load(fh)
        return ee.Geometry(geojson["features"][0]["geometry"])
    if not EE_ROI_ASSET:
        raise RuntimeError(
            "Set EE_ROI_ASSET (GEE asset ID) or EE_ROI_LOCAL (path to GeoJSON) "
            "in your environment."
        )
    return ee.FeatureCollection(EE_ROI_ASSET).geometry()
