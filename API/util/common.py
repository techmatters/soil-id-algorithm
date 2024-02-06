import collections

from flask_restful import inputs, reqparse


def getSpeciesParser():
    speciesparser = reqparse.RequestParser()
    speciesparser.add_argument("longitude", type=float, required=True, help="Longitude is required")
    speciesparser.add_argument("latitude", type=float, required=True, help="Latitude is required")
    speciesparser.add_argument("plot_id", type=float, required=True, help="The LandPKS plot ID")
    return speciesparser


def getTrendsParser():
    trendsparser = reqparse.RequestParser()
    trendsparser.add_argument("longitude", type=float, required=True, help="Longitude is required")
    trendsparser.add_argument("latitude", type=float, required=True, help="Latitude is required")
    trendsparser.add_argument("plot_id", type=float, required=False, help="The LandPKS plot ID")
    return trendsparser


def getClimateParser():
    speciesparser = reqparse.RequestParser()
    speciesparser.add_argument("longitude", type=float, required=True, help="Longitude is required")
    speciesparser.add_argument("latitude", type=float, required=True, help="Latitude is required")
    speciesparser.add_argument("plot_id", type=float, required=False, help="The LandPKS plot ID")
    return speciesparser


def getLocationParser():
    locationparser = reqparse.RequestParser()
    locationparser.add_argument(
        "longitude", type=float, required=True, help="Longitude is required"
    )
    locationparser.add_argument("latitude", type=float, required=True, help="Latitude is required")
    locationparser.add_argument("plot_id", type=float, required=False, help="The LandPKS plot ID")
    return locationparser


def getRankedParser():
    rankedparser = reqparse.RequestParser()
    rankedparser.add_argument("longitude", type=float, required=True, help="Longitude is required")
    rankedparser.add_argument("latitude", type=float, required=True, help="Latitude is required")
    rankedparser.add_argument(
        "plot_id", type=float, required=False, nullable=True, help="The LandPKS plot ID"
    )
    rankedparser.add_argument(
        "soilHorizon1",
        type=str,
        required=False,
        nullable=True,
        choices=(
            "",
            "SILTY CLAY",
            "SILTY CLAY LOAM",
            "CLAY LOAM",
            "SILT",
            "SILT LOAM",
            "SANDY CLAY",
            "LOAM",
            "SANDY CLAY LOAM",
            "SANDY LOAM",
            "LOAMY SAND",
            "SAND",
            "CLAY",
        ),
        help="soilHorizon1 is required and must be one of SILTY CLAY, SILTY CLAY LOAM, CLAY LOAM, SILT, SILT LOAM, SANDY CLAY, LOAM, SANDY CLAY LOAM, SANDY LOAM, LOAMY SAND, SAND, CLAY",  # noqa: E501
    )
    rankedparser.add_argument(
        "soilHorizon2",
        type=str,
        required=False,
        nullable=True,
        choices=(
            "",
            "SILTY CLAY",
            "SILTY CLAY LOAM",
            "CLAY LOAM",
            "SILT",
            "SILT LOAM",
            "SANDY CLAY",
            "LOAM",
            "SANDY CLAY LOAM",
            "SANDY LOAM",
            "LOAMY SAND",
            "SAND",
            "CLAY",
        ),
        help="soilHorizon2 must be one of SILTY CLAY, SILTY CLAY LOAM, CLAY LOAM, SILT, SILT LOAM, SANDY CLAY, LOAM, SANDY CLAY LOAM, SANDY LOAM, LOAMY SAND, SAND, CLAY",  # noqa: E501
    )
    rankedparser.add_argument(
        "soilHorizon3",
        type=str,
        required=False,
        nullable=True,
        choices=(
            "",
            "SILTY CLAY",
            "SILTY CLAY LOAM",
            "CLAY LOAM",
            "SILT",
            "SILT LOAM",
            "SANDY CLAY",
            "LOAM",
            "SANDY CLAY LOAM",
            "SANDY LOAM",
            "LOAMY SAND",
            "SAND",
            "CLAY",
        ),
        help="soilHorizon3 must be one of SILTY CLAY, SILTY CLAY LOAM, CLAY LOAM, SILT, SILT LOAM, SANDY CLAY, LOAM, SANDY CLAY LOAM, SANDY LOAM, LOAMY SAND, SAND, CLAY",  # noqa: E501
    )
    rankedparser.add_argument(
        "soilHorizon4",
        type=str,
        required=False,
        nullable=True,
        choices=(
            "",
            "SILTY CLAY",
            "SILTY CLAY LOAM",
            "CLAY LOAM",
            "SILT",
            "SILT LOAM",
            "SANDY CLAY",
            "LOAM",
            "SANDY CLAY LOAM",
            "SANDY LOAM",
            "LOAMY SAND",
            "SAND",
            "CLAY",
        ),
        help="soilHorizon4 must be one of SILTY CLAY, SILTY CLAY LOAM, CLAY LOAM, SILT, SILT LOAM, SANDY CLAY, LOAM, SANDY CLAY LOAM, SANDY LOAM, LOAMY SAND, SAND, CLAY",  # noqa: E501
    )
    rankedparser.add_argument(
        "soilHorizon5",
        type=str,
        required=False,
        nullable=True,
        choices=(
            "",
            "SILTY CLAY",
            "SILTY CLAY LOAM",
            "CLAY LOAM",
            "SILT",
            "SILT LOAM",
            "SANDY CLAY",
            "LOAM",
            "SANDY CLAY LOAM",
            "SANDY LOAM",
            "LOAMY SAND",
            "SAND",
            "CLAY",
        ),
        help="soilHorizon5 must be one of SILTY CLAY, SILTY CLAY LOAM, CLAY LOAM, SILT, SILT LOAM, SANDY CLAY, LOAM, SANDY CLAY LOAM, SANDY LOAM, LOAMY SAND, SAND, CLAY",  # noqa: E501
    )
    rankedparser.add_argument(
        "soilHorizon6",
        type=str,
        required=False,
        nullable=True,
        choices=(
            "",
            "SILTY CLAY",
            "SILTY CLAY LOAM",
            "CLAY LOAM",
            "SILT",
            "SILT LOAM",
            "SANDY CLAY",
            "LOAM",
            "SANDY CLAY LOAM",
            "SANDY LOAM",
            "LOAMY SAND",
            "SAND",
            "CLAY",
        ),
        help="soilHorizon6 must be one of SILTY CLAY, SILTY CLAY LOAM, CLAY LOAM, SILT, SILT LOAM, SANDY CLAY, LOAM, SANDY CLAY LOAM, SANDY LOAM, LOAMY SAND, SAND, CLAY",  # noqa: E501
    )
    rankedparser.add_argument(
        "soilHorizon7",
        type=str,
        required=False,
        nullable=True,
        choices=(
            "",
            "SILTY CLAY",
            "SILTY CLAY LOAM",
            "CLAY LOAM",
            "SILT",
            "SILT LOAM",
            "SANDY CLAY",
            "LOAM",
            "SANDY CLAY LOAM",
            "SANDY LOAM",
            "LOAMY SAND",
            "SAND",
            "CLAY",
        ),
        help="soilHorizon7 must be one of SILTY CLAY, SILTY CLAY LOAM, CLAY LOAM, SILT, SILT LOAM, SANDY CLAY, LOAM, SANDY CLAY LOAM, SANDY LOAM, LOAMY SAND, SAND, CLAY",  # noqa: E501
    )
    rankedparser.add_argument(
        "soilHorizon1_RFV",
        type=str,
        required=False,
        nullable=True,
        help="soilHorizon1 must be one of 0-1%, 1-15%, 15-35%, 35-60%, or >60% rock fragments by volume",  # noqa: E501
    )
    rankedparser.add_argument(
        "soilHorizon2_RFV",
        type=str,
        required=False,
        nullable=True,
        help="soilHorizon2 must be one of 0-1%, 1-15%, 15-35%, 35-60%, or >60% rock fragments by volume",  # noqa: E501
    )
    rankedparser.add_argument(
        "soilHorizon3_RFV",
        type=str,
        required=False,
        nullable=True,
        help="soilHorizon3 must be one of 0-1%, 1-15%, 15-35%, 35-60%, or >60% rock fragments by volume",  # noqa: E501
    )
    rankedparser.add_argument(
        "soilHorizon4_RFV",
        type=str,
        required=False,
        nullable=True,
        help="soilHorizon4 must be one of 0-1%, 1-15%, 15-35%, 35-60%, or >60% rock fragments by volume",  # noqa: E501
    )
    rankedparser.add_argument(
        "soilHorizon5_RFV",
        type=str,
        required=False,
        nullable=True,
        help="soilHorizon5 must be one of 0-1%, 1-15%, 15-35%, 35-60%, or >60% rock fragments by volume",  # noqa: E501
    )
    rankedparser.add_argument(
        "soilHorizon6_RFV",
        type=str,
        required=False,
        nullable=True,
        help="soilHorizon6 must be one of 0-1%, 1-15%, 15-35%, 35-60%, or >60% rock fragments by volume",  # noqa: E501
    )
    rankedparser.add_argument(
        "soilHorizon7_RFV",
        type=str,
        required=False,
        nullable=True,
        help="soilHorizon7 must be one of 0-1%, 1-15%, 15-35%, 35-60%, or >60% rock fragments by volume",  # noqa: E501
    )
    rankedparser.add_argument(
        "soilHorizon1_Depth",
        type=int,
        required=False,
        nullable=True,
        help="Depth of horizon 1 in centimmeters",
    )
    rankedparser.add_argument(
        "soilHorizon2_Depth",
        type=int,
        required=False,
        nullable=True,
        help="Depth of horizon 2 in centimmeters",
    )
    rankedparser.add_argument(
        "soilHorizon3_Depth",
        type=int,
        required=False,
        nullable=True,
        help="Depth of horizon 3 in centimmeters",
    )
    rankedparser.add_argument(
        "soilHorizon4_Depth",
        type=int,
        required=False,
        nullable=True,
        help="Depth of horizon 4 in centimmeters",
    )
    rankedparser.add_argument(
        "soilHorizon5_Depth",
        type=int,
        required=False,
        nullable=True,
        help="Depth of horizon 5 in centimmeters",
    )
    rankedparser.add_argument(
        "soilHorizon6_Depth",
        type=int,
        required=False,
        nullable=True,
        help="Depth of horizon 6 in centimmeters",
    )
    rankedparser.add_argument(
        "soilHorizon7_Depth",
        type=int,
        required=False,
        nullable=True,
        help="Depth of horizon 7 in centimmeters",
    )
    rankedparser.add_argument(
        "soilHorizon1_LAB", type=str, required=False, nullable=True, help="Dry color of horizon 1"
    )
    rankedparser.add_argument(
        "soilHorizon2_LAB", type=str, required=False, nullable=True, help="Dry color of horizon 2"
    )
    rankedparser.add_argument(
        "soilHorizon3_LAB", type=str, required=False, nullable=True, help="Dry color of horizon 3"
    )
    rankedparser.add_argument(
        "soilHorizon4_LAB", type=str, required=False, nullable=True, help="Dry color of horizon 4"
    )
    rankedparser.add_argument(
        "soilHorizon5_LAB", type=str, required=False, nullable=True, help="Dry color of horizon 5"
    )
    rankedparser.add_argument(
        "soilHorizon6_LAB", type=str, required=False, nullable=True, help="Dry color of horizon 6"
    )
    rankedparser.add_argument(
        "soilHorizon7_LAB", type=str, required=False, nullable=True, help="Dry color of horizon 7"
    )
    rankedparser.add_argument(
        "cracks", type=inputs.boolean, required=False, nullable=True, help="Deep, vertical cracks"
    )
    rankedparser.add_argument(
        "bedrock", type=int, required=False, nullable=True, help="bedrock depth in cm"
    )
    rankedparser.add_argument("slope", type=int, required=False, nullable=True, help="slope in %")
    rankedparser.add_argument(
        "elevation", type=int, required=False, nullable=True, help="elevation in m"
    )
    return rankedparser


class MultipleLevelsOfDictionary(collections.OrderedDict):
    def __getitem__(self, item):
        try:
            return collections.OrderedDict.__getitem__(self, item)
        except:
            value = self[item] = type(self)()
            return value
