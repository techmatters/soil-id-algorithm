import json

import date
from flask import Blueprint
from flask_restful import Api, Resource
from util import common

from soil_id import global_soil, speciesModel_v3, us_soil

API_VERSION_V3 = 3
API_VERSION = API_VERSION_V3

api_v3_bp = Blueprint("api_v3", __name__)
api_v3 = Api(api_v3_bp)


class soilIDList(Resource):
    def get(self):
        locationparser = common.getLocationParser()
        args = locationparser.parse_args()
        pointer = us_soil.findSoilLocation(args.longitude, args.latitude)
        if pointer is None:
            return "Soil ID not available in this area"
        elif pointer == "US":
            data = us_soil.getSoilLocationBasedUS(args.longitude, args.latitude, args.plot_id)
        elif pointer == "Global":
            data = global_soil.getSoilLocationBasedGlobal(
                args.longitude, args.latitude, args.plot_id
            )
        return data

    def put(self):
        return {"put": "Put"}


class soilIDRank(Resource):
    def get(self):
        rankedparser = common.getRankedParser()
        args = rankedparser.parse_args()
        userTextureByDepth = [None] * 7
        for i in range(len(userTextureByDepth)):
            if hasattr(args, "soilHorizon%s" % str(i + 1)):
                soilHorizon = getattr(args, "soilHorizon%s" % str(i + 1))
                if soilHorizon:
                    userTextureByDepth[i] = soilHorizon
        userHorizonDepth = [None] * 7
        for i in range(len(userHorizonDepth)):
            if hasattr(args, "soilHorizon%s_Depth" % str(i + 1)):
                HorizonDepth = getattr(args, "soilHorizon%s_Depth" % str(i + 1))
                # if HorizonDepth:
                userHorizonDepth[i] = HorizonDepth
        userRFVDepth = [None] * 7
        for i in range(len(userRFVDepth)):
            if hasattr(args, "soilHorizon%s_RFV" % str(i + 1)):
                RFVDepth = getattr(args, "soilHorizon%s_RFV" % str(i + 1))
                if RFVDepth:
                    userRFVDepth[i] = RFVDepth
        lab_Color = [None] * 7
        for i in range(len(lab_Color)):
            if hasattr(args, "soilHorizon%s_LAB" % str(i + 1)):
                LABDepth = getattr(args, "soilHorizon%s_LAB" % str(i + 1))
                if LABDepth:
                    lab_Color[i] = LABDepth
        for i in range(len(lab_Color)):
            if lab_Color[i] is not None:
                lab_Color[i] = [float(x) for x in lab_Color[i].split(",")]
        pointer = global_soil.findSoilLocation(args.longitude, args.latitude)
        if pointer is None:
            return "Soil ID not available in this area"
        elif pointer == "US":
            data = us_soil.rankPredictionUS(
                args.longitude,
                args.latitude,
                userTextureByDepth,
                userHorizonDepth,
                userRFVDepth,
                lab_Color,
                args.slope,
                args.elevation,
                args.bedrock,
                args.cracks,
                args.plot_id,
            )
        elif pointer == "Global":
            data = global_soil.rankPredictionGlobal(
                args.longitude,
                args.latitude,
                userTextureByDepth,
                userHorizonDepth,
                userRFVDepth,
                lab_Color,
                args.bedrock,
                args.cracks,
                args.plot_id,
            )
        return data

    def put(self):
        return {"put": "Put"}


class soilgridsList(Resource):
    def get(self):
        locationparser = common.getLocationParser()
        args = locationparser.parse_args()
        pointer = global_soil.findSoilLocation(args.longitude, args.latitude)
        if pointer is None:
            return "SoilGrids not available in this area"
        elif pointer == "US":
            data = global_soil.getSoilGridsUS(args.longitude, args.latitude, args.plot_id)
        elif pointer == "Global":
            data = global_soil.getSoilGridsGlobal(args.longitude, args.latitude, args.plot_id)
        return data

    def put(self):
        return {"put": "Put"}


class species(Resource):
    def get(self):
        speciesparser = common.getSpeciesParser()
        args = speciesparser.parse_args()
        species = speciesModel_v3.findSpecies(args.longitude, args.latitude)
        species["metadata"] = {"version": 2.0, "date": date.today().strftime("%m/%d/%Y")}
        if args.plot_id is not None and "message" not in species:
            speciesModel_v3.saveSpeciesModel(args.plot_id, json.dumps(species), 2)
        return species


api_v3.add_resource(soilIDList, "/soilidlist")
api_v3.add_resource(soilIDRank, "/soilidrank")
api_v3.add_resource(soilgridsList, "/soilgridslist")
api_v3.add_resource(species, "/species")
