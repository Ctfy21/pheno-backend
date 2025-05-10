
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from influxdb_client import InfluxDBClient
from pydantic import BaseModel, Field
from datetime import date
from bson import ObjectId
from typing import List, Optional

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Provide the mongodb url to connect python to mongodb using pymongo
connection_string = 'mongodb://gen_user:qwerty120978@5.129.197.38:27017/default_db?authSource=admin&directConnection=true'
influx_url = "http://103.74.93.149:8086"
influx_token = "cQ5_9qwHUehOeGa-iE1SM_7bGqM5oxKwcewc3PDtGWhcgJ6rD2pIIEJc4BvAYkFUcm8ClO3odZ_IQQwayigI9w=="
influx_org = "cbt"

from pymongo import MongoClient
client = MongoClient(connection_string)

class RangeDate(BaseModel):
    startDate: int
    endDate: int

# class Spectrum(BaseModel):
#     name: str
#     value: int
#     measurementType: str
#
# class Light(BaseModel):
#     number: int
#     spectrum: List[Spectrum]

class EnvironmentData(BaseModel):
    dateTime: str
    temperature: float
    humidity: float
    co2: int
    # startDayTime: int
    # workDayTime: int
    # light: Optional[List[Light]] = []

class ClimaticChamber(BaseModel):
    name: str
    tableId: List[int]

class IndicatorType(BaseModel):
    name: str
    measurementType: str
    options: str

class Indicator(BaseModel):
    indicatorType: str
    executionDate: int
    value: str

class Plant(BaseModel):
    uid: str
    plantType: str
    indicator: Optional[List[Indicator]] = []
    photoUrl: Optional[List[str]] = []
    tableId: int
    experimentId: str

class Experiment(BaseModel):
    name: str
    startDate: int
    endDate: int
    chamberId: str


db = client['default_db']
plant_collection = db.plant_collection
experiment_collection = db.experiment_collection
chamber_collection = db.chamber_collection
indicator_type_collection = db.indicator_type_collection


# environment_data

@app.post("/environment_data/{place}/temperature")
def retrieve_data(range_date: RangeDate, place: str):
    with InfluxDBClient(url=influx_url, token=influx_token, org=influx_org) as influx_client:
        query_api = influx_client.query_api()

        query = f'''
        from(bucket: "{place}")
        |> range(start: {range_date.startDate}, stop: {range_date.endDate})
        |> filter(fn: (r) => r.domain == "sensor")
        |> filter(fn: (r) => r.friendly_name =~ /ROOM1/)
        |> filter(fn: (r) => r._field == "value")
        |> filter(fn: (r) => r._measurement =~ /С/)
        |> aggregateWindow(every: 1h, fn: mean)
        |> yield()
    '''
        tables = query_api.query(query)
        array_values = []
        for table in tables:
            for record in table.records:
                array_values.append([int(record.get_time().timestamp()), record.get_value()])
            return array_values
        return None

@app.post("/environment_data/{place}/humidity")
def retrieve_data(range_date: RangeDate, place: str):
    with InfluxDBClient(url=influx_url, token=influx_token, org=influx_org) as influx_client:
        query_api = influx_client.query_api()

        query = f'''
        from(bucket: "{place}")
        |> range(start: {range_date.startDate}, stop: {range_date.endDate})
        |> filter(fn: (r) => r.domain == "sensor")
        |> filter(fn: (r) => r.friendly_name =~ /ROOM1/)
        |> filter(fn: (r) => r._field == "value")
        |> filter(fn: (r) => r._measurement =~ /%/)
        |> aggregateWindow(every: 1h, fn: mean)
        |> yield()
    '''
        tables = query_api.query(query)
        array_values = []
        for table in tables:
            for record in table.records:
                array_values.append([int(record.get_time().timestamp()), record.get_value()])
            return array_values
        return None



# CRUD climatic chamber
@app.post("/chamber/", response_model=ClimaticChamber)
def create_climatic_chamber(climatic_chamber: ClimaticChamber):
    result = chamber_collection.insert_one(climatic_chamber.model_dump())
    if result.inserted_id == 0:
        return HTTPException(status_code=501, detail="Error: cannot create climate chamber ")
    return climatic_chamber

@app.get("/chamber/{chamber_id}", response_model=ClimaticChamber)
def get_climatic_chamber(chamber_id: str):
    result = chamber_collection.find_one({"_id": ObjectId(chamber_id)})
    if result is None:
        return HTTPException(status_code=404, detail="Chamber not found")
    return result

@app.get("/chamber/")
def get_all_chambers():
        chambers = list(chamber_collection.find({}))
        for chamber in chambers:
            chamber["_id"] = str(chamber["_id"])
        return chambers

@app.put("/chamber/{chamber_id}")
def update_chamber(chamber_id: str, chamber: ClimaticChamber):
    existing_chamber = chamber_collection.find_one_and_update({"_id": ObjectId(chamber_id)}, {"$set": chamber.model_dump()})
    if existing_chamber is None:
        return HTTPException(status_code=404, detail="Chamber not found")
    return chamber

@app.delete("/chamber/{chamber_id}")
def delete_chamber(chamber_id: str):
    result = chamber_collection.delete_one({"_id": ObjectId(chamber_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="chamber not found")
    return JSONResponse({"message": "chamber deleted successfully"}, status_code=200)


#CRUD experiment

@app.post("/experiment", response_model=Experiment)
def create_experiment(experiment: Experiment):
    result = experiment_collection.insert_one(experiment.model_dump())
    if not result.inserted_id:
        return HTTPException(status_code=501, detail="Error: cannot create experiment ")
    return experiment

@app.get("/experiment")
def get_all_experiments():
        experiments = list(experiment_collection.find({}))
        for experiment in experiments:
            experiment["_id"] = str(experiment["_id"])
        return experiments

@app.get("/experiment/{experiment_id}", response_model=Experiment)
def get_experiment(experiment_id: str):
    result = experiment_collection.find_one({"_id": ObjectId(experiment_id)})
    if result is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return result


@app.delete("/experiment/{experiment_id}")
def delete_experiment(experiment_id: str):
    result = experiment_collection.delete_one({"_id": experiment_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="experiment not found")
    return JSONResponse({"message": "experiment deleted successfully"}, status_code=200)

@app.put("/experiment/{experiment_id}", response_model=Experiment)
def update_experiment(experiment: Experiment, experiment_id: str):
    existing_experiment = experiment_collection.find_one_and_update({"_id": ObjectId(experiment_id)}, {"$set": experiment.model_dump()})
    if existing_experiment is None:
        return HTTPException(status_code=404, detail="Chamber not found")
    return experiment

# CRUD plants
@app.post("/plant/", response_model=Plant)
def create_plant(plant: Plant):
    result = plant_collection.insert_one(plant.model_dump())
    if not result.inserted_id:
        return HTTPException(status_code=501, detail="Error: cannot create plant ")
    return plant

@app.get("/plant/")
def get_all_plants():
        plants = list(plant_collection.find({}))
        for plant in plants:
            plant["_id"] = str(plant["_id"])
        return plants

@app.get("/plant/{plant_id}", response_model=Plant)
def get_plant(plant_id: str):
    result = plant_collection.find_one({"_id": ObjectId(plant_id)})
    if result is None:
        raise HTTPException(status_code=404, detail="Plant not found")
    return result


@app.delete("/plant/{plant_id}")
def delete_plant(plant_id: str):
    result = plant_collection.delete_one({"_id": plant_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="plant not found")
    return JSONResponse({"message": "plant deleted successfully"}, status_code=200)

@app.put("/plant/{plant_id}", response_model=Plant)
def update_plant(plant: Plant, plant_id: str):
    existing_plant = plant_collection.find_one_and_update({"_id": ObjectId(plant_id)}, {"$set": plant.model_dump()})
    if existing_plant is None:
        return HTTPException(status_code=404, detail="Chamber not found")
    return plant



# CRUD indicator type
@app.post("/indicator_type/", response_model=IndicatorType)
def create_indicator_type(indicator_type: IndicatorType):
    print(indicator_type.model_dump())
    result = indicator_type_collection.insert_one(indicator_type.model_dump())
    if result.inserted_id == 0:
        return HTTPException(status_code=501, detail="Error: cannot create IndicatorType ")
    return indicator_type

@app.get("/indicator_type/")
def get_all_indicator_types():
        indicator_types = list(indicator_type_collection.find({}))
        for indicator_type in indicator_types:
            indicator_type["_id"] = str(indicator_type["_id"])
        return indicator_types

@app.delete("/indicator_type/{indicator_type_id}")
def delete_indicator_type(indicator_type_id: str):
    result = indicator_type_collection.delete_one({"_id": ObjectId(indicator_type_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="IndicatorType not found")
    return JSONResponse({"message": "indicator type deleted successfully"}, status_code=200)

@app.put("/indicator_type/{indicator_type_id}", response_model=IndicatorType)
def update_indicator_type(indicator_type: IndicatorType, indicator_type_id: str):
    existing_indicator_type = indicator_type_collection.find_one_and_update({"_id": ObjectId(indicator_type_id)}, {"$set": indicator_type.model_dump()})
    if existing_indicator_type is None:
        return HTTPException(status_code=404, detail="Chamber not found")
    return indicator_type

@app.get("/indicator_type/{indicator_type_id}", response_model=IndicatorType)
def get_indicator_type(indicator_type_id: str):
    result = indicator_type_collection.find_one({"_id": ObjectId(indicator_type_id)})
    if result is None:
        raise HTTPException(status_code=404, detail="IndicatorType not found")
    return result



# CRUD indicator
@app.post("/plant/{plant_id}/indicator/")
def create_indicator(plant_id: str, new_indicator: Indicator):
    print(new_indicator.model_dump())
    existing_plant = plant_collection.find_one({"_id": ObjectId(plant_id)})
    if existing_plant is None:
        raise HTTPException(status_code=400, detail="plant doesn't exists")
    indicator_type = indicator_type_collection.find_one({"_id": ObjectId(new_indicator.indicatorType)})
    if indicator_type is None:
        return HTTPException(status_code=501, detail="Error: cannot find indicator with this id ")
    result = plant_collection.update_one(
        {"_id": ObjectId(plant_id)},
        {"$push": {"indicator": new_indicator.model_dump()}},
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=501, detail="Error: cannot create plant indicator or indicator type already exists ")

    return JSONResponse({"message": "Indicator add successfully"}, status_code=201)


@app.delete("/plant/{plant_id}/indicator/{indicator_type}")
def delete_indicator(indicator_type: str, plant_id: str):
    result = plant_collection.update_one({"_id": ObjectId(plant_id)}, {"$pull": {"indicator": {"indicatorType": indicator_type }}})
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Indicator or plant doesn't exists")
    return JSONResponse({"message": "Indicator delete successfully"}, status_code=200)

@app.put("/plant/{plant_id}/indicator/{indicator_type}")
def update_indicator(plant_id: str, indicator_type: str, new_indicator: Indicator):
    result = plant_collection.find_one_and_update(
        {"_id": ObjectId(plant_id), "indicator.indicatorType": indicator_type},
        {"$set": {"indicator.$": new_indicator.model_dump()}},
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Error: Indicator or plant doesn't exists")

    return new_indicator

@app.get("/experiment/{experiment_id}/plant")
def get_all_plants_of_experiment(experiment_id: str):
    plants = list(plant_collection.find({"experimentId": experiment_id}))
    for plant in plants:
        plant["_id"] = str(plant["_id"])
    if plants is None:
        raise HTTPException(status_code=404, detail="Plants not found")
    return JSONResponse(plants, status_code=200)


# CRUD CCTV
# @app.post("/chamber/{climatic_chamber_id}/cctv/", response_model=CCTV)
# def create_cctv(climatic_chamber_id: str, cctv: CCTV):
#     result = chamber_collection.update_one(
#         {"_id": ObjectId(climatic_chamber_id)},
#         {"$push": {"cctv": cctv.model_dump()}},
#         upsert=True
#     )
#     if result.modified_count == 0:
#         raise HTTPException(status_code=400, detail="Error: climatic chamber with this id doesn't exist ")
#     return cctv
#
# @app.delete("/chamber/cctv/{cctv_chamber_id}")
# def delete_cctv(cctv_chamber_id: str):
#     result = chamber_collection.delete_one({"_id": cctv_chamber_id})
#     if result.deleted_count == 0:
#         raise HTTPException(status_code=404, detail="cctv not found")
#     return {"message": "plant deleted successfully"}
#
# @app.get("/chamber/{climatic_chamber_id}/cctv/{cctv_id}", response_model=CCTV)
# def get_cctv(climatic_chamber_id: str):
#     result = chamber_collection.find_one(
#         {"_id": ObjectId(climatic_chamber_id)},
#     )
#     if result is None:
#         raise HTTPException(status_code=400, detail="Error: CCTV with this id doesn't exist ")
#     return result.cctv



# @app.get("/plant/{plant_id}/indicator/{indicator_id}")
# def get_indicator(indicator_id: str):
#     existing_plant = plant_collection.find_one({"_id": plant_id})
#     if existing_plant is None:
#         raise HTTPException(status_code=400, detail="plant doesn't exists")
#     if result is not None:
#         raise HTTPException(status_code=404, detail="Indicator not found")
#     return result






if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
