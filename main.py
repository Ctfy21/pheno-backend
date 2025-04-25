
from fastapi import FastAPI, HTTPException
from influxdb_client import InfluxDBClient
from mongoengine import connect, Document, StringField, IntField, ReferenceField, ListField, FloatField, \
    EmbeddedDocument, EmbeddedDocumentField
from pydantic import BaseModel, Field
from datetime import date
from bson import ObjectId
from typing import List, Optional

app = FastAPI()

# Provide the mongodb url to connect python to mongodb using pymongo
connection_string = 'mongodb://gen_user:qwerty120978@5.129.197.38:27017/default_db?authSource=admin&directConnection=true'
influx_url = "http://103.74.93.149:8086"
influx_token = "cQ5_9qwHUehOeGa-iE1SM_7bGqM5oxKwcewc3PDtGWhcgJ6rD2pIIEJc4BvAYkFUcm8ClO3odZ_IQQwayigI9w=="
influx_org = "cbt"

from pymongo import MongoClient
client = MongoClient(connection_string)

class RangeDate(BaseModel):
    startUnixDateTime: int
    endUnixDateTime: int

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

class CCTV(BaseModel):
    name: str
    streamUrl: str
    table: int

class ClimaticChamber(BaseModel):
    name: str
    cctv: Optional[List[CCTV]] = []

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

class Experiment(BaseModel):
    startDate: str
    endDate: str
    chamber_id: str
    plant: List[str]

db = client['default_db']
plant_collection = db.plant_collection
experiment_collection = db.experiment_collection
chamber_collection = db.chamber_collection
indicator_type_collection = db.indicator_type_collection


# CRUD environment_data

@app.get("/environment_data/")
def retrieve_data(range_date: RangeDate):
    with InfluxDBClient(url=influx_url, token=influx_token, org=influx_org) as influx_client:
        query_api = influx_client.query_api()

        query = f'''
        from(bucket: "voronesh")
        |> range(start: {range_date.startUnixDateTime}, stop: {range_date.endUnixDateTime})
        |> filter(fn: (r) => r.domain == "sensor")
        |> filter(fn: (r) => r.friendly_name =~ /ROOM1/)
        |> filter(fn: (r) => r._field == "value")
        |> filter(fn: (r) => r._measurement =~ /С/)
        |> aggregateWindow(every: 1m, fn: mean)
        |> yield()
    '''
        tables = query_api.query(query)
        array_values = []
        for table in tables:
            print(table)
            for record in table.records:
                print(record)
                array_values.append(record.get_value())
            return array_values
        return None


# CRUD climatic chamber
@app.post("/chamber/", response_model=ClimaticChamber)
def create_climatic_chamber(climatic_chamber: ClimaticChamber):
    result = chamber_collection.insert_one(climatic_chamber.model_dump())
    if result.inserted_id == 0:
        return HTTPException(status_code=501, detail="Error: cannot create climate chamber ")
    return climatic_chamber


@app.post("/chamber/{climatic_chamber_id}/cctv/", response_model=CCTV)
def create_cctv(climatic_chamber_id: str, cctv: CCTV):
    result = chamber_collection.update_one(
        {"_id": ObjectId(climatic_chamber_id)},
        {"$push": {"cctv": cctv.model_dump()}},
        upsert=True
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=400, detail="Error: climatic chamber with this id doesn't exist ")
    return cctv

@app.delete("/chamber/cctv/{cctv_chamber_id}")
def delete_cctv(cctv_chamber_id: str):
    result = chamber_collection.delete_one({"_id": cctv_chamber_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="cctv not found")
    return {"message": "plant deleted successfully"}

@app.get("/chamber/{climatic_chamber_id}/cctv/")
def get_cctv(climatic_chamber_id: str):
    result = chamber_collection.find_one(
        {"_id": ObjectId(climatic_chamber_id)},
    )
    if result is None:
        raise HTTPException(status_code=400, detail="Error: CCTV with this id doesn't exist ")
    return result.cctv

# CRUD plants
@app.post("/plant/", response_model=Plant)
def create_plant(plant: Plant):
    existing_plant = plant_collection.find_one({"uid": plant.uid})
    if existing_plant:
        raise HTTPException(status_code=400, detail="plant already exists")
    result = plant_collection.insert_one(plant.model_dump())
    if not result.inserted_id:
        return HTTPException(status_code=501, detail="Error: cannot create plant ")
    return plant

@app.get("/plant/")
def get_all_plants():
        plants = list(plant_collection.find({}))
        for plant in plants:
            plant["_id"] = str(plant["_id"])
        print(plants)
        return plants

@app.delete("/plant/{plant_id}")
def delete_plant(plant_id: str):
    result = plant_collection.delete_one({"_id": plant_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="plant not found")
    return {"message": "plant deleted successfully"}



@app.put("/plant/{plant_id}", response_model=Plant)
def update_plant(plant: Plant, plant_id: str):
    existing_plant = plant_collection.find_one({"_id": plant_id})
    if not existing_plant:
        raise HTTPException(status_code=400, detail="plant doesn't exists")

    result = plant_collection.update_one(
        {"_id": plant_id},
        {"$set": plant},
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=501, detail="Error: cannot update plant")



# CRUD indicator type
@app.post("/indicator_type/", response_model=IndicatorType)
def create_indicator_type(indicator_type: IndicatorType):
    existing_indicator_type = indicator_type_collection.find_one({"name": indicator_type.name})
    if existing_indicator_type:
        raise HTTPException(status_code=400, detail="IndicatorType already exists")
    result = indicator_type_collection.insert_one(indicator_type.model_dump())
    if not result.inserted_id:
        return HTTPException(status_code=501, detail="Error: cannot create IndicatorType ")
    return IndicatorType

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
    return {"message": "IndicatorType deleted successfully"}

@app.put("/indicator_type/{indicator_type_id}", response_model=IndicatorType)
def update_indicator_type(indicator_type: IndicatorType, indicator_type_id: str):
    existing_indicator_type = indicator_type_collection.find_one({"_id": ObjectId(indicator_type_id)})
    if not existing_indicator_type:
        raise HTTPException(status_code=400, detail="IndicatorType doesn't exists")

    result = indicator_type_collection.update_one(
        {"_id": indicator_type_id},
        {"$set": indicator_type},
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=501, detail="Error: cannot update IndicatorType")

@app.get("/indicator_type/{indicator_type_id}")
def get_indicator_type(indicator_type_id: str):
    result = indicator_type_collection.find_one({"_id": ObjectId(indicator_type_id)})
    if result is None:
        raise HTTPException(status_code=404, detail="IndicatorType not found")
    return result



# CRUD indicator
@app.post("/plant/{plant_id}/indicator/", response_model=Indicator)
def create_indicator(plant_id: str, indicator: Indicator):
    existing_plant = plant_collection.find_one({"_id": plant_id})
    if existing_plant is None:
        raise HTTPException(status_code=400, detail="plant doesn't exists")
    indicator_type = indicator_type_collection.find_one({"_id": ObjectId(indicator.indicatorType)})
    if indicator_type is None:
        return HTTPException(status_code=501, detail="Error: cannot find indicator with this id ")

    result = plant_collection.update_one(
        {"_id": plant_id},
        {"$push": {"indicator": indicator.model_dump()}},
        upsert=True
    )
    return Indicator

@app.get("/plant/{plant_id}/indicator/", response_model=Indicator)
def get_all_indicators_of_plant(plant_id: str):
        existing_plant = plant_collection.find_one({"_id": plant_id})
        if existing_plant is None:
            raise HTTPException(status_code=400, detail="plant doesn't exists")
        return existing_plant.indicator

@app.delete("/plant/{plant_id}/indicator/{indicator_name}")
def delete_indicator(indicator_name: str, plant_id: str):
    existing_plant = plant_collection.find_one({"_id": plant_id})
    if existing_plant is None:
        raise HTTPException(status_code=400, detail="plant doesn't exists")
    plant_collection.delete_one({"_id": plant_id, "indicator.name": indicator_name})
    return {"message": "Indicator deleted successfully"}

@app.put("/plant/{plant_id}/indicator/{indicator_id}", response_model=Indicator)
def update_indicator(plant_id: str, indicator_id: str, indicator: Indicator):
    existing_plant = plant_collection.find_one({"_id": plant_id})
    if existing_plant is None:
        raise HTTPException(status_code=400, detail="plant doesn't exists")

    result = plant_collection.update_one(
        {"_id": plant_id, "indicator._id": indicator_id},
        {"$set": indicator},
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=501, detail="Error: cannot update Indicator")

# @app.get("/plant/{plant_id}/indicator/{indicator_id}")
# def get_indicator(indicator_id: str):
#     existing_plant = plant_collection.find_one({"_id": plant_id})
#     if existing_plant is None:
#         raise HTTPException(status_code=400, detail="plant doesn't exists")
#     if result is not None:
#         raise HTTPException(status_code=404, detail="Indicator not found")
#     return result




# @app.post("/plants/{plant_id}/indicators/", response_model=Indicator)
# def create_indicator(plant_id: str, indicator: Indicator):
#
#     indicator_types =
#     if indicator.indicatorType
#     result = plant_collection.update_one(
#         {"_id": plant_id},
#         {"$push": {"indicators": indicator.model_dump()}},
#         upsert=True
#     )
#     if result.modified_count == 0:
#         raise HTTPException(status_code=404, detail="plant not found")
#     return indicator



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
