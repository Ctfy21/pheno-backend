
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
    # environmentData: Optional[List[str]] = []

class Place(BaseModel):
    address: str
    climaticChamber: List[ClimaticChamber]

class Indicator(BaseModel):
    name: str
    executionDate: str
    value: float
    measurementType: str

class Plant(BaseModel):
    _id: str
    plantType: str
    indicator: List[Indicator]
    photoUrl: List[str]

class Experiment(BaseModel):
    startDate: str
    endDate: str
    place: Place
    plant: List[str]

db = client['default_db']
plant_collection = db.plant_collection
experiment_collection = db.experiment_collection


# CRUD climatic chamber
@app.post("/chamber/", response_model=ClimaticChamber)
def create_climatic_chamber(climatic_chamber: ClimaticChamber):
    result = pheno_collection.insert_one(climatic_chamber.model_dump())
    if result.inserted_id == 0:
        return HTTPException(status_code=501, detail="Error: cannot create climate chamber ")
    return climatic_chamber

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


@app.post("/chamber/{climatic_chamber_id}/cctv/", response_model=CCTV)
def create_cctv(climatic_chamber_id: str, cctv: CCTV):
    result = pheno_collection.update_one(
        {"_id": ObjectId(climatic_chamber_id)},
        {"$push": {"cctv": cctv.model_dump()}},
        upsert=True
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=400, detail="Error: climatic chamber with this id doesn't exist ")
    return cctv

@app.delete("/chamber/cctv/{cctv_chamber_id}")
def delete_cctv(cctv_chamber_id: str):
    result = pheno_collection.delete_one({"_id": cctv_chamber_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="cctv not found")
    return {"message": "plant deleted successfully"}

@app.get("/chamber/{climatic_chamber_id}/cctv/")
def get_cctvs(climatic_chamber_id: str):
    result = pheno_collection.find_one(
        {"_id": ObjectId(climatic_chamber_id)},
    )
    if not result:
        raise HTTPException(status_code=400, detail="Error: CCTV with this id doesn't exist ")
    return result.cctv

# CRUD plants
@app.post("/plants/", response_model=Plant)
def create_plant(plant: Plant):
    existing_plant = pheno_collection.find_one({"uid": plant.uid})
    if existing_plant:
        raise HTTPException(status_code=400, detail="plant already exists")
    result = pheno_collection.insert_one(plant.model_dump())
    if result.inserted_id:
        return plant
    return HTTPException(status_code=501, detail="Error: cannot create plant ")

@app.get("/plants/", response_model=List[Plant])
def get_all_plants():
        plants = list(pheno_collection.find({}))
        return plants

@app.delete("/plants/{plant_uid}")
def delete_plant(plant_uid: str):
    result = pheno_collection.delete_one({"uid": plant_uid})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="plant not found")
    return {"message": "plant deleted successfully"}


@app.put("/plants/{plant_uid}", response_model=Plant)
def update_plant(plant: Plant, plant_uid: str):
    existing_plant = pheno_collection.find_one({"uid": plant_uid})
    if not existing_plant:
        raise HTTPException(status_code=400, detail="plant doesn't exists")

    existing_another_plant = pheno_collection.find_one({"uid": plant.uid})
    if existing_another_plant.uid == existing_plant.uid:
        raise HTTPException(status_code=400, detail="plant existing with this uid")

    result = pheno_collection.update_one(
        {"uid": plant_uid},
        {"$set": plant},
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=501, detail="Error: cannot update plant")


@app.post("/plants/{plant_uid}/indicators/", response_model=Indicator)
def add_indicator(plant_uid: str, indicator: Indicator):
    result = pheno_collection.update_one(
        {"uid": plant_uid},
        {"$push": {"indicators": indicator.model_dump()}},
        upsert=True
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="plant not found")
    return indicator



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
