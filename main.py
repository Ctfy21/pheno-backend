from fastapi import FastAPI, HTTPException
from pymongo import MongoClient
from pydantic import BaseModel, Field
from datetime import date
from typing import List, Optional

app = FastAPI()

# Provide the mongodb url to connect python to mongodb using pymongo
connection_string = 'mongodb://gen_user:qwerty120978@5.129.197.38:27017/default_db?authSource=admin&directConnection=true'

# Create a connection using MongoClient.
from pymongo import MongoClient
client = MongoClient(connection_string)

db = client['default_db']
plants_collection = db.plants_collection

class Spectrum(BaseModel):
    name: str
    value: int
    type_of_measurment: str

class Light(BaseModel):
    number: int
    spectrum: List[Spectrum]

class EnvironmentData(BaseModel):
    datetime: str
    temperature: float
    humidity: float
    co2: int
    start_day_time: int
    work_day_time: int
    light: Optional[List[Light]]

class CCTV(BaseModel):
    name: str
    stream_url: str
    table: int

class ClimaticChamber(BaseModel):
    name: str
    cctv: Optional[List[CCTV]] = None 
    environment_data: Optional[List[EnvironmentData]] = None

class Place(BaseModel):
    address: str
    climatic_chamber: List[ClimaticChamber]

class Indicator(BaseModel):
    name: str
    executionDate: str
    value: float
    measurement_type: str

class Plant(BaseModel):
    uid: str
    type_of_plant: str
    indicator: List[Indicator]
    photo_url: List[str]

class Experiment(BaseModel):
    start_date: str
    end_date: str
    place: Place
    plant: List[Plant]




# CRUD plants
@app.post("/plants/", response_model=Plant)
def create_plant(plant: Plant):
    existing_plant = plants_collection.find_one({"uid": plant.uid})
    if existing_plant:
        raise HTTPException(status_code=400, detail="plant already exists")
    result = plants_collection.insert_one(plant.model_dump())
    return plant

@app.get("/plants/", response_model=List[Plant])
def get_all_plants():
        plants = list(plants_collection.find({}))
        return plants

@app.delete("/plants/{plant_uid}")
def delete_plant(plant_uid: str):
    result = plants_collection.delete_one({"uid": plant_uid})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="plant not found")
    return {"message": "plant deleted successfully"}


@app.put("/plants/{plant_uid}", response_model=Plant)
def update_plant(plant: Plant, plant_uid: str):
    existing_plant = plants_collection.find_one({"uid": plant_uid})
    if not(existing_plant):
        raise HTTPException(status_code=400, detail="plant doesn't exists")
    
    existing_another_plant = plants_collection.find_one({"uid": plant.uid})

    result = plants_collection.update_one(
        {"uid": plant_uid},
        {"$set": plant},
    )

@app.post("/plants/{plant_uid}/indicators/", response_model=Indicator)
def add_indicator(plant_uid: str, indicator: Indicator):
    result = plants_collection.update_one(
        {"uid": plant_uid},
        {"$push": {"indicators": indicator.model_dump()}}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="plant not found")
    return indicator



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)