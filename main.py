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
crops_collection = db.crops_collection

class PhotoDetail(BaseModel):
    photo_url: str

class Indicator(BaseModel):
    name: str
    executionDate: str
    executors: List[str]
    value: float
    measurement_type: str

class CropBase(BaseModel):
    id: str
    cropType: str = Field(..., alias="cropType")
    indicators: Optional[List[Indicator]] = None
    photo_url: Optional[List[PhotoDetail]] = None

class CropCreate(CropBase):
    pass

class CropUpdate(CropBase):
    pass


# CRUD crops
@app.post("/crops/", response_model=CropCreate)
async def create_crop(crop: CropCreate):
    existing_crop = crops_collection.find_one({"id": crop.id})
    if existing_crop:
        raise HTTPException(status_code=400, detail="Crop already exists")
    result = crops_collection.insert_one(crop.model_dump(by_alias=True))
    return crop

@app.get("/crops/", response_model=List[CropBase])
async def get_all_crops():
        crops = list(crops_collection.find({}))
        return crops

@app.delete("/crops/{crop_id}")
async def delete_crop(crop_id: str):
    result = crops_collection.delete_one({"id": crop_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Crop not found")
    return {"message": "Crop deleted successfully"}





@app.post("/crops/{crop_id}/indicators/", response_model=Indicator)
async def add_indicator(crop_id: str, indicator: Indicator):
    result = crops_collection.update_one(
        {"id": crop_id},
        {"$push": {"indicators": indicator.model_dump()}}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Crop not found")
    return indicator



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)