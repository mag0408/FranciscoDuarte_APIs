from fastapi import APIRouter, HTTPException
from controller import create_item, read_items, read_item, update_item, delete_item
from model import Item
from typing import List

router = APIRouter()

@router.post("/items/", response_model=Item)
async def create_item_handler(item: Item):
    created_item = create_item(item)
    return created_item

@router.get("/items/", response_model=List[Item])
async def read_items_handler():
    return read_items()

@router.get("/items/{item_id}", response_model=Item)
async def read_item_handler(item_id: int):
    item = read_item(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return item

@router.put("/items/{item_id}", response_model=Item)
async def update_item_handler(item_id: int, item: Item):
    updated_item = update_item(item_id, item)
    if updated_item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return updated_item

@router.delete("/items/{item_id}", response_model=Item)
async def delete_item_handler(item_id: int):
    deleted_item = delete_item(item_id)
    if deleted_item is None:
        raise HTTPException(status_code=404, detail="Item not found")
    return deleted_item
