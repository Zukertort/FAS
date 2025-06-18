from fastapi import FastAPI
from pydantic import BaseModel, Field
class Item(BaseModel):
    input: str | None = Field(
        default=None, title="Insert text to analyse", max_length=500
    )

app = FastAPI()

@app.post("/items/")
async def create_item(item: Item):
    return item