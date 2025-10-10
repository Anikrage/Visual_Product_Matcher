import os
import numpy as np
from io import BytesIO

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from pymongo import MongoClient
from pymongo.server_api import ServerApi

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

from PIL import Image
from requests import get
from sklearn.metrics.pairwise import cosine_similarity

app=FastAPI()

#environment secrets
load_dotenv() 

username=os.getenv('MONGO_USERNAME')
password=os.getenv('MONGO_PASSWORD')
cluster=os.getenv('MONGO_CLUSTER')
database=os.getenv('MONGO_DATABASE','product_matcher_db')

#database connectivity to MondodbATlas
connect_url=f"mongodb+srv://{username}:{password}@{cluster}/?retryWrites=true&w=majority&appName=ProductImageCluster"

client = MongoClient(connect_url, server_api=ServerApi('1'))

db=client[database]
products_collection=db['products']

#caching all documents at startup
docs=list(products_collection.find({"embedding":{"$exists": True}},{"product_id": 1, "name": 1,"image_url": 1,"embedding":1}))

#storing as numpy array for faster computation
P_IDS=[doc["product_id"]for doc in docs]
P_NAME=[doc["name"]for doc in docs]
P_IMG=[doc["image_url"]for doc in docs]
P_EMB=np.array([doc["embedding"]for doc in docs])

#ResNet50 Model Init
model=ResNet50(weights='imagenet',include_top=False,pooling='avg')

#extracts the embeddings from the image
#accepts image in byte stream and returns ndarray
def get_image_features(imgb:bytes) -> np.ndarray:
  try:
    img=Image.open(BytesIO(imgb))
    if img.mode!="RGB":
        img = img.convert('RGB')
    img = img.resize((224,224))
    img_array=img_to_array(img)
    img_array=np.expand_dims(img_array,axis=0)
    img_array=preprocess_input(img_array)
    features=model.predict(img_array,verbose=0)
    features=features.flatten()
    normalized=np.linalg.norm(features)
    if normalized > 0:
      features=features/normalized

    return features
  except Exception as e:
    print(f"Error: {e}")

#api endpoint for /find_similar
@app.post("/find_similar")
async def find_similar(file: UploadFile=File(...),k:int=5):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="image file invalid")
    content = await file.read()
    img_vec=get_image_features(content) #gets the image embeddings 
    img_vec_2d=img_vec.reshape(1,-1)

    #calculate similarity
    sim=cosine_similarity(img_vec_2d,P_EMB)[0]
    
    #sort and select top k items
    top_item_index=np.argsort(sim)[::-1][:k]    
    response=[]
    for i in top_item_index:
        response.append({
            "product_id": int(P_IDS[i]),
            "name": P_NAME[i],
            "image_url": P_IMG[i],
            "similarity": float(sim[i])
        })
        
    return {"results":response}

#testing db server access
@app.get("/test_db")
def test_db():
    try:
        client.admin.command('ping')
        return {
            "status": "connected",
            "database": database,
        }
    except Exception as e:
        raise HTTPException(status_code=500,detail=f"db connectivity error: {str(e)}")

#health check endpoint
@app.get("/health")
def health_check():
    return {"status": "OK"}