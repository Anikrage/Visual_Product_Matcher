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
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

db=client[database]
products_collection=db['products']

#ResNet50 Model Init
model=ResNet50(weights='imagenet',include_top=False,pooling='avg')

#extracts the embeddings from the image
#accepts image in byte stream and returns ndarray
def get_image_features(imgb:bytes) -> np.ndarray:
  try:
    img=Image.open(BytesIO(imgb))

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
    
    docs=list(products_collection.find({"embedding":{"$exists": True}})) 
    if not docs:
        raise HTTPException(status_code=500, detail="Database error")
    
    sim_ar=[]
    for item in docs: #computes and stores the similarity scores
        emb=np.array(item["embedding"])
        score=cosine_similarity([img_vec],[emb])[0][0]
        sim_ar.append((score,item["product_id"],item["name"],item["image_url"]))
    
    #sort and select top k items
    sim_ar.sort(key=lambda x:x[0], reverse=True)
    top_items=sim_ar[:k]
    
    response=[]
    for score,pid,name,url in top_items:
        response.append({
            "product_id": pid,
            "name": name,
            "image_url": url,
            "similarity": float(score)
        })
        
    return {"results":response}

#health check endpoint
@app.get("/health")
def health_check():
    return {"status": "OK"}