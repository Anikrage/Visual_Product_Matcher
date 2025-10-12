import os
import numpy as np
from io import BytesIO

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse
from pymongo import MongoClient
from pymongo.server_api import ServerApi

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

from PIL import Image
from requests import get
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from heapq import nlargest

import clip
import torch

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
docs=list(products_collection.find({"embedding":{"$exists": True}},{"product_id": 1, "name": 1,"image_url": 1,"embedding":1, "clip_embedding":1,"text_embedding":1}))

#storing as numpy array for faster computation
P_IDS=[doc["product_id"]for doc in docs]
P_NAME=[doc["name"]for doc in docs]
P_IMG=[doc["image_url"]for doc in docs]
P_EMB=np.array([doc["embedding"]for doc in docs])
P_CEMB = np.array([doc.get("clip_embedding", doc["embedding"]) for doc in docs])
P_TEXT= np.array([doc.get("text_embedding", doc["embedding"]) for doc in docs])

#ResNet50 Model Init
model=ResNet50(weights='imagenet',include_top=False,pooling='avg')

#load CLIP
device='cpu'
clip_model,clip_preprocess=clip.load("ViT-B/32",device=device)

#extracts the embeddings from the image
#accepts image in byte stream and returns ndarray
def get_image_features(imgb):
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
      print(f"Error resnet: {e}")
      return None

def get_clip_features(imgb):
    try:
        img=Image.open(BytesIO(imgb))
        if img.mode!="RGB":
            img = img.convert('RGB')
        img_input=clip_preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            ftrs=clip_model.encode_image(img_input)
            ftrs=ftrs.cpu().numpy().flatten()
        norm = np.linalg.norm(ftrs)
        if norm > 0:
            ftrs=ftrs/norm
        return ftrs
    except Exception as e:
        print(f"Error Clip: {e}")
        return None
        
        
#api endpoint for /find_similar
@app.post("/find_similar")
async def find_similar(file: UploadFile=File(...),k:int=5, a:float=0.5):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="image file invalid")
    content = await file.read()
    
    img_vec=get_image_features(content) #gets the image embeddings 
    img_vec_2d=img_vec.reshape(1,-1)

    clip_vec=get_clip_features(content)
    clip_vec_2d=clip_vec.reshape(1,-1)
    
    #calculate similarity
    res_sim=cosine_similarity(img_vec_2d,P_EMB)[0]
    clip_sim=cosine_similarity(clip_vec_2d,P_CEMB)[0]
    text_sim=cosine_similarity(clip_vec_2d,P_TEXT)[0]
    res_sim_norm = (res_sim - res_sim.min()) / (res_sim.max() - res_sim.min() + 1e-8)
    clip_sim_norm = (clip_sim - clip_sim.min()) / (clip_sim.max() - clip_sim.min() + 1e-8)
    text_sim_norm = (text_sim - text_sim.min()) / (text_sim.max() - text_sim.min() + 1e-8)
    
    #Added a 3 Stage Similarity Ranking
    #1st Stage or Visual Similarity Ranking
    visual_sim = (0.6 * res_sim_norm) + (0.4 * clip_sim_norm)
    visual_top100 = np.argsort(visual_sim)[::-1][:100]
    
    # STAGE 2: Filter by semantic similarity threshold
    MIN_TEXT_THRESHOLD = 0.25
    filtered_candidates = []
    for i in visual_top100:
        if text_sim_norm[i] >= MIN_TEXT_THRESHOLD:
            # Calculate final score for filtered candidate
            final_score = (0.35 * res_sim_norm[i]) + (0.25 * clip_sim_norm[i]) + (0.4 * text_sim_norm[i])
            filtered_candidates.append((i, final_score))
            
    ''' #commenting out cause it feels bad :(
    # THis part triggers if there is not enough matches
    if len(filtered_candidates) < k:
        MIN_TEXT_THRESHOLD = 0.15  # More lenient
        filtered_candidates = []
        for i in visual_top100:
            if text_sim_norm[i] >= MIN_TEXT_THRESHOLD:
                final_score = (0.35 * res_sim_norm[i]) + (0.25 * clip_sim_norm[i]) + (0.4 * text_sim_norm[i])
                filtered_candidates.append((i, final_score))

    # last fallback to just visual matches if still not enough
    if len(filtered_candidates) < k:
        filtered_candidates = [(i, visual_sim[i]) for i in visual_top100[:k]]
    '''
    # Stage 3 of Sorting by Final Score and getting the top itenms
    top_k = nlargest(k, filtered_candidates, key=lambda x: x[1])
    response=[]
    for i, score in top_k:
        response.append({
            "product_id": int(P_IDS[i]),
            "name": P_NAME[i],
            "image_url": P_IMG[i],
            "similarity": float(score)
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
@app.api_route("/health",methods=["GET","HEAD"])
async def health_check():
    return PlainTextResponse("OK", status_code=200)

@app.get("/")
async def root():
    return {"message": "Api service working"}