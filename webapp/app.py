import os
import requests

from flask import Flask
from flask import render_template
from flask import request,redirect
from flask import url_for,flash,jsonify

from dotenv import load_dotenv
from pymongo import mongo_client
from pymongo.server_api import ServerApi

load_dotenv()


app= Flask(__name__)
app.secret_key=os.getenv('SECRET_KEY')

#config
api_url=os.getenv('HF_API_URL')
UPLOAD_FOLDER='static/uploads'
allowed_extensions={'png','jpg','jpeg','webp'}
max_file_size=5242880 #5MB

app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH']=max_file_size

if __name__=='__main__':
    app.run(debug=True,port=5000)
