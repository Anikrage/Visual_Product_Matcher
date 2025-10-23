import os
import requests
from io import BytesIO
from PIL import Image
import uuid

from flask import Flask
from flask import render_template
from flask import request,redirect
from flask import url_for,flash,jsonify

from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from pymongo.server_api import ServerApi

load_dotenv()


app= Flask(__name__)
app.secret_key=os.getenv('SECRET_KEY')

#config
api_url = "https://anikrage1-product-matcher-api.hf.space/find_similar"
UPLOAD_FOLDER='static/uploads'
allowed_extensions={'png','jpg','jpeg','webp'}
max_file_size=5242880 #5MB

os.makedirs(UPLOAD_FOLDER,exist_ok=True)

username=os.getenv('MONGO_USERNAME')
password=os.getenv('MONGO_PASSWORD') 
cluster=os.getenv('MONGO_CLUSTER')
database=os.getenv('MONGO_DATABASE','product_matcher_db')

app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH']=max_file_size

#database connectivity to MondodbATlas
#Uncomment the below link and use for Deployment and testing if you have credentials
connect_url=f"mongodb+srv://{username}:{password}@{cluster}/?retryWrites=true&w=majority&appName=ProductImageCluster"

#Testing Access. Uncomment for local testing
'''
connect_url="mongodb+srv://visitor:gn5j5bMbakTnI3FX@productimagecluster.j4w7buk.mongodb.net/?retryWrites=true&w=majority&appName=ProductImageCluster"
'''
client = MongoClient(connect_url, server_api=ServerApi('1'))

#use for actual deployment 
db=client[database]
#use for test access
'''
db=client['product_matcher_db']
'''
products_collection=db['products']

#startup cache
master_cache = sorted(products_collection.distinct('master_category'))
sub_cache = sorted(products_collection.distinct('sub_category'))

def check_filetype(fname):
    return '.' in fname and fname.rsplit('.', 1)[1].lower() in allowed_extensions

#Homepage
@app.route('/')
def index():
    master_category=request.args.get('master_category','')
    sub_category=request.args.get('sub_category','')
    
    query={}
    if master_category:
        query['master_category']=master_category
    if sub_category:
        query['sub_category']=sub_category
    
    total_count=products_collection.count_documents(query)
    prod_list=list(products_collection.find(query).limit(100))
    
    return render_template('index.html',
                           products=prod_list,
                           total_products=total_count,
                           master_categories=master_cache,
                           sub_categories=sub_cache,
                           current_master=master_category,
                           current_sub=sub_category)

@app.route('/health')
def health():
    client.admin.command('ping')
    return {
        "service:": "running",
        "db_status": "connected",
        "database": database,
    }


@app.route('/search',methods=['POST']) #added oversight where in multiple users are using to prevent racearound conditions with image uploads
def search_product():
    fpath=None
    temp_path=None
    
    img_source=request.form.get('image_source','upload')
    min_sim=float(request.form.get('min_similarity',0.0))
    
    try:
        if img_source=='url':
            # User provided a URL instead of uploading
            img_url=request.form.get('image_url')
            if not img_url:
                flash('Please provide an image URL')
                return redirect(url_for('index'))
            
            resp=requests.get(img_url,timeout=10)
            image_data=resp.content
            
            # Need to save this locally for display purposes
            img=Image.open(BytesIO(image_data))
            import uuid
            temp_path=f'static/temp_upload_{uuid.uuid4().hex}.jpg'
            img.save(temp_path)
            uploaded_img_url='/'+temp_path
            
        else:
            # Regular file upload path
            if 'image' not in request.files:
                flash('Please upload an image')
                return redirect(url_for('index'))

            file=request.files['image']
            if file.filename=='' or not check_filetype(file.filename):
                # Invalid file
                return redirect(url_for('index'))

            filename=secure_filename(file.filename)
            fpath=os.path.join(app.config['UPLOAD_FOLDER'],filename)
            file.save(fpath)

            with open(fpath,'rb') as f:
                image_data=f.read()

            img=Image.open(BytesIO(image_data))
            import uuid  # generate unique filename
            temp_path=f'static/temp_upload_{uuid.uuid4().hex}.jpg'
            img.save(temp_path)
            uploaded_img_url='/'+temp_path

        # Max number of results we want to display
        res_num=20
        master_filter=request.form.get('master_category','')
        sub_filter=request.form.get('sub_category','')

        # Call the similarity API
        filedata={'file':('image.jpg',image_data,'image/jpeg')}
        api_res=requests.post(api_url,files=filedata,params={'k':50,'a':0.6},timeout=30)

        if api_res.status_code != 200:
            # Something went wrong with the API
            flash('API error')
            return redirect(url_for('index'))

        results=api_res.json()['results']
        filtered=[]
        
        # Loop through results and apply filters
        for item in results[:]:
            if item['similarity'] < min_sim:
                continue
            
            prod_data=products_collection.find_one({'product_id':item['product_id']})
            if not prod_data:
                # Product not found in DB
                continue
                
            # Check category filters
            if master_filter and prod_data['master_category']!=master_filter:
                continue
            if sub_filter and prod_data['sub_category']!=sub_filter:
                continue
                
            item['master_category']=prod_data['master_category']
            item['sub_category']=prod_data['sub_category']
            filtered.append(item)
            
            if len(filtered) >= res_num:
                # Got enough results
                break
                
        return render_template('results.html',
                            results=filtered,
                            uploaded_image=uploaded_img_url,
                            min_similarity=min_sim,
                            master_categories=master_cache,
                            sub_categories=sub_cache,
                            filter_master=master_filter,
                            filter_sub=sub_filter)
                            
    except Exception as e:
        flash(str(e),'error')
        return redirect(url_for('index'))

    finally:
        # Cleanup - remove temporary files
        if fpath and os.path.exists(fpath):
            os.remove(fpath)
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/test')
def test():
    test_data = ['apple', 'banana']
    return render_template('test.html', fruits=test_data)

#loads more products to homepage
@app.route('/load_more')
def load_more():
    master_cat=request.args.get('master_category','')
    sub_cat=request.args.get('sub_category','')
    offset=int(request.args.get('offset',0))
    query={}
    if master_cat:
        query['master_category']=master_cat
    if sub_cat:
        query['sub_category']=sub_cat
    prod_list=list(products_collection.find(query).skip(offset).limit(24))
    return jsonify({
        'products': [
            {
                'name': p['name'],
                'image_url': p['image_url'],
                'master_category': p['master_category'],
                'sub_category': p['sub_category']
            } for p in prod_list
        ],
        'has_more': len(prod_list)==24
    })

if __name__=='__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.getenv('PORT', 5000)))  

