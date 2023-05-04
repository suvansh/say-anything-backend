from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS
from PIL import Image
from io import BytesIO
from get_points import get_points
import numpy as np
import json
import requests
import logging

app = Flask(__name__)
CORS(app, origins=["*"]) #, resources={r"/api/*": {"origins": ["chrome-extension://hoijgmpnklmkakbhnkimlfppcfegegmp", "chrome-extension://PUBLISHED_EXTENSION_ID", "https://segment-anything.com"]}})

@app.route('/api/coordinates', methods=['GET', 'POST'])
def handle_requests():
    if request.method == "GET":
        logging.info('received get request')
        return {'message': 'You have reached the SAM-CLIP API!'}, 200
    elif request.method == "POST":
        logging.info('received post request')
        text_query = request.json['textQuery']
        image_URL= request.json['imageURL']
        img_data = requests.get(image_URL).content
        image = Image.open(BytesIO(img_data))

        num_detections = request.json['numDetections']
        num_detections = num_detections and int(num_detections)
        im_width, im_height = image.size

        # Process the image and text query using your custom function
        points = get_points(text_query, image, num_detections)

        return {'points': points,
                'image_width': im_width,
                'image_height': im_height}, 200


def lambda_handler(event, context):
    http_method = event['httpMethod']
    if http_method == 'OPTIONS':
        # Return CORS headers for OPTIONS request
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST',
            'Access-Control-Allow-Headers': 'Content-Type'
        }
        
        return {
            'statusCode': 200,
            'headers': headers
        }
    elif http_method == "GET":
         return {
              'statusCode': 200,
              'body': "Hello, you've reached the API for the SAM_CLIP extension by Brilliantly.ai!"
         }
    elif http_method == "POST":
        body = json.loads(event['body'])
        text_query = body['textQuery']
        image_file = body['image']
        image_data = image_file.read()
        with open("images/received_image.png", "wb") as f:
                f.write(image_data)
        num_detections = int(request.form['numDetections']) if 'numDetections' in body else None
        image = Image.open(image_file)
        im_width, im_height = image.size

        # Process the image and text query using your custom function
        points = get_points(text_query, image, num_detections)
        result = {'points': points,
                'image_width': im_width,
                'image_height': im_height}

        return {
            'StatusCode': 200,
            'body': json.dumps(result)
        }
    else:
         return {
                'statusCode': 400,
                'body': "Invalid HTTP method"
            }

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
