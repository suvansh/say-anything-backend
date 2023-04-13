from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS
from PIL import Image
from io import BytesIO
from get_points import get_points
import numpy as np
import json
import logging
logging.basicConfig(level=logging.INFO)
handler = logging.FileHandler('/home/ubuntu/logs/sam-clip-server.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


app = Flask(__name__)
CORS(app, origins=["*"]) #, resources={r"/api/*": {"origins": ["chrome-extension://hoijgmpnklmkakbhnkimlfppcfegegmp", "chrome-extension://PUBLISHED_EXTENSION_ID", "https://segment-anything.com"]}})
api = Api(app)

class CoordinatesResource(Resource):
    def get(self):
        logging.info('received get request')
        return {'message': 'You have reached the SAM CLIP API', 'points': [{"x": 1200, "y": 1000}],
                              'image_width': 2000,  # TODO remove
                                            'image_height': 1200}, 200
    def post(self):
        logging.info('received post request')
        text_query = request.form['textQuery']
        image_file = request.files['image']
        image_data = image_file.read()

        with open("images/received_image.png", "wb") as f:
            f.write(image_data)
        num_detections = int(request.form['numDetections']) if 'numDetections' in request.form else None
        image = Image.open(image_file)
        im_width, im_height = image.size
        image.save("images/image.png")
        np.save("images/image_np.npy", np.asarray(image.convert("RGB")))

        # Process the image and text query using your custom function
        points = get_points(text_query, image, num_detections)

        return {'points': points,
                'image_width': im_width,
                'image_height': im_height}, 200

api.add_resource(CoordinatesResource, '/api/coordinates')

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
