from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS
from PIL import Image
from io import BytesIO
from get_points import get_points
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["chrome-extension://hoijgmpnklmkakbhnkimlfppcfegegmp", "chrome-extension://PUBLISHED_EXTENSION_ID"]}})
api = Api(app)

class CoordinatesResource(Resource):
    def post(self):
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

if __name__ == '__main__':
    app.run(debug=True)
