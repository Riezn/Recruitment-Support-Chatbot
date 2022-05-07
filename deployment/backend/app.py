from flask import Flask, request, jsonify
import pickle
import tensorflow as tf


app = Flask(__name__)


# Load TF model
model = tf.keras.models.load_model('saved_model/model')


# Load text vectorization layer
from_disk = pickle.load(open("saved_model/textvect.pkl", "rb"))
textvect = tf.keras.layers.TextVectorization.from_config(from_disk['config'])

# Adapt to dummy data
textvect.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
textvect.set_weights(from_disk['weights'])


@app.route("/", methods=['GET','POST'])
def model_prediction():
    if request.method == "POST":
        content = request.json
        try:
            data = [content['user_input']]
            pred = model.predict(data)
            if textvect(data).numpy().max() <= 1:
                label_pred = str(1000)
            else:
                label_pred = str(pred.argmax())
            response = {
                    "code" : 200,
                    "status" : "OK",
                    "prediction" : label_pred
                    }
            return jsonify(response)
        except Exception as e:
            label_pred = str(1000)
            response = {
                "code" : 500,
                "status" : "ERROR",
                "prediction" : label_pred,
                "error_msg" : str(e)
                }
            print(str(e))
            return jsonify(response)
    return "<p>Please insert your data in frontend side.</p>"