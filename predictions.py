from sklearn.externals import joblib
from boto.s3.key import Key
from boto.s3.connection import S3Connection
from flask import Flask
from flask import request
from flask import json

BUCKET_NAME = 'sw-day-bucket'
MODEL_FILE_NAME = 'sw_day_model.pkl'
MODEL_LOCAL_PATH = MODEL_FILE_NAME
LABELS_PATH = 'sw_day_label_encoder.pkl'

app = Flask(__name__)

@app.route('/', methods=['POST'])
def index():
  payload = json.loads(request.get_data().decode('utf-8'))
  print "Inside \n"
  print payload
  prediction = predict(json.loads(payload['payload']))
  data = {}
  data['data'] = prediction[-1]
  return load_encoder().inverse_transform(data['data'])+"\n"
  # return json.dumps(data)

def load_encoder():
    dd = joblib.load(LABELS_PATH) 
    return dd['type']

def load_model():
  conn = S3Connection()
  bucket = conn.get_bucket(BUCKET_NAME)
  key_obj = Key(bucket)
  key_obj.key = MODEL_FILE_NAME

  contents = key_obj.get_contents_to_filename(MODEL_LOCAL_PATH)
  return joblib.load(MODEL_LOCAL_PATH)

def predict(data):
  # Process your data, create a dataframe/vector and make your predictions
  final_formatted_data = data
  return load_model().predict(final_formatted_data)