import sys

from google.cloud import automl_v1beta1
from google.cloud.automl_v1beta1.proto import service_pb2
from google.cloud.automl_v1beta1 import PredictionServiceClient
import pandas as pd
import time
def get_prediction(content, project_id, model_id):
  #storage_client = storage.Client.from_service_account_json(r"C:\Users\dell laptop\AppData\Roaming\gcloud\chicago-hospitals-social-media-7892696dfd9c.json")
  #credentials = service_account.Credentials.from_service_account_file("C:\\Users\\dell laptop\\AppData\\Roaming\\gcloud\\chicago-hospitals-social-media-7892696dfd9c.json")
  #prediction_client = automl_v1beta1.PredictionServiceClient(credentials=credentials)
  prediction_client = PredictionServiceClient.from_service_account_file("C:\\Users\\dell laptop\\AppData\\Roaming\\gcloud\\chicago-hospitals-social-media-7892696dfd9c.json")
  name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)
  payload = {'text_snippet': {'content': content, 'mime_type': 'text/plain' }}
  params = {}
  request = prediction_client.predict(name, payload, params)
  return request  # waits till request is returned

project_id = "chicago-hospitals-social-media"
model_id = "TCN1654989300174946304"

df = pd.read_csv("E:\\Harshitha\\MastersProgram\\FallTerm\\Strategy&Policy\\FieldWork\\Prediction\\input2.5k_7th_iteration.csv", header=None)
output = []

for index, line in df.iterrows():
  print(line[0] + '\n')
  time.sleep(1.1)
  prediction = get_prediction(str(line[0]), project_id,  model_id)
  row = {"Test Case": str(line[0])}
  for payload in prediction.payload:
    print(payload)
    row[payload.display_name] = payload.classification.score
  output.append(row)
outputDf1=pd.DataFrame(output)
#outputDf = pd.DataFrame(output, columns=["Test Case", "ch_event", "ch_literacy", "ch_promotion", "ch_testimonial", "ch_notrelated", "ch_healthalert"])
outputDf1.to_csv("E:\\Harshitha\\MastersProgram\\FallTerm\\Strategy&Policy\\FieldWork\\Prediction\\output2.5k_7th_iteration.csv", index=False)
