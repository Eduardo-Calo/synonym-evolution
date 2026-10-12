import requests
import json
import os
import sys
import zipfile
from pathlib import Path
from tqdm import tqdm
import random
import time
import io
import csv
from datetime import datetime, timezone, timedelta

with open('qualtrics_config.json') as f:
    qualtrics_config = json.load(f)

HEADERS = {
    "x-api-token": qualtrics_config['API_TOKEN'],
    "content-type": "application/json"
}

# Directory where survey exports and parsed results are stored
SURVEYS_DIR = os.path.join('..', 'data', 'surveys')

def parse_results(survey_id):

    def get_ids(k):
        D = {}
        k_split = k.split('/')
        type_, sentence_id = k_split
        D['idx'] = sentence_id
        D['sentence_id'] = sentence_id
        D['type'] = type_
        return D

    survey_dir = os.path.join(SURVEYS_DIR, survey_id)
    csv_file_path = os.path.join(survey_dir, [fname for fname in os.listdir(survey_dir) if fname.endswith('.csv')][0])

    with open(os.path.join(survey_dir, 'questionid2tag.json')) as f:
        question2tag = json.load(f)

    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        col2k = {}
        k2col = {}
        A = {}

        for j,row in enumerate(reader):
            if j == 2:
                for i,v in enumerate(row):
                    v = json.loads(v)
                    column_id = v['ImportId']
                    if "choiceId" in v:
                        column_id = column_id + f'_{v["choiceId"]}'
                    col2k[i] = column_id
                    k2col[column_id] = i
            elif j >=3:
                annotator = row[k2col['PROLIFIC_PID']]
                for i,v in enumerate(row):
                    question_id = col2k[i].split('_')[0]
                    if question_id in question2tag:
                        idx = question2tag[question_id]
                        D = get_ids(idx)
                        idx = '/'.join(idx.split('/')[1:])
                        if 'sense_id' in D:
                            idx = '/'.join(idx.split('/')[:-1])
                        if not idx in A:
                            A[idx] = {'annotators':[], 'answers':[], 'timing': {}}

                        if not annotator in A[idx]['annotators']:
                            A[idx]['annotators'].append(annotator)

                        if D['type'] == 'timing':
                            attribute = '_'.join(col2k[i].split('_')[1:])
                            if not attribute in A[idx]['timing']:
                                A[idx]['timing'][attribute] = []
                            try:
                                A[idx]['timing'][attribute].append(float(v))
                            except:
                                pass
                            
                        elif D['type'] == 'questions':
                            question_id_split = col2k[i].split('_')
                            A[idx]['answers'].append(int(v))


        with open(os.path.join(survey_dir, 'results.json'),'w+') as g:
            json.dump(A,g, indent=4)



def export_survey(survey_id):
    # Setting static parameters
    requestCheckProgress = 0.0
    retry_time = 10
    progressStatus = "inProgress"
    url = f"{qualtrics_config['BASE_URL']}/surveys/{survey_id}/export-responses/"


    # Step 1: Creating Data Export
    data = {
            "format": "csv"
           }

    downloadRequestResponse = requests.request("POST", url, json=data, headers=HEADERS)
    print(downloadRequestResponse.json())

    try:
        progressId = downloadRequestResponse.json()["result"]["progressId"]
    except KeyError:
        print(downloadRequestResponse.json())
        sys.exit(2)
        
    fileId = None

    max_retries = 5
    retry_count = 0

    # Step 2: Checking on Data Export Progress and waiting until export is ready
    while progressStatus != "complete" and progressStatus != "failed" and fileId is None:
        if fileId is None:
            print("File not ready")
        else:
            print("ProgressStatus=", progressStatus)
        
        requestCheckUrl = url + progressId
        requestCheckResponse = requests.request("GET", requestCheckUrl, headers=HEADERS)
        
        try:
            fileId = requestCheckResponse.json()["result"]["fileId"]
        except KeyError:
            1==1

        print(requestCheckResponse.json())
        requestCheckProgress = requestCheckResponse.json()["result"]["percentComplete"]
        print("Download is " + str(requestCheckProgress) + " complete")
        progressStatus = requestCheckResponse.json()["result"]["status"]

        if progressStatus not in ["complete", "failed"]:
            # Implement exponential backoff with a maximum number of retries
            retry_count += 1
            if retry_count > max_retries:
                print("Exceeded maximum retries. Exiting.")
                sys.exit(1)
            
            # Calculate the next sleep interval using exponential backoff (The first retry will occur after 200 seconds, the second after 400 seconds, the third after 600 seconds, and so on, until the max_retries limit is reached)
            sleep_interval = retry_time ** retry_count
            print(f"Retrying in {sleep_interval} seconds...")
            time.sleep(sleep_interval)

    # Step 3: Downloading file
    requestDownloadUrl = url + fileId + '/file'
    requestDownload = requests.request("GET", requestDownloadUrl, headers=HEADERS, stream=True)

    # Step 4: Unzipping the file
    zipfile.ZipFile(io.BytesIO(requestDownload.content)).extractall(os.path.join(SURVEYS_DIR, survey_id))
    print('Complete')



if __name__ == "__main__":
    survey_id = sys.argv[1]
    url = f"{qualtrics_config['BASE_URL']}/survey-definitions"
    response = requests.get(f'{url}/{survey_id}', headers=HEADERS)
    response = response.json()["result"]
    questions = response['Questions']
    D = {}
    for question_id in questions:
        D[question_id] = questions[question_id]['DataExportTag']
    # Ensure base surveys directory and survey-specific directory exist
    if not os.path.exists(SURVEYS_DIR):
        os.makedirs(SURVEYS_DIR, exist_ok=True)
    survey_dir = os.path.join(SURVEYS_DIR, survey_id)
    if not os.path.exists(survey_dir):
        os.makedirs(survey_dir, exist_ok=True)
    with open(os.path.join(survey_dir, 'questionid2tag.json'),'w+') as f:
        json.dump(D,f)
    export_survey(survey_id)
    parse_results(survey_id)
    print('Completed!')
