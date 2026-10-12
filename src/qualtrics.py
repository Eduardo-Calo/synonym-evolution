import requests
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
import random
from datetime import datetime, timezone, timedelta
import time
import re

def create_survey(dataset_name, language, qualtrics_config, HEADERS):
    url = f"{qualtrics_config['BASE_URL']}/survey-definitions"

    data = {"SurveyName": f'{dataset_name}', "Language": language, "ProjectCategory": "CORE"}

    response = requests.post(url, json=data, headers=HEADERS)

    survey_id = response.json()["result"]["SurveyID"]

    response = requests.get(f'{url}/{survey_id}/options', json=data, headers=HEADERS)

    option_payload = json.loads(response.text)["result"]

    option_payload["ProgressBarDisplay"] =  "VerboseText"

    # Prolific integration: redirect participants back to Prolific on completion
    prolific_url = qualtrics_config.get('PROLIFIC_COMPLETION_URL', '')
    if prolific_url:
        option_payload["SurveyTermination"] = "Redirect"
        option_payload["EOSRedirectURL"] = prolific_url
    else:
        option_payload["SurveyTermination"] = "ResponseSummary"

    response = requests.put(f'{url}/{survey_id}/options', json=option_payload, headers=HEADERS)

    return survey_id


def add_prolific_embedded_data(survey_id, qualtrics_config, HEADERS):
    """Add PROLIFIC_PID, STUDY_ID, SESSION_ID embedded data to the survey flow."""
    url = f"{qualtrics_config['BASE_URL']}/survey-definitions/{survey_id}/flow"
    response = requests.get(url, headers=HEADERS)
    flow = response.json()["result"]

    embedded_data_element = {
        "Type": "EmbeddedData",
        "FlowID": "FL_prolific",
        "EmbeddedData": [
            {"Description": "PROLIFIC_PID", "Type": "Recipient", "Field": "PROLIFIC_PID", "VariableType": "String", "DataVisibility": [], "AnalyzeText": False},
            {"Description": "STUDY_ID", "Type": "Recipient", "Field": "STUDY_ID", "VariableType": "String", "DataVisibility": [], "AnalyzeText": False},
            {"Description": "SESSION_ID", "Type": "Recipient", "Field": "SESSION_ID", "VariableType": "String", "DataVisibility": [], "AnalyzeText": False}
        ]
    }

    # Insert embedded data at the very beginning of the flow
    flow["Flow"].insert(0, embedded_data_element)

    requests.put(url, headers=HEADERS, json=flow)
    return flow


def create_contacts(survey_id, language, qualtrics_config, HEADERS):
    url = f"{qualtrics_config['BASE_URL']}/directories/{qualtrics_config['DIRECTORY_ID']}/mailinglists"

    data = {"name": f'ML_{survey_id}'}

    response = requests.post(url, json=data, headers=HEADERS)
    mailingListId = json.loads(response.text)["result"]["id"]
    contacts = []
    
    for contact in qualtrics_config['CONTACTS'][language]:
        contact_payload = {
          "firstName": contact['name'],
          "lastName": contact['lastname'],
          "email": contact['email'],
          "extRef": contact['email'],
          "language": language,
          "unsubscribed": False
        }

        url = f"{qualtrics_config['BASE_URL']}/directories/POOL_0HRTkdQP4ZBNcbP/mailinglists/{mailingListId}/contacts"
        response = requests.post(url, headers=HEADERS, json=contact_payload)
        contact_id = json.loads(response.text)["result"]
        contacts.append(contact_id)
    return contacts, mailingListId


def add_instructions_page(survey_id, language, qualtrics_config, HEADERS, mode='semantic'):
    url = f"{qualtrics_config['BASE_URL']}/survey-definitions/{survey_id}/blocks"

    block_payload = {
      "Type": "Standard",
      "Description": "Instructions Block",
      "Options": {
            "BlockLocking": "false",
            "RandomizeQuestions": "false",
            "BlockVisibility": "Expanded",
    }}

    instruction_key = 'INSTRUCTION_TEXT_BINARY' if mode == 'binary' else 'INSTRUCTION_TEXT'

    question_payload = {
        "QuestionText": qualtrics_config[instruction_key],
        "DataExportTag": "instructions",
        "QuestionType": "DB",
        "Selector": "TB",
        "Configuration": {
            "QuestionDescriptionOption": "UseText"
        }
    }

    response = requests.post(url, headers=HEADERS, json=block_payload)
    block_id = json.loads(response.text)["result"]["BlockID"]

    url_question = f"{qualtrics_config['BASE_URL']}/survey-definitions/{survey_id}/questions?blockId={block_id}"
    question_id = json.loads(requests.post(url_question, headers=HEADERS, json=question_payload).text)["result"]["QuestionID"]
    return block_id, question_id


def add_consent_page(survey_id, language, qualtrics_config, HEADERS):
    url = f"{qualtrics_config['BASE_URL']}/survey-definitions/{survey_id}/blocks"

    block_payload = {
      "Type": "Standard",
      "Description": "Consent Block",
      "Options": {
            "BlockLocking": "false",
            "RandomizeQuestions": "false",
            "BlockVisibility": "Expanded",
    }}

    consent_text = qualtrics_config['CONSENT_TEXT'] + "<br><br><b>By continuing, you consent to participate in this study.</b>"

    question_payload = {
        "QuestionText": consent_text,
        "DataExportTag": "consent",
        "QuestionType": "DB",
        "Selector": "TB",
        "Configuration": {
            "QuestionDescriptionOption": "UseText"
        }
    }

    response = requests.post(url, headers=HEADERS, json=block_payload)
    block_id = json.loads(response.text)["result"]["BlockID"]

    url_question = f"{qualtrics_config['BASE_URL']}/survey-definitions/{survey_id}/questions?blockId={block_id}"
    question_id = json.loads(requests.post(url_question, headers=HEADERS, json=question_payload).text)["result"]["QuestionID"]

    return block_id, question_id


def add_question(survey_id, sentence_idx, sentence, mode, question_num, question2idx, qualtrics_config, HEADERS, attention_check=False):
    url_block = f"{qualtrics_config['BASE_URL']}/survey-definitions/{survey_id}/blocks"

    ann_choices = {}
    if mode == 'semantic':
        ann_choices = {
            '1': '<b>IDENTICAL</b>',
            '2': '<b>CLOSELY RELATED</b>',
            '3': '<b>DISTANTLY RELATED</b>',
            '4': '<b>UNRELATED</b>',
            '5': '<b>CANNOT DECIDE</b>'
        }
        if attention_check:
            question_text = "Choose <b>IDENTICAL</b>."
        else:
            question_text = "Please, rate the degree of <u>meaning relatedness</u> between these two definitions."
            question_text = question_text + '</br></br>' + sentence
    elif mode == 'binary':
        ann_choices = {
            '1': '<b>SIMILAR</b>',
            '2': '<b>DISSIMILAR</b>',
            '3': '<b>CANNOT DECIDE</b>'
        }
        if attention_check:
            question_text = "Choose <b>SIMILAR</b>."
        else:
            question_text = "Please, rate whether these two definitions are <u>meaningfully similar</u>."
            question_text = question_text + '</br></br>' + sentence

    choices = {k: {'Display': ann_choices[k], 'ExclusiveAnswer': True} for k in sorted(ann_choices.keys())}
    choices_order = [k for k in sorted(choices.keys())]
    block_payload = {
      "Type": "Standard",
      "Description": f"blocks/{sentence_idx}",
      "Options": {
            "BlockLocking": "false",
            "RandomizeQuestions": "false",
            "BlockVisibility": "Expanded",
    }}

    question_payload = {
        "QuestionText": question_text,
        "DataExportTag": f"questions/{sentence_idx}",
        "QuestionType": "MC",
        "Selector": "SAVR",  
        "SubSelector": "TX",
        "Configuration": {
            "QuestionDescriptionOption": "UseText"
        },
        "Choices": choices,
        "ChoiceOrder": choices_order,
        "QuestionDescription": f"Meaning of '{sentence_idx}'",
        'Validation': {'Settings': {'ForceResponse': 'ON', 'ForceResponseType': 'ON', 'Type': 'None'}}
    }

    question_timing = {
            'QuestionText': 'Timing', 
            'DefaultChoices': False, 
            'DataExportTag': f'timing/{sentence_idx}', 
            'QuestionType': 'Timing', 
            'Selector': 'PageTimer', 
            'DataVisibility': {'Private': False, 'Hidden': False}, 
            'Configuration': {'QuestionDescriptionOption': 'UseText', 'MinSeconds': '0', 'MaxSeconds': '0'}, 
            'QuestionDescription': 'Timing', 
            'Choices': {
            '1': {'Display': 'First Click'}, 
            '2': {'Display': 'Last Click'}, '3': {'Display': 'Page Submit'}, 
            '4': {'Display': 'Click Count'}}, 
            'GradingData': [], 
            'Language': [], 
            'NextChoiceId': 16, 
            'NextAnswerId': 1, 
            'QuestionText_Unsafe':'Timing'
    }
    
    response = requests.post(url_block, headers=HEADERS, json=block_payload)
    block_id = json.loads(response.text)["result"]["BlockID"]

    url_question = f"{qualtrics_config['BASE_URL']}/survey-definitions/{survey_id}/questions?blockId={block_id}"
    question_id = json.loads(requests.post(url_question, headers=HEADERS, json=question_timing).text)["result"]["QuestionID"]

    url_question = f"{qualtrics_config['BASE_URL']}/survey-definitions/{survey_id}/questions?blockId={block_id}"
    question_id = json.loads(requests.post(url_question, headers=HEADERS, json=question_payload).text)["result"]["QuestionID"]


    question2idx[question_num] = {'sentence_idx':sentence_idx}
    question_num = question_num + 1

    return question_num, question2idx

def retry_add_question(survey_id, idx, sentence, mode, question_num, question2idx, qualtrics_config, HEADERS):
    try:
        question_num, question2idx = add_question(survey_id, idx, sentence, mode, question_num, question2idx, qualtrics_config, HEADERS)
        return question_num, question2idx
    except Exception as e:
        print(e)
        print('Waiting 10 seconds..')
        time.sleep(10)
        retry_add_question(survey_id, idx, sentence, mode, question_num, question2idx, qualtrics_config, HEADERS)

def create_survey_and_add_questions(file_name, dataset_name, language, qualtrics_config, HEADERS):
    survey_id = create_survey(dataset_name, language, qualtrics_config, HEADERS)
    add_prolific_embedded_data(survey_id, qualtrics_config, HEADERS)
    question2idx = {}
    question_num = 1
    
    mode = 'binary'
    idx2sentence = {}

    with open(file_name) as f:
        for line in f:
            line = json.loads(line)
            idx = line['idx']
            word1 = line.get('word1', '')
            word2 = line.get('word2', '')
            definition1, definition2 = line['definition1'], line['definition2']
            sample_sentence1 = line.get('sample_sentence1', '')
            sample_sentence2 = line.get('sample_sentence2', '')
            text = ""
            if word1:
                text = text + '<b>Word 1</b>: ' + word1 + '</br>'
            text = text + '<b>Definition 1</b>: ' + definition1 + '</br>'
            if sample_sentence1:
                text = text + '<b>Sample sentence 1</b>: ' + sample_sentence1 + '</br></br>'
            else:
                text = text + '</br>'
            if word2:
                text = text + '<b>Word 2</b>: ' + word2 + '</br>'
            text = text + '<b>Definition 2</b>: ' + definition2 + '</br>'
            if sample_sentence2:
                text = text + '<b>Sample sentence 2</b>: ' + sample_sentence2
            idx2sentence[idx] = text 

    idxs = list(idx2sentence.keys())
    random.shuffle(idxs)

    # Add consent page as the first block
    add_consent_page(survey_id, language, qualtrics_config, HEADERS)

    # Add instructions page as the second block
    add_instructions_page(survey_id, language, qualtrics_config, HEADERS, mode=mode)

    for idx in tqdm(idxs,f'Uploading questions...'):
        try:
            question_num, question2idx = add_question(survey_id, idx, idx2sentence[idx], mode, question_num, question2idx, qualtrics_config, HEADERS)
    
            # add attention check question every 11 questions
            if question_num % 12 == 0:
                question_num, question2idx = add_question(survey_id, f'{idx}_attention_check', '', mode, question_num, question2idx, qualtrics_config, HEADERS, attention_check=True)
        except Exception as e:
            print(e)
            print('Waiting 10 seconds..')
            time.sleep(10)
            retry_add_question(survey_id, idx, idx2sentence[idx], mode, question_num, question2idx, qualtrics_config, HEADERS)
    return survey_id


def activate_survey(survey_id, qualtrics_config, HEADERS):
    """Activate (publish) a Qualtrics survey so it can collect responses."""
    url = f"{qualtrics_config['BASE_URL']}/surveys/{survey_id}"
    resp = requests.put(url, headers=HEADERS, json={"isActive": True})
    if not resp.ok:
        print(f"Warning: failed to activate survey ({resp.status_code}): {resp.text}")
    else:
        print(f"       Qualtrics survey activated.")
    return resp


if __name__ == "__main__":
    language = "EN"
    file_name = sys.argv[1]
    dataset_name = f'definition_similarity_{file_name.split("/")[-1].replace(".jsonl", "")}'
    with open('qualtrics_config.json') as f:
        qualtrics_config = json.load(f)
    HEADERS = {
        "x-api-token": qualtrics_config['API_TOKEN'],
        "content-type": "application/json"
    }
    survey_id = create_survey_and_add_questions(file_name, dataset_name, language, qualtrics_config, HEADERS)  # keep track of this id to retrieve results later
    # activate survey (make it live) so it can collect responses
    activate_survey(survey_id, qualtrics_config, HEADERS)
    print('Completed!', 'Survey ID:', survey_id, 'Batch name:', dataset_name)