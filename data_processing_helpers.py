from json.decoder import JSONDecodeError
import json
import re


def flatten_dict(input_dict):
    result = {
        'text': input_dict['text'],
        'object': None,
        'object_type': None,
        'predicate': None,
        'subject': None,
        'subject_type': None
    }
    
    if input_dict['spo_list']:
        spo = input_dict['spo_list'][0]  # Assuming we're only dealing with the first item in spo_list
        result.update({
            'object': spo['object'],
            'object_type': spo['object_type'],
            'predicate': spo['predicate'],
            'subject': spo['subject'],
            'subject_type': spo['subject_type']
        })
    
    return result


def parse_json_markdown(md_string):
    start = md_string.find('{')
    # Find the ending position of the JSON part
    end = md_string.rfind('}') + 1

    # Extract the JSON string
    json_string = md_string[start:end]

    # Parse the JSON string into a Python dictionary
    try:
        data = json.loads(json_string)
    except JSONDecodeError as e:
        data = eval(json_string)

    # Print the dictionary
    return data


def load_schema(data_path):
    with open(data_path, "r") as f:
        schema_list = [eval(x) for x in f.readlines()]

    return schema_list


def load_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as file:
        content = file.read()

    json_strings = content.strip().split('\n')
    data_list = [json.loads(json_str) for json_str in json_strings]

    return data_list


def process_raw_data(raw_data_list):
    train_list = []

    for row in raw_data_list:
        record = {"spo": {}}

        record['text'] = row['text']
        record['spo']['subject'] = row["spo_list"][0]["subject"]
        record['spo']['subject_type'] = row["spo_list"][0]["subject_type"]
        record['spo']['object'] = row["spo_list"][0]["object"]
        record['spo']['object_type'] = row["spo_list"][0]["object_type"]
        record['spo']['predicate'] = row["spo_list"][0]["predicate"]
        train_list.append(record)
    
    return train_list


def wrap_numbers_with_backticks(text):
    return re.sub(r'(\d+)', r'`\1`', text)