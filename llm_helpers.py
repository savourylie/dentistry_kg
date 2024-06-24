import random
from collections import namedtuple
from typing import List

from openai import OpenAI
import pandas as pd
import torch

from data_processing_helpers import parse_json_markdown


def create_chat_completions(model, system_message, user_message, api_key, base_url=None, max_tokens=1024, temperature=0.7):
    client = OpenAI(api_key=api_key, base_url=base_url)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
      ],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False
    )

    return response.choices[0].message.content


def create_prompt(train_list, val_list, schema_list, sample_size=10):
    Template = namedtuple("Template", ["system_message", "user_message", "label"])

    for row in val_list:
        random_sample = random.choices(train_list, k=sample_size)

        system_message = "你是专门进行实体抽取的专家。请在 schema 中定义的范畴，参考范例中的格式，从 user 给定句子中抽取出符合 schema 定义的实体，不存在的实体类型返回空列表。请按照JSON字符串的格式回答。\n\n"

        system_message += "Schema: \n"
        for schema in schema_list:
            system_message += str(schema) + "\n"
        
        system_message += "\nSamples: \n"
        for record in random_sample:
            system_message += f"{record['text']} ==> {str(record['spo'])}" + "\n"

        user_message = row['text'] + " ==> "

        template = Template(
            system_message=system_message, 
            user_message=user_message,
            label=row['spo'])

        yield (template, row['text'])


def process_template(
        prompt,
        model,
        api_key,
        base_url,
    ):

    template, original_text = prompt
    system_message, user_message, label = template.system_message, template.user_message, template.label

    try:
        result = create_chat_completions(
            model=model, 
            system_message=system_message, 
            user_message=user_message, 
            api_key=api_key, 
            base_url=base_url
        )
        result_parsed = parse_json_markdown(result)
        result_parsed['text'] = original_text

        # return (result_parsed, label)
        return ('good', result_parsed)
    
    except Exception as e:
        return ('error', (result, label))
    

def sentences2embeddings(sentence: str, model, tokenizer) -> pd.Series:
    sentence_list = [sentence]
    # Tokenize sentences
    encoded_input = tokenizer(sentence_list, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]
    # normalize embeddings
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

    embeddings = sentence_embeddings.numpy()[0].tolist()

    return embeddings