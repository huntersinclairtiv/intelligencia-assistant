# import base64
# import json
# import os
# import requests
# from requests.structures import CaseInsensitiveDict

# # Endpoint and options
# completions_endpoint = "https://api.openai.com/v1/chat/completions"
# headers = CaseInsensitiveDict({"Content-Type": "application/json"})
# json_options = {
#     "indent": 2,
# }

# # Flag for running loop
# token = os.environ.get('OPEN_AI_KEY')
# file_path = 'figures/figure-14-4.jpg'
# try:
#     with open(file_path, "rb") as file:
#         image_bytes = file.read()
#         image_base64_string = base64.b64encode(image_bytes).decode("utf-8")
#         file_extension = file_path.split(".")[-1]
#         request = {
#             "model": "gpt-4-vision-preview",
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": f"Describe the image from the following message. {image_base64_string} ",
#                 }
#             ]
#         }
#         print(len(image_base64_string))
#         json_data = json.dumps(request, **json_options)
#         # print(json_data)
#         headers["Authorization"] = f"Bearer {token}"
#         response = requests.post(completions_endpoint,
#                                  headers=headers, data=json_data)
#         print(response.__dict__)
#         vision_response = response.json()
#         content = vision_response.get("choices", [{}])[
#             0].get("message", {}).get("content")
#         if content:
#             print(
#                 f"Here is a description of your provided image: \033[33m{content}\033[0m")
#             print()
#         else:
#             print("Unfortunately, there is no content available to display.")
# except Exception as ex:
#     print(f"Something went wrong: \033[31m{str(ex)}\033[0m")


# THIS SHOULD BE IMPLEMENTED USING PRE-BUILT CHAINS


from openai import OpenAI
import base64

import paragraph_parser as custom_parser


def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_openai_respone(messages, model='gpt-4-vision-preview', max_tokens=300):
    client = OpenAI()
    return client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )


def get_image_description(img_url, header=None, narrative_text=None):
    client = OpenAI()

    # PROMPT = f"""
    # From the following image, extract data and insights such that they can be indexed for retrieval augmeneted generator model.
    # Also make use of the Header and the narrative text for the image.
    # Header: {header}
    # Narrative Text: {narrative_text}
    # If the Header and Narrative text do not make any sense, gather insights based on the image itself.
    # If the image is low resolution and unclear, try your best to extract as much data and insights as possible.
    # """
    PROMPT = f"""
    From the following image, along with an optional header and narrative text, extract relevant data and insights.
    Ensure flexibility in handling cases where the header and narrative text might be absent or unclear. 
    The aim is to index the extracted information for retrieval in an augmented generator model.
    Header: {header}
    Narrative Text: {narrative_text}
    If the image is low resolution and unclear, try your best to extract as much data and insights as possible.
    If no data can be extracted return empty string.
    """
    encoded_image = encode_image(img_url)
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
              "role": "user",
              "content": [
                  {"type": "text", "text": PROMPT},
                  {
                      "type": "image_url",
                      "image_url": {
                          "url": f"data:image/jpeg;base64,{encoded_image}",
                      },
                  },
              ],
            }
        ],
        max_tokens=300,
    )

    return response.choices[0].message.content


def get_paragraph_description(paragraph, metadata={}, add_summary=False):
    PROMPT = f"""
    From the given **context**, provide its detailed summary and insights that can be easily indexed for a Retrieval Augmented Genration Model. 
    The response output should be in the form paragraph(s). 
    context: {paragraph}
    """
    QUES_LIST_PROMPT = f"""
    From the given **context**, provied a list of questions that can be answered using the context.
    The response output should be in the form of paragraph(s). 
    Provide at most top 5 most relevant questions. 
    Do not proivde any additional context or statements besides the questions. 
    Also do not provide the answers to questions. 
    The response output should be in the form paragraph(s).
    context: {paragraph}
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT}
            ],
        }
    ]
    model = 'gpt-4-1106-preview'
    docs = []
    if add_summary:
        response = get_openai_respone(messages, model)
        docs = custom_parser.parse_paragraph(
            response.choices[0].message.content, metadata)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": QUES_LIST_PROMPT}
            ],
        }
    ]
    response = get_openai_respone(messages, model)
    docs.extend(custom_parser.parse_paragraph(
        response.choices[0].message.content, metadata))
    return docs


def get_table_description(table, metadata={}):
    PROMPT = f"""
    From the given **table**, provide its detailed summary and insights that can be easily indexed for a Retrieval Augmented Genration Model. 
    The response output should be in the form paragraph(s). 
    table: {table}
    """
    QUES_LIST_PROMPT = f"""
    From the given **table**, provied a list of questions that can be answered using the table.
    The response output should be in the form of paragraph(s).
    Provide at most top 5 most relevant questions. 
    Do not proivde any additional context or statements besides the questions. 
    Also do not provide the answers to questions. 
    The response output should be in the form paragraph(s). 
    table: {table}
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT}
            ],
        }
    ]
    model = 'gpt-4-1106-preview'
    response = get_openai_respone(messages, model)
    docs = custom_parser.parse_paragraph(
        response.choices[0].message.content, metadata)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": QUES_LIST_PROMPT}
            ],
        }
    ]
    response = get_openai_respone(messages, model)
    docs.extend(custom_parser.parse_paragraph(
        response.choices[0].message.content, metadata))
    return docs


# print(get_table_description("""<table><thead><th></th><th colspan="2">Three months ended September 30,</th><th colspan="2">Nine months ended September 30,</th></thead><thead><th></th><th>2020</th><th>2019</th><th>2020</th><th>2019</th></thead><tr><td></td><td>in K€</td><td>in K€</td><td>in K€</td><td>in K€</td></tr><tr><td>Asia</td><td>61,957</td><td>54,225</td><td>170,532</td><td>167,626</td></tr><tr><td>Europe</td><td>53,593</td><td>58,852</td><td>162,285</td><td>173,791</td></tr><tr><td>The Americas</td><td>36,175</td><td>43,107</td><td>120,702</td><td>125,810</td></tr><tr><td>Rest of the world</td><td>282</td><td>41</td><td>342</td><td>106</td></tr><tr><td>Total</td><td>152,007</td><td>156,225</td><td>453,861</td><td>467,333</td></tr></table>"""))


def get_llm_response(context, question):
    PROMPT = f"""
    Using the provided **context** answer the following **question**.
    Generate the response at best of your capabilities.
    If the question can not be answered using the provided **context** do not make up any response.
    context: {context}
    question: {question}
    """
    model = 'gpt-4-1106-preview'
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT}
            ],
        }
    ]
    response = get_openai_respone(messages, model)
    return response.choices[0].message.content
