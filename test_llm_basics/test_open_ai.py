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
    Extract data and insights from the following image. 
    Make use of **Header** and the **Narrative Text** if they are present and associate more information with the image.
    The aim is to index the extracted information for retrieval in an augmented generator model.
    Header: {header}
    Narrative Text: {narrative_text}

    Rules:
    If the image is low resolution and unclear, try your best to extract as much data and insights as possible.
    If the data from the image does not make any sense, return "UNEXTRACTABLE_DATA" in response.
    If the request can not be proccessed return "UNPROCESSABLE_ENTITY" in response.
    Response should be precize and to the point.
    Following is an example response for when data can be extracted from image:
    The sales distribution for the world is as follows: Asia contributed 30%, Europe is contributing 25% and rest of the world contributed 45%.
    Asia and europe contribute over 50% of the sales across the world.
    
    Following should be the response when the image does not make any sense:
    UNEXTRACTABLE_DATA
    
    """
    encoded_image = encode_image(img_url)
    print(encoded_image)
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


# """
# Given a header, narrative text, and an image, the language model's primary objective is to parse the image and index the extracted data for retrieval in augmented generation.
# If the image content is clear, focus on extracting information directly from the image.
# In cases where the image is unclear or ambiguous, utilize the provided narrative text and header to improve the extraction results.
# Avoid generating statements explicitly describing the image, and refrain from providing responses such as "The image displays..." or "I'm sorry, but I can't assist with that request." Instead, prioritize concise and relevant information.
# If the image itself does not provide any meaningful data, return an empty string.
# The goal is to generate responses suitable for indexing in a vector store, with a focus on informative and contextually relevant insights.
# """

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
    Following is an example response for ques list being generated from the **context**:
    What was the dis
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


def get_table_description(table, metadata={}, header=None, narrative_text=None):
    PROMPT = f"""
    From the given **table**, provide its detailed summary and insights that can be easily indexed for a Retrieval Augmented Genration Model. 
    The response output should be in the form paragraph(s).
    Header: {header}
    Narrative Text: {narrative_text}
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


# a = 'X'
# b = 'Y'
# page_content = get_image_description(
#     'figures/figure-1-2.jpg', a, b)
# # page_content = """
# Header: Sales by Region 9M/2020 (9M/2019)

# Narrative Text: The following graphic shows the still balanced split of sales by region.

# Extracted Data and Insights:

# - Europe accounts for 35.9% of sales, a slight decrease from 37.2% in the previous year.
# - The Americas contribute 26.5% to the sales, down from 26.9% the previous year.
# - Asia represents 37.6% of sales, up from 35.9% in the prior year.
# - The total sales amount is 453.9 million Euros, down from 467.3 million Euros the previous year.
# - The sales by region are relatively balanced, with Europe and Asia having a more significant share compared to The Americas.
# - A subtle shift in sales distribution is noticed, with Europe and The Americas seeing a slight decline while Asia experiences an increase.
# """
# print(page_content)
# print(custom_parser.parse_paragraph(page_content))
