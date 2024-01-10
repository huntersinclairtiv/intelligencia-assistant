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
    From the given **context**, provide a detailed summary and insights that can be easily indexed for a Retrieval Augmented Genration Model. 
    The response should be precise and to the point.
    The response must not contain any explicit mention of any context, summary or insights.
    The response should not contain any bullet points or numbering for insights or any other aspect of response.
    context: {paragraph}
    # """
    # QUES_LIST_PROMPT = f"""
    # Provide a list of questions that can be answered using the following **context**.
    # The questions must cover various aspects of the context.
    # The response should be precise and to the point.
    # The response should not contain any bullet points or numbering for questions.
    # context: {paragraph}
    # Generate questions that can be answered directly from the provided text.
    # Ensure that the questions are relevant and focused on extracting information present in the context.
    # Avoid generating questions that require external knowledge or assumptions.
    # Following is an example response for when some questions can be generated using the given context:
    # What conclusions can be drawn about the market strategy based on the balanced regional sales portrayed in the graphic? What are the stats for number of shares issued?
    # What was the value of a share in Sept 2020?
    # Following should be the response when no question can be generated using the given context:
    # UNEXTRACTABLE_DATA
    # """
    QUES_LIST_PROMPT = f"""
    Given Text:
    {paragraph}
    
    Prompt:
    Generate questions that can be answered directly from the provided text and are very factual and informative.
    Ensure that the questions are relevant and focused on extracting information present in the context.
    Avoid generating questions that require external knowledge or assumptions.
    The response should not contain any bullet points or numbering for questions.
    """
    # If the paragraph contains no relevant information that might be used for answering, return "UNEXTRACTABLE_DATA"
    messages = [
        {
            "role": "system",
            "content": "You are a trained assistant that extracts the questions that can be answered using a given context"
        },
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
        print(response.choices[0].message.content)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": QUES_LIST_PROMPT}
            ],
        }
    ]
    response = get_openai_respone(messages, model)
    print(response.choices[0].message.content)
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
    The response should be precise and to the point.
    The response must not contain any explicit mention of any table, header, narrative text, summary or insights.
    """
    QUES_LIST_PROMPT = f"""
    Provide a list of questions that can be answered using the following **table**.
    Provide at most top 5 most relevant questions. The questions must cover various aspects of the table.
    The response should be precise and to the point.
    The response should not contain any bullet points or numbering for questions.
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
    print(response.choices[0].message.content)
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
    print(response.choices[0].message.content)
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


# a = 'Key Figures'
# b = """"""
# # page_content = get_image_description(
# #     'figures/figure-1-2.jpg', a, b)

# # print(page_content)
# # print(custom_parser.parse_paragraph(page_content))

# table = """
# <table><thead><th></th><th></th><th>Q3 2020</th><th>Q3 2019</th><th>Change</th><th>2020</th><th>2019</th><th>Change</th></thead><tr><td colspan="8">Sales and profit</td></tr><tr><td>Total sales</td><td>K€</td><td>152,007</td><td>156,225</td><td>-2.7%</td><td>453,861</td><td>467,330</td><td>-2.9%</td></tr><tr><td>Germany</td><td>K€</td><td>24,196</td><td>25,056</td><td>-3.4%</td><td>70,565</td><td>72,664</td><td>-2.9%</td></tr><tr><td>Other countries</td><td>K€</td><td>127,811</td><td>131,169</td><td>-2.6%</td><td>383,296</td><td>394,669</td><td>-2.9%</td></tr><tr><td>Operating profit</td><td>K€</td><td>16,138</td><td>16,059</td><td>0.5%</td><td>35,686</td><td>48,904</td><td>-27.0%</td></tr><tr><td>EBIT margin</td><td>%</td><td>10.6</td><td>10.3</td><td>0.3 Pp</td><td>7.9</td><td>10.5</td><td>-2.6 Pp</td></tr><tr><td>Net income</td><td>K€</td><td>11,279</td><td>11,427</td><td>-1.3%</td><td>24,810</td><td>34,736</td><td>-28.6%</td></tr><tr><td>Return on sales</td><td>%</td><td>7.4</td><td>7.3</td><td>0.1 Pp</td><td>5.5</td><td>7.4</td><td>-1.9 Pp</td></tr><tr><td>Operating cash flow</td><td>K€</td><td>14,070</td><td>12,471</td><td>12.8%</td><td>36,957</td><td>35,513</td><td>4.1%</td></tr><tr><td>Capital expenditures</td><td>K€</td><td>6,403</td><td>6,273</td><td>2.1%</td><td>19,675</td><td>19,307</td><td>1.9%</td></tr><tr><td>Earnings per share</td><td>€</td><td>1.14</td><td>1.16</td><td>-1.7%</td><td>2.51</td><td>3.52</td><td>-28.4%</td></tr><tr><td colspan="8">Workforce</td></tr><tr><td>Workforce (average)</td><td></td><td>3,334</td><td>3,243</td><td>2.8%</td><td>3,317</td><td>3,244</td><td>2.3%</td></tr><tr><td>Germany</td><td></td><td>1,124</td><td>1,092</td><td>2.9%</td><td>1,119</td><td>1,072</td><td>44%</td></tr><tr><td>Other countries</td><td></td><td>2,210</td><td>2,152</td><td>2.7%</td><td>2,198</td><td>2,172</td><td>1.2%</td></tr><tr><td rowspan="2">Sales per employee</td><td>K€</td><td>46</td><td>48</td><td>-4.8%</td><td>137</td><td>144</td><td>-5.5%</td></tr><tr><td></td><td></td><td></td><td></td><td>Sept. 30, 2020</td><td colspan="2" rowspan="2">December 31, 2019</td><td>Change</td></tr><tr><td colspan="8">Balance sheet</td><td colspan="2" rowspan="2">659,575</td><td colspan="2"></td></tr><tr><td>Balance sheet total</td><td></td><td></td><td>K€</td><td>659,111</td><td colspan="2"></td><td>-0.1%</td></tr><tr><td colspan="3">Cash and cash equivalents</td><td>K€</td><td>112,217</td><td colspan="2">111,980</td><td>0.2%</td></tr><tr><td colspan="2">Number of shares issued</td><td></td><td colspan="2">9,867,659</td><td colspan="2">9,867,659</td><td>-</td></tr><tr><td>Shareholders' equity</td><td></td><td></td><td colspan="2">K€ 396,377</td><td colspan="2">393,445</td><td>-0.9%</td></tr></table>
# """

# print(get_table_description(table, {}, a, b))

# p = """
# Interim Management Report:
# The following graphic shows the still balanced split of sales by region.
# """
# get_paragraph_description(p, {}, False)
