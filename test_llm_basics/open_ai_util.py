# THIS SHOULD BE IMPLEMENTED USING BUILT-IN CHAINS
# MAKE USE OF SYSTEM MESSAGES FOR BETTER RESULTS

import base64
import json

from openai import OpenAI

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


def quest_list_from_context(context, metadata={}, header=None, narrative_text=None, model='gpt-4-1106-preview'):
    SYSTEM_PROMT = """
    Use the given **context** to generate a list of questions.
    Ensure that the questions are relevant and focused on extracting information present in the context.
    Avoid generating questions that require external knowledge or assumptions.
    The response should not contain any bullet points or numbering for questions.
    The respone should be an json array of strings where each string represents an individual question.
    """
    USER_PROMPT = f'context: {context}'
    #TODO Update the prompt to work with header and narrative text as well
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMT
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_PROMPT}
            ],
        }
    ]
    response = get_openai_respone(messages=messages, model=model)
    return json.loads(response.choices[0].message.content)


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
    QUES_LIST_PROMPT = {
        'SYSTEM': """
        Use the given **context** to generate a list of questions.
        Ensure that the questions are relevant and focused on extracting information present in the context.
        Avoid generating questions that require external knowledge or assumptions.
        The response should not contain any bullet points or numbering for questions.
        The respone should be an json array of strings where each string represents an individual question.
        """,
        'USER': f'context: {paragraph}'
    }

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
            "role": "system",
            "content": QUES_LIST_PROMPT['SYSTEM']
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": QUES_LIST_PROMPT['USER']}
            ],
        }
    ]
    response = get_openai_respone(messages, model)
    print(json.loads(response.choices[0].message.content))

    docs.extend(custom_parser.parse_paragraph(
        response.choices[0].message.content, metadata))
    return docs


def get_table_description(table, metadata={}, header=None, narrative_text=None):
    PROMPT = f"""
    From the given **table**, provide its detailed summary and insights that can be easily indexed for a Retrieval Augmented Genration Model. 
    The response output should be in the form of paragraph(s).
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


p = """
Activities
A9 Thinktiv will prepare and conduct 10 – 15 interviews with a combination of Client’s end-
customers, representative commercial buyers, and/or customers of competing products (as
possible), focusing on identifying pain points, fundamental requirements, and perceived
value of key Client product capabilities.
"""
# get_paragraph_description(p)
