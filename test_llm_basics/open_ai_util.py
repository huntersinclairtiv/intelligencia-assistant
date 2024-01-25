# THIS SHOULD BE IMPLEMENTED USING BUILT-IN CHAINS
# MAKE USE OF SYSTEM MESSAGES FOR BETTER RESULTS

from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
import base64
import json

from langchain.docstore.document import Document

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
    # TODO Update the prompt to work with header and narrative text as well
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
        The generated response should be a valid json object without any explicit mention of json and with following format:
        ["ques1", "ques2", .... ]
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
    print(response.choices[0].message.content)
    metadata['doc_id'] = metadata.get('id')
    try:
        docs.extend([Document(page_content=question, metadata=metadata)
                     for question in json.loads(response.choices[0].message.content)])
    except Exception as e:
        print("ERROR CAUSED--->", e)

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
    The generated response should be a valid json object without any explicit mention of json and with following format:
    ["ques1", "ques2", .... ]
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
    metadata['doc_id'] = metadata.get('id')
    try:
        docs.extend([Document(page_content=question, metadata=metadata)
                     for question in json.loads(response.choices[0].message.content)])
    except Exception as e:
        print("ERROR OCCURED--> ", e)
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


def query():
    PROMPT = """
    Using the given **context**, answer the following **query**.
    context:
    I want to index my RDS database to implement a Retrieveal Augmeneted Generator Model.
    While exploring, I came across a few tools such as pgvector to store embeddings in the same RDS database.
    However I am confused on what will be the data/(exact sentence or keywords) 
    that I'll create embeddings from as embedding an entire row might cause into loss of relations If foreign keys are present.
    **query**:
    1). What data should I create embeddings from ?
    2). What can be other strategies to solve the entire issue of indexing RDS database of RAG? I am open to any sort of solution that would work.
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
    response = get_openai_respone(messages, model, 1000)
    print(response)


def get_ques_list_for_rds_table(command):
    SYSTEM_PROMPT = """
    Given the schema for an SQL table, generate an extensive list of questions that can be answered using this table. Ensure the questions are data-driven and not specific to the table's structure. 
    The questions must cover various aspects of the table. Categorize the questions based on the columns that answer those questions
    The generated response should be a valid json object without any explicit mention of json and with following format:
    {
    "column_1": ["ques1", "ques2", ...],
    "column_3": ["ques4", "ques3", ...],
    .....
    }
    """
    model = 'gpt-4-1106-preview'
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": command
        }
    ]
    response = get_openai_respone(messages, model)
    print(response.choices[0].message.content)
    return json.loads(response.choices[0].message.content)


def nl_to_Sql(table_schema, query):
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
              "role": "system",
              "content": f"Given the following SQL tables, your job is to write SQL queries given a user's request. {table_schema}. \n Return only the sql query and nothing else."
            },
            {
                "role": "user",
                "content": query
            }
        ],
        temperature=0,
        max_tokens=300,
        top_p=1
    )
    return response.choices[0].message.content


def ppt_slide_summary_generator(ppt_title, slide_text, metadata={}):
    SYSTEM_PROMPT = """
    Given the the **title** of the entire presentation and **extracted text** from one of the slides from the same presentation, generate extensive summary for the slide and the insights that the slide holds.
    The response should be concsie and upto the point.
    The response must not contain any explicit mention of any summary or insights.
    The response should not contain any bullet points or numbering for insights or any other aspect of response.
    """
    USER_PROMPT = f"""
    title: {ppt_title}
    extracted text: {slide_text}
    """
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
              "role": "system",
              "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": USER_PROMPT
            }
        ],
        temperature=0,
        max_tokens=300,
        top_p=1
    )
    print("SUMMARY-->", response.choices[0].message.content)
    return custom_parser.parse_paragraph(response.choices[0].message.content, metadata)


def ppt_slide_question_generator(ppt_title, slide_text, metadata={}):
    SYSTEM_PROMPT = """
    Given the the **title** of the entire presentation and **extracted text** from one of the slides from the same presentation, generate list of upto 6 questions that can be answered using this slide.
    The generated response should be a valid json object without any explicit mention of json and with following format:
    ["ques1", "ques2", .... ]
    """
    USER_PROMPT = f"""
    title: {ppt_title}
    extracted text: {slide_text}
    """
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
              "role": "system",
              "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": USER_PROMPT
            }
        ],
        temperature=0,
        max_tokens=300,
        top_p=1
    )
    print("QUESTION-->", response.choices[0].message.content)
    metadata['doc_id'] = metadata.get('id')
    docs = []
    try:
        docs = [Document(page_content=question, metadata=metadata)
                for question in json.loads(response.choices[0].message.content)]
    except Exception as e:
        print("ERROR CAUSED--->", e)
    return docs


def ppt_slide_parser(ppt_title, slide_text, generate_summary=False, metadata={}):
    docs = []
    if generate_summary:
        docs.extend(ppt_slide_summary_generator(
            ppt_title=ppt_title, slide_text=slide_text, metadata=metadata))
    docs.extend(ppt_slide_question_generator(
        ppt_title=ppt_title, slide_text=slide_text, metadata=metadata))
    return docs

