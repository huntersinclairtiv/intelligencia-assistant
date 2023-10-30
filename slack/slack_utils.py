import random
import math
import os
import uuid
import logging
import langchain
import streamlit as st
from typing import List, Dict, Any
from dotenv import load_dotenv
from consts import thinking_thoughts
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client
from supabase.lib.client_options import ClientOptions
from langchain.docstore.document import Document
from ai.ai_chains import storeVectorsInDB, storeSummaryInDB
from sklearn.cluster import KMeans
import numpy as np
import numpy.linalg as npla

langchain.debug = True
langchain.verbose = True

# requires importing logging
logging.basicConfig(level=logging.INFO)

# Load .env variables
load_dotenv()

# LLM Initialization
openai_api_key = os.getenv("OPENAI_API_KEY")
model_name = os.environ.get("MODEL_NAME")

embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key, chunk_size=3500)

simple_template = """Summarize the conversation chat messages in the CONTENT section below. You should follow the following rules when generating the summary: \n
- Attempt to preserve important questions and answers related to business context, companies, or projects as best you can. \n
- The summary should not exceed 800 characters or 3 paragraphs. \n
- Return the summary without any explainations or introductions. \n
- Maintain any links or channel references with the exact syntax or markdown that you are provided. \n
- Prioritize the context of questions and answers over the participants in the conversation. \n
- REMEMBER: Go directly into summarizing the content of the conversation and DO NOT start with generalizations like 'This is a conversation between ...'.\n\n 

*******************\n
CONTENT:\n {document}\n
*******************\n\n

Final answer:"""

template = """You are an assistant to help summarize long conversations between multiple employees at Thinktiv, a business consulting company, that happen in a Slack channel. 
Summarize the conversation chat messages in the CONTENT section below. You should follow the following rules when generating the summary:
- Attempt to preserve important questions and answers related to business context, companies, or projects as best you can.
- The summary should be as concise as possible without losing important details
- The summary should not exceed 800 characters or 3 paragraphs.
- Do not repeat back information I have instructed you. For instance do not say "This is a conversation between employees at Thinktiv" or similar.  You can summarize what specific individuals say though.  
- Do not include any introduction summary about the Slack channel or channel id that the summary is about.  If the Slack channel is related to a particular company / client then you can assume that the majority of the discussion is in relation to a project for that company if not otherwise explicitly stated in the conversation.
- Do not summarize the metadata provided in the content below; only use the metadata as context for the related conversation messages. 
- Maintain any links or channel refences with the exact syntax / markdown that you find.
- Prioritize the context of the questions and answers over the participants in the conversation.
- REMEMBER: Go directly into summarizing the content of the conversation and DO NOT start with generalizations like "This is a conversation between employees at Thinktiv in a Slack Channel".  I already know the slack channel, the name of the related company or topic and the company everyone works for. Assume I know everything I have told you already.

CONTENT: {document}

Final answer:"""

sum_template = """You are an assistant to help organize and refine the summary of long conversations between multiple employees at Thinktiv, a business consulting company, that happen in a Slack channel. 
Organize and refine the text in the CONTENT section below. 
The CONTENT provided below contains a list of sub-summaries of the various conversation threads. 
Your job is to assess the entire context and refine it to remove unnessary language while maintaining ALL of the important context provided.  
IF there is any duplicate or overlapping information you should feel free to make the summary of that information more concise without duplication. 
You should follow the following rules when generating the final summary:
- Attempt to preserve important questions, answers and details related to business context, companies, or projects as best you can.
- Do not repeat back information I have instructed you. For instance do not say "This is a conversation between employees at Thinktiv" or similar. 
- Maintain any links or channel refences with the exact syntax / markdown that you find.
- Each paragraph provided may have a reference link in the following FORMAT with URL representing the full URL. If you find a link like this format it should be kept fully intact with that part of the summary. FORMAT: <URL|🔗ref> 
- Please provide logical paragraph line breaks to make the summary well structured and easily readable.
- The final summary should not exceed 6000 character, 2000 tokens, or 4 paragraphs, whichever is greater.

CONTENT: {document}

Final answer:"""

ques_template = """You are an AI to help write questions based on conversations between employees at my company.  
The context below will typically involve conversations about client companies or projects.  
This context will get stored as embeddings in a vector DB and will be used to help answer questions by employees.
Based on the following context I would like you to write up to three questions which can be answered by this context.  
You should phrase the questions as if being asked by someone unfamiliar with the context and should make them as concise as possible.  
If you determine that the context can reasonably answer any of the 'Priority Questions' listed below, then you should prioritize generating at least one question which addresses the subject of that priority question.  
Otherwise, if the context can not answer any priority questions, you should simply use the context to return the most appropriate questions which the context CAN answer. 
Lastly, your final answer should only include the questions you generate and you must not add additional thoughts or explanation.
Remember: You are generating up to three questions, but do not have to return three if there is not enough context to support three.\n\n

For the purposes of the context provided below you may assume the following:\n
{metadata}
Our program types are: \n
 - GRO, which means Growth, Readiness, and Opportunities program. \n
 - BVA, which means Business Value Architecture program. \n
 - Accelerator, which means a product accelerator program. \n
 - Brand, which means a branding program. \n\n

*************** \n
Priority Questions: \n
1. What are the main challenges, issues, or risks experienced on this program? \n
2. What were the learnings from VOC, voice of customer, interviews? \n
3. What is the current messaging hierarchy, top-line message, or supporting pillars? \n
4. Did we perform a GRO, BVA, Accelerator or Brand program for the client? \n
5. What framework was used or considered in the discussion? \n
6. What insight was supported or nullified by provided links or context? \n
7. What other programs or companies is a good comp for this? \n
8. Is there a link to a LinkedIn account, or discussion around a certain program individual who is important to the conversation?"\n\n

*************** \n
Context: {context} \n\n

Final answer:"""


SUMMARY_PROMPT = PromptTemplate(
    input_variables=["document"], template=template
)
SIMPLE_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["document"], template=simple_template
)
SUM_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["document"], template=sum_template
)
QUES_PROMPT = PromptTemplate(
    input_variables=["metadata", "context"], template=ques_template
)

llm=ChatOpenAI(temperature=0.5, openai_api_key=openai_api_key, model=model_name) # type: ignore
llm_summary_chain = LLMChain(
    llm=llm,
    prompt=SUMMARY_PROMPT
)
llm_simple_summary_chain = LLMChain(
    llm=llm,
    prompt=SIMPLE_SUMMARY_PROMPT
)
llm_sum_summary_chain = LLMChain(
    llm=llm,
    prompt=SUM_SUMMARY_PROMPT
)
llm_ques_chain = LLMChain(
    llm=llm,
    prompt=QUES_PROMPT
)

def is_dm(message) -> bool:
    # Check if the message is a DM by looking at the channel ID
    if message['channel'].startswith('D'):
        return True
    return False


def get_random_thinking_message():
    return random.choice(thinking_thoughts)


def send_slack_message_and_return_message_id(app, channel, message: str):
    response = app.client.chat_postMessage(
        channel=channel,
        text=message)
    if response["ok"]:
        message_id = response["message"]["ts"]
        return message_id
    else:
        return ("Failed to send message.")

def chunkSubstrByChar(content: List[str], size: int):
    joined_content = '==============================\n\n'.join(content)
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['==============================\n\n','\n\n','\n',' '],
        chunk_size = size,
        chunk_overlap  = int(size/30),
        length_function = len,
        is_separator_regex = False,
    )

    texts = text_splitter.create_documents([joined_content])
    return texts

def chunkSubstr(content: List[str], size: int):
    joined_content = '==============================\n\n'.join(content)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        separators=['==============================\n\n','\n\n','\n',' '],
        chunk_size = size,
        chunk_overlap  = int(size/30),
        encoding_name='cl100k_base',
    )
    #length_function = len,
    #is_separator_regex = False,
    texts = text_splitter.create_documents([joined_content])
    return texts

    # numChunks = math.ceil(len(content) / size)
    # chunks = []
    # o = 0
    # for i in range(numChunks):
    #     chunks[i] = content[o:size]
    #     o += size
    # return chunks

def chunkStr(content: str, size: int):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size = size,
        chunk_overlap  = int(size/30),
        encoding_name='cl100k_base',
    )
    texts = text_splitter.create_documents([content])
    return texts

def summarize(document, inquiry):
    #removing all the extra meta context added for summarization purposes.
    doc_parts = document.page_content.split("Conversation chat messages below:\n")
    document.page_content = doc_parts[len(doc_parts) - 1]
    result = llm_simple_summary_chain(document)
    #result = llm_summary_chain(document)
    return result["text"]

def summarize_summary(document):
    result = llm_sum_summary_chain(document)
    return result["text"]

def getQuestionsForChunk(metadata, context):
    result = llm_ques_chain({'metadata': metadata, 'context': context})
    return result["text"]

#TODO: create a command to store embeddings for a channel - I think we should probably have a table that tracks 
# the last timestamp for embedding for channel history and run catchup for it on some interval or trigger???
# We probably also want to store the actual summaries so we can append to them and just return them when asked vs regenerating each time.
# Likely should update the summary method to do like the article discusses - find the 20 or so most central conversations to summarize vs all content

def storeEmbeddings(documents: List[str], links: List[str], last_ts: float, channel: str, channel_name: str, reference_name: str):
    #Chunk document into 3500 token chunks - leaving room for prompts and memory
    chunks = chunkSubstr(documents, 2500)
    logging.info('CHUNKS : %s', str(len(chunks)))

    convo_sep = '==============================\n\n'
    link_index = 0
    for chunk in chunks:
        chunk.metadata["source"] = "Slack"
        chunk.metadata["source_type"] = "Conversation"
        chunk.metadata["channel"] = channel
        chunk.metadata["channel_name"] = channel_name
        chunk.metadata["reference_name_cap"] = reference_name.capitalize()
        chunk.metadata["reference_name_lcase"] = reference_name.lower()
        chunk.metadata["reference_name_ucase"] = reference_name.upper()
        chunk.metadata["reference_name"] = reference_name
        convos = chunk.page_content.split(convo_sep)
        message_url = ""
        if (len(links) > link_index):
            message_url = links[link_index]
        chunk.metadata["source_url"] = message_url
        append_str = f"Slack Channel Name: {channel_name}\n"
        append_str += f"Slack Channel ID: {channel}\n"
        append_str += f"Related Channel Company, Program, or Topic: {reference_name}\n"
        metadata_str = append_str
        append_str += f"Link to this part of the conversation: {message_url}\n"

        append_str += "Questions answered by conversation below:\n"

        #prepend good questions to a chunk
        ques_content = getQuestionsForChunk(metadata_str, chunk.page_content)
        append_str += ques_content + "\n\n"
        append_str += "Conversation chat messages below:\n"
        append_str =  "Slack Channel metadata below:\n" + append_str
        chunk.page_content = append_str + chunk.page_content
        link_index += (len(convos) - 1) #note: need to split each chunk by the convo separator to keep correct index for links

    id_list: List[str] = []
    vector_list: List[List[float]] = []
    batch_size = 200 #maybe increase this - default is 500 for JS
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        logging.info('BATCH SIZE : %s', str(len(batch)))
        batch_texts = [doc.page_content for doc in batch]
        # replacing add docs with add vectors instead so we can hold on to vectors in memory to cluster
        #result = vectordb.add_documents(batch)  # type: ignore
        vectors = embeddings.embed_documents(batch_texts, chunk_size=3500) #increase chunksize to account for prepended meta/ques
        logging.info('VECTORS SIZE : %s', str(len(batch)))
        vector_list.extend(vectors)
        ids = [str(uuid.uuid4()) for _ in batch_texts]

        result = storeVectorsInDB(vectors, batch, ids, embeddings)
        logging.info('DB ADD RESULT : %s', str(result))

        if len(result) == 0:
            raise Exception("Error inserting: No rows added")

        id_list.extend(result)
    logging.info('id_list: %s', id_list)


    numClusters = 20 # Play around with this number
    numberOfChunks = len(chunks)
    if (numberOfChunks <= 10) : 
        numClusters = numberOfChunks
    else : 
        numClusters = 10
   # elif (numberOfChunks <= 60) : 
    #     numClusters = numberOfChunks // 5 
    # elif (numberOfChunks <= 120) : 
    #     numClusters = numberOfChunks // 8 
 
    if (numClusters == 0) :
        numClusters = 1
    
    logging.info('STARTING CLUSTERING FOR : %s', str(len(chunks)))
    clusters = KMeans(n_init = 100, n_clusters = numClusters).fit(vector_list)
    clusterCenters = clusters.cluster_centers_
    logging.info('CLUSTERS : %s', str(len(clusterCenters)))

    closestPoints = []
    for center in clusterCenters:
        # List of distances between each point and the current cluster center
        distances = npla.norm(vector_list - center, axis = 1)
        
        # Finds index corresponding to closest of those vectors and adds to our list
        closestPoint = np.argmin(distances)
        closestPoints.append(closestPoint)

    logging.info('closestPoints: %s', closestPoints)
    closestPoints.sort()
    selectedDocs = [ chunks[idx] for idx in closestPoints ]

    logging.info('START SUMMARIZING : %s', str(len(selectedDocs)))
    
    summarizedChunks = []
    for thisdoc in selectedDocs:
        summary = summarize(thisdoc, "")
        if ("source_url" in thisdoc.metadata):
            message_url = thisdoc.metadata["source_url"]
            if (message_url != "" and message_url != None):
                summary += f" <{message_url}|🔗ref>"
        summarizedChunks.append(summary)

    full_summary = "\n\n".join(summarizedChunks)
    logging.info('full_summary: %s', full_summary)

    #now take the full summary from all the parts and chunk it so we can summarize it further.
    subChunks = chunkStr(full_summary, 3500)
    sum_summary = ""
    if (len(subChunks) > 1):
        subChunks = chunkStr(full_summary, 2000)
        for subChunk in subChunks:
            sum_summary += subChunk.page_content
            sum_summary = summarize_summary(sum_summary)
    else :
        sum_summary = summarize_summary(full_summary)
    logging.info('sum_summary: %s', sum_summary)

    sum_result = storeSummaryInDB(full_summary, last_ts, channel, channel_name, reference_name)
    sum_result = storeSummaryInDB(sum_summary, last_ts, channel, channel_name, reference_name)
    return sum_summary, full_summary

def summarizeLongDocument(document: str, inquiry: str):
    #Chunk document into 8000 char chunks
    if (len(document) > 7500):
        chunks = chunkSubstrByChar([document], 8000)
        summarizedChunks = []
        for chunk in chunks:
            summary = summarize(chunk, inquiry)
            summarizedChunks.append(summary)

        result = "\n\n".join(summarizedChunks)

        #allow for a bit extra
        if (len(result) > 9000):
            return summarizeLongDocument(result, inquiry)
        else:
            return result
    else:
      return document