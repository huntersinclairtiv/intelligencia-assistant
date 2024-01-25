import os
import sys
import shutil
import string
import uuid
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from tabulate import tabulate
from bs4 import BeautifulSoup
from langchain.docstore.document import Document
from langchain.document_loaders import UnstructuredFileLoader

import open_ai_util
import paragraph_parser
import supabase_docstore as custom_supabase
import chroma_db_util

from dotenv import load_dotenv
load_dotenv()


image_count = 0
image_folder = 'ppt_images'
documents_table = os.environ.get('SUPABASE_DOCUMENTS_TABLE', '')
embeddings_table = os.environ.get('SUPABASE_EMBEDDINGS_TABLE', '')
store = custom_supabase.SupabaseDocstore(documents_table)


def pptx_table_to_html(table):
    """
    Accepts a Table object from the pptx and returns a Html table generated from it
    """
    rows = []
    for row in table.rows:
        cells = [cell.text for cell in row.cells]
        rows.append(cells)

    html_table = tabulate(rows, headers='firstrow', tablefmt='html')
    return html_table


def process_shapes(slide, page_number, skip_text=None):
    """
    Extracts data from the slide/Group Shapes and returns a List of Document(s)
    """
    global image_count
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    parsed_docs = []
    run_text = ""
    for shape in slide.shapes:
        metadata = {
            'category': shape.shape_type,
            'page_number': page_number
        }
        page_content = ''
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
            # TODO:: Handle cases when the image has charts,tables etc
            page_content = ''  # This could be updated if needed
            metadata['image_path'] = f'{image_folder}/img_{image_count}.jpeg'
            # metadata['cordinates'] # TODO:: Add support for narrative text via coridinates if possible
            with open(metadata['image_path'], 'wb') as f:
                f.write(shape.image.blob)
                # TODO:: VERIFY IF PARSING IMAGE VIA UNSTRUCTURED HERE IS MORE REASONABLE
                image_count += 1
            parsed_docs.append(Document(page_content='', metadata=metadata))
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            # this will add the groups as parent doc as well
            group_list = process_shapes(shape, page_number)
            print('GROUP--> ', group_list)
            parsed_docs.extend(group_list)
        if shape.has_table:
            # Concvert Table object to html table
            metadata['text_as_html'] = pptx_table_to_html(shape.table)
            parsed_docs.append(Document(page_content='', metadata=metadata))
            # Now this table will function same as unstructured's table
        if not shape.has_text_frame:
            continue
        for paragraph in shape.text_frame.paragraphs:
            for run in paragraph.runs:
                # is a part of the text run.text
                text = run.text
                if skip_text:
                    text = text.replace(skip_text, "")
                if text:
                    run_text += f' {text.strip()}'
    # This should be done in order to
    parsed_docs.append(Document(page_content=run_text, metadata={
                       'category': 'text', 'page_number': page_number}))
    return parsed_docs


def process_pptx_files(filepath):
    """
    Process a pptx file and returns a List of Documents(s)
    """
    f = open(f'{filepath}', "rb")
    prs = Presentation(f)
    docs = []
    page_number = 1
    last_title = None
    for slide in prs.slides:
        curr_title = slide.shapes.title
        update_title = True
        if curr_title:
            skip_text = curr_title.text
            if last_title and curr_title.text.startswith(last_title.rstrip(string.punctuation + string.whitespace)):
                if 0.7 * len(curr_title.text) < len(last_title):
                    # matches over 70 % from the last title
                    update_title = False
            # append this to list as well
            if update_title:
                last_title = curr_title.text
                docs.append(Document(page_content=last_title, metadata={
                            'page_number': page_number,
                            'category': 'Title'
                            }))
        docs.extend(process_shapes(slide, page_number, skip_text))
        page_number += 1
    return docs


def process_ppt_via_unstructured(file_path):
    """
    Process ppt files via UnstructuredFileLoader
    # CURRENTLY NOT IN USE
    """
    docs = UnstructuredFileLoader(
        file_path=file_path,
        strategy='hi_res',
        mode='elements',
        include_slide_notes=True,  # WILL INCLUDE READER NOTES AS WELL
        include_page_breaks=True,
        skip_infer_table_types=[],
        pdf_infer_table_structure=True,
        pdf_extract_images=True,
        chunking_strategy='by-title'
    ).load()
    parent_doc = None
    last_narrative_text = None
    last_header = None
    child_doc_list = []
    parent_doc_list = []
    doc_uid = str(uuid.uuid4())
    for document in docs:
        page_content = document.page_content
        metadata = document.metadata
        category = metadata['category']
        table_detals = metadata.get('text_as_html')
        metadata['id'] = doc_uid
        if category == 'Title':
            last_header = page_content
        if category in ['Text', 'NarrativeText']:
            last_narrative_text = page_content
        if category == 'PageBreak':
            # create indexing here
            parent_doc = None
            continue
        if table_detals:
            table_uuid = str(uuid.uuid4())
            metadata['id'] = table_uuid
            parent_doc_list.append((
                metadata['id'],
                Document(
                    page_content=table_detals, metadata=metadata)
            ))
            child_doc_list.extend(open_ai_util.get_table_description(
                table=table_detals,
                header=last_header,
                narrative_text=last_narrative_text,
                metadata=metadata
            ))
        parent_doc += page_content


def index_doc_for_parent_child_retriever(docs):
    """
    Accepts a list of Document(s) and creates vectorstore and doc store for it
    """
    parent_doc_list = []
    child_doc_list = []
    ppt_title = None
    larger_chunk = ""
    doc_id = str(uuid.uuid4())
    for doc in docs:
        page_content = doc.page_content
        metadata = doc.metadata
        category = metadata['category']
        metadata['id'] = doc_id
        table = metadata.get('text_as_html')
        image_path = metadata.get('image_path')

        if category == 'Title':
            if larger_chunk:
                parent_doc_list.append((metadata['id'], Document(
                    page_content=larger_chunk, metadata=metadata)))  # hold the metadata for it?
                # Generate the summary  and questions for larger chunk only
                if paragraph_parser.get_word_count(larger_chunk) > 50:
                    generate_summary = paragraph_parser.get_word_count(
                        larger_chunk) > 150
                    child_doc_list.extend(open_ai_util.ppt_slide_parser(
                        metadata=metadata,
                        ppt_title=ppt_title,
                        slide_text=larger_chunk,
                        generate_summary=generate_summary
                    ))
            doc_id = str(uuid.uuid4())  # update the doc id
            metadata['id'] = doc_id
            child_doc_list.append(Document(page_content=page_content, metadata={
                                  'doc_id': doc_id}))  # indexing the current title
            larger_chunk = ""
            ppt_title = page_content
        elif page_content:
            larger_chunk += page_content
            child_doc_list.extend(
                paragraph_parser.parse_paragraph(page_content, metadata))
        elif table:
            metadata['id'] = str(uuid.uuid4())
            parent_doc_list.append((
                metadata['id'],
                Document(page_content=table, metadata=metadata)
            ))
            child_doc_list.extend(open_ai_util.get_table_description(
                table=table, metadata=metadata))
        elif image_path:
            # DO NOT PARSE IMAGES AS OF NOW
            doc = UnstructuredFileLoader(
                file_path=image_path,
                strategy='hi_res',
                mode='elements',
                include_slide_notes=True,  # WILL INCLUDE READER NOTES AS WELL
                include_page_breaks=True,
                skip_infer_table_types=[],
                pdf_infer_table_structure=True,
                pdf_extract_images=True,
                chunking_strategy='by-title'
            ).load()
            for d in doc:
                larger_chunk += d.page_content+" "
            # page_content = open_ai_util.get_image_description(image_path)
            # # CHECK FOR A VALID RESPONSE
            # if page_content not in ['UNEXTRACTABLE_DATA', 'UNPROCESSABLE_ENTITY']:
            #     image_id = str(uuid.uuid4())
            #     # use the same metadata as of the parent image
            #     if not image_path == 'test_0.jpeg':
            #         continue
            #     metadata['id'] = image_id
            #     parent_doc_list.append(
            #         (image_id, Document(page_content=page_content, metadata=metadata)))
            #     child_doc_list.extend(
            #         paragraph_parser.parse_paragraph(page_content, metadata))
            #     # GENERATING QUESTIONS BASED ON
            #     child_doc_list.extend(
            #         open_ai_util.get_paragraph_description(paragraph=page_content, metadata=metadata))
        if len(larger_chunk) > 4000:
            child_doc_list.extend(open_ai_util.ppt_slide_parser(
                metadata=metadata,
                ppt_title=ppt_title,
                slide_text=larger_chunk,
                generate_summary=True
            ))
            parent_doc_list.append((metadata['id'], Document(
                page_content=larger_chunk, metadata=metadata)))
            larger_chunk = ""
            doc_id = str(uuid.uuid4())  # updated the uuid

    if larger_chunk:
        metadata = {'id': doc_id}
        parent_doc_list.append((metadata['id'], Document(
            page_content=larger_chunk, metadata=metadata)))  # hold the metadata for it?
        # Generate the summary  and questions for larger chunk only
        if paragraph_parser.get_word_count(larger_chunk) > 50:
            generate_summary = paragraph_parser.get_word_count(
                larger_chunk) > 150
            child_doc_list.extend(open_ai_util.ppt_slide_parser(
                metadata=metadata,
                ppt_title=ppt_title,
                slide_text=larger_chunk,
                generate_summary=generate_summary
            ))
    with open('child_list_ppt.txt', 'w') as f:
        for d in child_doc_list:
            f.write(f'{d.page_content}\n\n')
    with open('parent_list_ppt.txt', 'w') as f:
        for _, d in parent_doc_list:
            f.write(f'{d.page_content}\n\n')
    chroma_db_util.create_persistent_vector_database(child_doc_list)
    store.mset(parent_doc_list)


def process(file_path):
    docs = process_pptx_files(file_path)
    for doc in docs:
        image_path = doc.metadata.get('image_path')
        if image_path:
            size_in_kb = os.stat(image_path).st_size/1024

            if size_in_kb < 50:
                print(image_path)
                image_data = UnstructuredFileLoader(
                    file_path=image_path,
                    strategy='hi_res',
                    # mode='elements',
                    include_slide_notes=True,  # WILL INCLUDE READER NOTES AS WELL
                    include_page_breaks=True,
                    skip_infer_table_types=[],
                    pdf_infer_table_structure=True,
                    pdf_extract_images=True,
                    chunking_strategy='by-title'
                ).load()[0].page_content
                if paragraph_parser.get_word_count(image_data) < 5:
                    # do not parse it

                    pass
                # ONLY LOAD THIS IMAGE VIA UNSTRUCTURED
            # print(image_path, size_in_kb, doc)
    index_doc_for_parent_child_retriever(docs)
