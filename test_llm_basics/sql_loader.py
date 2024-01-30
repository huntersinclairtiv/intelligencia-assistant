import os

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.documents import Document

from chroma_db_util import ChromaDB
import open_ai_util

from dotenv import load_dotenv

load_dotenv()
supabase_conn_uri = os.environ.get('SUPABASE_CONN_URI')
db = SQLDatabase.from_uri(supabase_conn_uri)

table_list = ['mappingharvest', 'fullcompanylist', 'hubspotcontactcompanies',
              'mainprograms', 'hubspotcompanydetails', 'programtracker', 'hubspotcontacts']


def document_list_from_text(text_dict, metadata={}):
    docs = []
    for coulmn_name, ques_list in text_dict.items():
        metadata['column_name'] = coulmn_name
        for ques in ques_list:
            docs.append(Document(page_content=ques, metadata=metadata))
    return docs


def get_primary_and_foreign_keys(create_commad, relevant_columns={}, table_name=''):
    if not table_name:
        table_name = create_commad.split()[2]

    fetch_column_query = f"""
    SELECT column_name
    FROM information_schema.columns
    WHERE table_name = '{table_name}' AND table_schema = 'public';
    """
    column_name_list = {column_details['column_name']
                        for column_details in db._execute(fetch_column_query)}
    create_commad.upper()
    prefix = f'\nCREATE TABLE {table_name} ('
    suffix = ');'
    create_commad = create_commad.strip(prefix).rstrip(
        suffix).replace('\n', '').replace('\t', '')
    column_list = create_commad.split(',')

    # ADD FK AND PK TO relevant columns
    for column_name in column_name_list.difference(relevant_columns):
        for column_sql in column_list:
            is_pk_or_fk = "PRIMARY KEY" in column_sql or "FOREIGN KEY" in column_sql
            if column_name in column_sql and is_pk_or_fk:
                relevant_columns.add(column_name)

    relevant_sql_statement = set()
    # ADD RELEVANT SQL STATEMENTS FOR CREATE COMMAND
    for column_name in relevant_columns:
        for column_sql in column_list:
            if column_name in column_sql:
                relevant_sql_statement.add(column_sql.strip())

    updated_create_command = f'CREATE TABLE {table_name} ('
    # UPDATE THE CREATE COMMAND
    for sql_statement in relevant_sql_statement:
        updated_create_command += f'\n\t{sql_statement},'
    updated_create_command = updated_create_command.rstrip(',')
    updated_create_command += f'\n\t);'
    return (updated_create_command, relevant_columns)


def format_sample_rows(sample_rows, base_string):
    columns = sample_rows[0].keys()
    formatted_string = f'{base_string} \n'

    header_row = " ".join(columns)
    formatted_string += f"{header_row}\n"

    for row in sample_rows:
        values_row = " ".join(str(row[key]) for key in columns)
        formatted_string += f"{values_row}\n"

    formatted_string += "*/"
    return formatted_string


def get_create_command(table_name, relevant_columns={}, include_sample_rows=False, top_k=3):
    create_command = db.get_table_info_no_throw([table_name]).split('\n\n')[0]
    if relevant_columns:
        create_command, relevant_columns = get_primary_and_foreign_keys(
            create_command, relevant_columns, table_name)
        if include_sample_rows:
            query_to_select_top_k = f"""SELECT {', '.join(column_name for column_name in relevant_columns)} from {table_name} LIMIT {top_k}"""
            sample_rows = db._execute(query_to_select_top_k)
            base_string = f'\n/* Sample {len(sample_rows)} rows from the table {table_name}:'
            create_command += format_sample_rows(sample_rows, base_string)
    return create_command


def create_database_documents(table_name_list):
    document_list = []
    for table_name in table_name_list:
        metadata = {'table_name': table_name}  # TODO:: improvise this
        create_command = get_create_command(table_name)
        column_question_map = open_ai_util.get_ques_list_for_rds_table(
            create_command)
        document_list.extend(document_list_from_text(
            column_question_map, metadata))
    return document_list


def create_vector_store(table_list):
    docs = create_database_documents(table_list)
    print('DOCUMENTS LOADED')
    vectorstore = ChromaDB().create_persistent_vector_database(docs)
    print('VECTORSTORE CREATED')
    return vectorstore


def answer_queries(query):
    vectorstore = ChromaDB().get_persistent_vector_database()
    retrieved_docs = vectorstore.as_retriever().get_relevant_documents(query)
    retrieved_tables = {
        retrieved_doc.metadata['table_name'] for retrieved_doc in retrieved_docs}
    print('FINAL TABLES--> ', retrieved_tables)
    schema_info = ""
    relevant_column_kwargs = {
        'score_threshold': 0.8,  # TODO:: Figure out the optimal threshold value
        'k': 10,
    }
    for table_name in retrieved_tables:
        relevant_column_kwargs['filter'] = {'table_name': table_name}
        relevant_columns = {
            relevant_col.metadata['column_name']
            for relevant_col, similarity_score in
            vectorstore.similarity_search_with_relevance_scores(
                query=query, ** relevant_column_kwargs)
        }
        create_command = get_create_command(table_name, relevant_columns, True)
        schema_info += f'\n{create_command}'
    print('FINAL SCHEMA--> ', schema_info)
    sql_command = open_ai_util.nl_to_Sql(schema_info, query)
    print('SQL-->', sql_command)
    # print('SQL_RESULT---> ', db._execute(sql_command))


create_vector_store(table_list)  # COMMENT THIS AFTER FIRST GO
answer_queries(
    "What is the JobTitle for the Company Contact of the project 'HRSoft BVA'")
