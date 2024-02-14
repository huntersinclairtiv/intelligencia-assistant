## The following Readme Explains on How to setup and provides an overview of the flow


### SETUP
- Install the requirements using:
```
pip install -r requirements.txt; pip install sqlalchemy --upgrade;
```
- Create a .env file using the following command and fill out the values for the same:
```shell
echo -e "SUPABASE_URL=\nSUPABASE_KEY=\nSUPABASE_DOCUMENTS_TABLE=\nSUPABASE_EMBEDDINGS_TABLE=\nSUPABASE_QUERY_NAME=\nOPENAI_API_KEY=\n" > .env
```
- Schema for SUPABASE_EMBEDDINGS_TABLE will have 4 columns of types:
```
{
    "id": uuid,
    "content": text,
    "metadata": jsonb,
    "embedding": vector
}
```
- Schema for SUPABASE_DOCUMENTS_TABLE will have 3 columns of types:
```
{
    "id": uuid,
    "content": text,
    "metadata": jsonb
}
```

### FLOW OVERVIEW
- The file [loader.py](/test_llm_basics/loader.py) is responsible for parsing the documents and indexing.
- To parse a file using the loader.py file, run the following command
- ```
  python loader.py <file_path_1> <file_path_2> ......
  ```
- The file [retriever.py](/test_llm_basics/retriever.py) is responsible for answering the queries from the parsed data.
- Modify the **query_list** array from the retriever.py file and run the following command:
- ```
  python retriever.py
  ```
- The response to each of the query will be written under a seperate file inside of the 'outputs' dir.

- Note: The above flow can be used for .pptx and .pdf files as of now. 

### USING CSV FLOW
The entire flow for CSV parsing and retrival is inside of csv_loader.py[] file.
Currently we make use of the infamous Titanic Datatset. 
To See its working run 
- ```
  cd test_llm_basics;
  python csv_loader.py
  ```
- The csv retiever is currently in very basic POC state.

### USING SQL FLOW
The flow is under POC as of now and needs improvements. Usage not recommended.
