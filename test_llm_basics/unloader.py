import sys

from chroma_db_util import ChromaDB

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python loader.py file_path1 file_path2 ...")
        exit(1)
    file_paths = sys.argv[1:]
    custom_chroma_db_util = ChromaDB()
    for file_path in file_paths:
        print('Rolling Back Changes for: ', file_path)
        custom_chroma_db_util._clear(file_path)
