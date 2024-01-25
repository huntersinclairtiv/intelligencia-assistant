import sys

from unstructured.file_utils.filetype import detect_filetype, FileType

import pdf_loader
import ppt_loader

FILE_LOADER_MAPPING = {
    FileType.PDF: pdf_loader,
    FileType.PPTX: ppt_loader,
    # ADD MORE MAPPINGS HERE
}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python loader.py file_path1 file_path2 ...")
        exit(1)
    file_paths = sys.argv[1:]
    failed_files = []
    for file_path in file_paths:
        filetype = detect_filetype(file_path)
        _loader_class = None
        try:
            _loader_class = FILE_LOADER_MAPPING.get(filetype)
            _loader_class.process(file_path)
        except Exception as e:
            failed_files.append(file_path)
            print(f'Encountered \n {e} for file \n {file_path}')
    if failed_files:
        print("FOLLOWING FILES COULD NOT BE PARSED CORRECTLY", failed_files)

