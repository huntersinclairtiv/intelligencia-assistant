import nltk
from nltk.tokenize import sent_tokenize
from langchain_core.documents import Document


def extract_sentences(paragraph):
    # nltk.download('punkt') required once
    sentences = sent_tokenize(paragraph)
    return sentences


def parse_paragraph(paragraph, metadata={}, last_title=None):
    sentences = extract_sentences(paragraph)
    doc_list = []
    for i, sentence in enumerate(sentences, start=1):
        metadata['doc_id'] = metadata.get('id')
        if last_title:
            sentence = last_title+sentence
        doc_list.append(Document(page_content=sentence, metadata=metadata))
    return doc_list
