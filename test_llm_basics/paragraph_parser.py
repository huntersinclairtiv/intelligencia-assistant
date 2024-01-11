import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from langchain_core.documents import Document


def extract_sentences(paragraph):
    # TODO: Remove this and setup as a package.
    # Can comment this out once this has been downloaded.
    # nltk.download('punkt')
    sentences = sent_tokenize(paragraph)
    return sentences


def parse_paragraph(paragraph, metadata={}, last_title=None):
    sentences = extract_sentences(paragraph)
    doc_list = []
    for i, sentence in enumerate(sentences, start=1):
        metadata['doc_id'] = metadata.get('id')
        if last_title:
            sentence = last_title + sentence
        doc_list.append(Document(page_content=sentence, metadata=metadata))
    return doc_list


def get_word_count(paragraph):
    return len(word_tokenize(paragraph))
