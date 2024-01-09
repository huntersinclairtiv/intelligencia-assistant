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


# p = """"
# Mr. Ek, the esteemed leader of Hex Ltd., showcases remarkable leadership qualities in steering the company towards unprecedented success. Mr. Ek's visionary approach has been instrumental in positioning Hex Ltd. as a key player in the industry. The dynamic strategies implemented under Mr. Ek's guidance have propelled the company to new heights, solidifying its reputation for innovation and excellence. The entire team at Hex Ltd. is inspired by Mr. Ek's leadership, contributing to the company's continued growth and success. The collaborative efforts of Mr. Ek and the dedicated team at Hex Ltd. exemplify the commitment to excellence that defines the company's ethos.
# """
# print(extract_sentences(p))
