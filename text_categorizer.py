import spacy
import classy_classification as cc

import json

f = open("topics-enquiries.json")
data = json.load(f)


nlp = spacy.blank("hr")
# Install package with !python3 -m spacy download hr_core_news_lg

# Sentence Transformers model - MiniLM - L12-v2
# https://www.sbert.net/
nlp.add_pipe("text_categorizer",
             config={
                 "data": data,
                 "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 "device": "gpu",
             }
             )


def topicClassifier(inquiry):
    results = (nlp(inquiry)._.cats)
    processed_data = [{"topic": key, "similarity": round(value, 2)}
                      for key, value in results.items()]
    return processed_data
