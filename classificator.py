import spacy
import classy_classification
from spacy import displacy

import json

f = open("data.json")
data = json.load(f)


nlp = spacy.load("hr_core_news_lg")
# Install package with !python3 -m spacy download hr_core_news_lg

nlp.add_pipe("text_categorizer",
             config={
                 "data": data,
                 "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
             }
             )


def topicClassifier(inquiry):
    results = (nlp(inquiry)._.cats)
    processed_data = [{"topic": key, "similarity": round(value, 2)}
                      for key, value in results.items()]
    return processed_data
