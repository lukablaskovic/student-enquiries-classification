import spacy
import classy_classification as cc
import json

#Load data from JSON
f = open("qa-data.json", 'r', encoding="utf8")
data = json.load(f)

nlp = spacy.blank("hr")
# Pipeline: Spacy TextCategorizer 
# Model: Sentence Transformers model - MiniLM - L12-v2
# Library: Classy Classsification
# https://www.sbert.net/
nlp.add_pipe("text_categorizer",
             config={
                 "data": data,
                 "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 "device": "gpu",
             }
             )

def sentencizer(inquiry):
    sentence_model = spacy.blank("hr")
    sentence_model.add_pipe("sentencizer")
    sentences = sentence_model(inquiry)
    return sentences

async def getPredefinedAnswer(inquiry):
    
    sentences = sentencizer(inquiry)
    final_data = []
    for sen in sentences.sents:
        doc = nlp(sen.text)
        processed_data = [{"question": sen.text, "answer": key, "similarity": round(value, 2)}
                      for key, value in doc._.cats.items()]
        strongest_topics = max(processed_data, key = lambda x: x["similarity"])
        final_data.append(strongest_topics)
    return final_data
