import spacy
import classy_classification as cc
import json

#Load data from JSON
f = open("enquiries.json", 'r', encoding="utf8")
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
#Categorizes text in predefined classes
#Example
#Input: "Pozdrav, kako da upišem drugu godinu informatike? Dodatno, koja je cijena upisnine ako nisam prošao 1 predmet od 6 ECTS bodova?"
"""
Output: [{'topic': 'ispitni-rok', 'similarity': 0.12}, 
{'topic': 'opcenito', 'similarity': 0.12}, 
{'topic': 'placanje', 'similarity': 0.26}, 
{'topic': 'upis-vise-godine', 'similarity': 0.46}, 
{'topic': 'zavrsni-rad', 'similarity': 0.04}]
"""

def classifier(inquiry):
    results = (nlp(inquiry)._.cats)
    processed_data = [{"topic": key, "similarity": round(value, 2)}
                      for key, value in results.items()]
    return processed_data
#processed_data = classifier("Pozdrav, kako da upišem drugu godinu informatike? Dodatno, koja je cijena upisnine ako nisam prošao 1 predmet od 6 ECTS bodova?")

