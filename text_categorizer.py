import spacy
import classy_classification as cc

import json

f = open("topics-enquiries.json")
data = json.load(f)

# Sentence Transformers model - MiniLM - L12-v2
# https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
# https://www.sbert.net/
nlp = spacy.blank("hr")
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

# Sample sentence for testing
#sen = "Pozdrav, kako da upišem drugu godinu informatike? Dodatno, koja je cijena upisnine ako nisam prošao 1 predmet od 6 ECTS bodova?"


def topicClassifier(inquiry):
    # print("Full inquiry:", inquiry)
    sentences = sentencizer(inquiry)

    final_data = []
    for sen in sentences.sents:
        doc = nlp(sen.text)
        final_data.append({"sentence": doc.text, "cats": doc._.cats})

    processed_data = [{"sentence": data["sentence"], "categories": [{"topic": top, "similarity": round(value, 2)} for top, value in data["cats"].items()]}
                      for data in final_data]
    print(processed_data)
    return processed_data
