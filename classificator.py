import spacy
from spacy import displacy

nlp = spacy.load("hr_core_news_lg")
#Install package with !python3 -m spacy download hr_core_news_lg

sample_inquiry = """Pozdrav, zanima me koliko će me koštati upis više godine s obzirom da nisam položio dva kolegija od 12 ECTS bodova.
Dodatno, do kada moram uplatiti navedeni iznos?"""

classes = ["topic", "department"]
departments = ["Referada", "Studentski centar", "Rektorat", "Knjižnica"]
topics = ["Upis više godine", "Uplata", "Rok", "Ispis"]

doc = nlp(sample_inquiry)

def classifyInquiry():
    pass

def tokenize(doc):
    return [token for token in doc]

def sentenize(doc):
    return [sent for sent in doc.sents]

async def topicClassifier(inquiry):
    doc = nlp(inquiry)
    for topic in topics:
        doc2 = nlp(topic)
        print("Inquiry : ", topic, "-->", doc.similarity(doc2))
    return [{"topic" : topic, "similarity" : round(doc.similarity(nlp(topic)), 3)} for topic in topics]

#print(topicClassifier(doc))
#print(tokenize(doc))