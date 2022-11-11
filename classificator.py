import spacy
from spacy import displacy

nlp = spacy.load("hr_core_news_lg")
#Install package with !python3 -m spacy download hr_core_news_lg

sample_inquiry = """Pozdrav, zanima me koliko će me koštati upis više godine s obzirom da nisam položio dva kolegija od 12 ECTS bodova.
Dodatno, do kada moram uplatiti navedeni iznos?"""

doc = nlp(sample_inquiry)

def tokenize(doc):
    return [token for token in doc]

print(tokenize(doc))



