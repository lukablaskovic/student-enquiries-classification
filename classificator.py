import spacy
import classy_classification
from spacy import displacy

nlp = spacy.load("hr_core_news_lg")
# Install package with !python3 -m spacy download hr_core_news_lg


sample_inquiry = """Pozdrav, zanima me koliko će me koštati upis više godine s obzirom da nisam položio dva kolegija od 12 ECTS bodova.
Dodatno, do kada moram uplatiti navedeni iznos?"""

data = {
    "referada": [
        "Pozdrav, kako da se upišem na fakultet",
        "Poštovani, koliko će me koštati upis više godine",
        "Poštovani, kako se ispištem s fakulteta",
        "Dobar dan, koliko moram platiti prebacivanje predmeta od 6 ECTS bodova",
        "Poštovanje, kako se upišem na višu godinu fakulteta?"
    ],
    "Studentski centar": [
        "Pozdrav, gdje mogu naći ponudu studentskih poslova",
        "Dobar dan, gdje se nalazi studentski ugovor",
        "Poštovani, kako da potpišem studentski ugovor. Gdje ga mogu naći?",
        "Pozdrav, htio bih raditi preko ljeta preko studentskog ugovora."
    ],
    "Knjižnica": [
        "Pozdrav, imate li knjigu osnove programiranja za prvu godinu informatike",
        "Dobar dan, mogu li dobiti potvrdu da sam vratio sve knjige",
        "Pozdrav, molio bih ponudu da nemam dugovanja prema knjižnici",
        "Poštovanje, imate li kakvu dostupnu literaturu za strukture podataka i algoritme",
        "Poštovani, imate li u knjižnici knjige vezane za računalne mreže"
    ],
}
nlp.add_pipe("text_categorizer",
             config={
                 "data": data,
                 "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
             }
             )

print(nlp("Bok, kako da dobim studentski ugovor za rad po ljeti?")._.cats)


classes = ["topic", "department"]
departments = ["Referada", "Studentski centar", "Rektorat", "Knjižnica"]
topics = ["Upis više godine", "Uplata", "Rok",
          "Ispis s fakulteta", "Prebacivanje predmeta"]

doc = nlp(sample_inquiry)


def classifyInquiry():
    pass


def tokenize(doc):
    return [token for token in doc]


def sentenize(doc):
    return [sent for sent in doc.sents]


def removeStopWords(doc):
    return nlp(" ".join([str(t) for t in doc if not t.is_stop]))


async def topicClassifier(inquiry):
    print(inquiry)
    doc = nlp(inquiry)
    doc_cleaned = removeStopWords(doc)
    print(doc)
    return [{"topic": topic, "similarity": round(doc_cleaned.similarity(nlp(topic)), 3)} for topic in topics]

print(topicClassifier(doc))
# print(tokenize(doc))
