import torch
import pandas as pd
import json, requests
from decouple import config

from datasets import load_dataset
from sentence_transformers.util import semantic_search

from topic_classifier import getStrongestTopics

#Load data from JSON
f = open("enquiries2.json", 'r', encoding="utf8")
data = json.load(f)

#Sentence Transformers model - an existing pre-trained model for creating the embeddings.
model_id = "sentence-transformers/all-MiniLM-L6-v2"
hf_token = str(config("TOKEN"))
#Feature extraction pipeline
api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}

#Return model-processed data (vector embeddings) from inputs (questions)
def generateEmbeddings(texts):
    response = requests.post(api_url, headers=headers, json={
                             "inputs": texts, "options": {"wait_for_model": True}})
    return response.json()

#Extract questions from question: answer data pairs
questions_data = [pair["question"] for pair in data]
print("questions_data>>>>>>>>>>", questions_data)
#Generate embeddings and export them to csv file
model_output = generateEmbeddings(questions_data)
embeddings = pd.DataFrame(model_output)
embeddings.to_csv("embeddings/enquiries.csv", index=False)


def getDatasetEmbeddings():
    data_embeddings = load_dataset('lukablaskovic/student-enquiries-cro')
    dataset_embeddings = torch.from_numpy(data_embeddings["train"].to_pandas().to_numpy()).to(torch.float)

    return dataset_embeddings


def search(inquiry):

    #Create an embedding for inquiry
    inquiry_output = generateEmbeddings(inquiry)
    inquiry_embeddings = torch.FloatTensor(inquiry_output)
    
    dataset_embeddings = getDatasetEmbeddings();
    
    # Find top hit
    hits = semantic_search(inquiry_embeddings, dataset_embeddings, top_k=3)
    print(hits)
    # #Extract
    questions = ([questions_data[hits[0][i]['corpus_id']] for i in range(len(hits[0]))])
    print(questions)
    return questions
