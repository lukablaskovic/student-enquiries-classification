from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sentence_transformers import InputExample

from sentence_transformers import losses
from torch.utils.data import DataLoader

model_id = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_id)

dataset_id = "lukablaskovic/student-enquiries-cro_train"
dataset = load_dataset(dataset_id)

print(f"- The {dataset_id} dataset has {dataset['train'].num_rows} examples.")
print(f"- Each example is a {type(dataset['train'][0])} with a {type(dataset['train'][0]['set'])} as value.")
print(f"- Examples look like this: {dataset['train'][0]}")


train_examples = []
train_data = dataset['train']['set']
# For agility we only 1/2 of our available data
n_examples = dataset['train'].num_rows // 2

for i in range(n_examples):
  example = train_data[i]
  train_examples.append(InputExample(texts=[example['query'], example['pos'][0], example['neg'][0]]))



train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

#The next step is to choose a suitable loss function that can be used with the data format.
"""
Case 4: If you don't have a label for each sentence in the triplets, you should use TripletLoss.
This loss minimizes the distance between the anchor and the positive sentences while maximizing the distance between the anchor and the negative sentences.
"""
#We will use TripletLoss function

train_loss = losses.TripletLoss(model=model)

#How to train or fine-tune a Sentence Transformer model

num_epochs = 10

model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=num_epochs, output_path="C:/Users/Luka/Documents/GitHub/student-enquiries-classification/model")