from sentence_transformers import losses
from torch.utils.data import DataLoader
from sentence_transformers import InputExample
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

model_id = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_id)


dataset_id = "embedding-data/QQP_triplets"
dataset = load_dataset(dataset_id)
# This guide uses an unlabeled triplets dataset, the fourth case above.

print(f"- The {dataset_id} dataset has {dataset['train'].num_rows} examples.")
print(
    f"- Each example is a {type(dataset['train'][0])} with a {type(dataset['train'][0]['set'])} as value.")
print(f"- Examples look like this: {dataset['train'][0]}")


train_examples = []
train_data = dataset['train']['set']
# For agility we only 1/2 of our available data
n_examples = dataset['train'].num_rows // 2

for i in range(n_examples):
    example = train_data[i]
    train_examples.append(InputExample(
        texts=[example['query'], example['pos'][0], example['neg'][0]]))

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

train_loss = losses.TripletLoss(model=model)

model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10)
