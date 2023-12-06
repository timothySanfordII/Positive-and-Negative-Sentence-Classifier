import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch import optim

# Data Preparation
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
sentences = ["The book was great I loved it","The book was bad, I hated it"]
labels = torch.tensor([1, 0])  # 1 for good, 0 for bad

# Tokenize and convert to tensors
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

dataset = TensorDataset(input_ids, attention_mask, labels)

# Model Architecture and Fine-tuning
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
optimizer = optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()

# Split Dataset into Training and Validation Sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Training
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

for epoch in range(9):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = loss_fn(outputs.logits, labels)
        total_loss += loss.item()
        loss.backward()

        # Update parameters after processing the entire training loader
        optimizer.step()
        optimizer.zero_grad()

# Save the fine-tuned model
model.save_pretrained("fine_tuned_bert")


# Inference
model.eval()
input_sentence = "I hated the book"  # User input
input_ids = tokenizer(input_sentence, return_tensors="pt")["input_ids"]
input_ids = input_ids.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

with torch.no_grad():
    output = model(input_ids, return_dict=True)
prediction = torch.argmax(output.logits, dim=1).item()

if prediction == 1:
    print("Good")
else:
    print("Bad")

