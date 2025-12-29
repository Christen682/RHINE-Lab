import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel 
from torch.optim import AdamW 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    print("Step 1: Loading and preparing dataset...")
    dataset = load_dataset("imdb")

   
    train_dataset = dataset['train'].shuffle(seed=42).select(range(10000))
    test_dataset = dataset['test']

    train_texts = train_dataset['text']
    train_labels = train_dataset['label']
    test_texts = test_dataset['text']
    test_labels = test_dataset['label']

   
    unique, counts = np.unique(train_labels, return_counts=True)
    print(f"Train labels distribution before split: {dict(zip(unique, counts))}")

   
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.1, random_state=42, stratify=train_labels
    )

   
    unique_train, counts_train = np.unique(train_labels, return_counts=True)
    unique_val, counts_val = np.unique(val_labels, return_counts=True)
    print(f"Train labels distribution after split: {dict(zip(unique_train, counts_train))}")
    print(f"Validation labels distribution after split: {dict(zip(unique_val, counts_val))}")

    print(f"Loaded {len(train_texts)} training, {len(val_texts)} validation, and {len(test_texts)} test samples.")

    
    print("\n--- Step 2: Training TF-IDF + Logistic Regression ---")
    start_time = time.time()

    
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    X_val_tfidf = vectorizer.transform(val_texts)
    X_test_tfidf = vectorizer.transform(test_texts)

    
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_tfidf, train_labels)

   
    y_pred_lr_val = lr_model.predict(X_val_tfidf)
    val_acc_lr = accuracy_score(val_labels, y_pred_lr_val)

    y_pred_lr_test = lr_model.predict(X_test_tfidf)
    test_acc_lr = accuracy_score(test_labels, y_pred_lr_test)

    lr_train_time = time.time() - start_time

    print(f"TF-IDF+LR Validation Accuracy: {val_acc_lr:.4f}")
    print(f"TF-IDF+LR Test Accuracy: {test_acc_lr:.4f}")
    print(f"TF-IDF+LR Training Time: {lr_train_time:.2f} seconds")

    
    print("\n--- Step 3: Fine-tuning DistilBERT ---")

    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    
    def encode_texts(texts, labels, tokenizer, max_length=512):
        input_ids = []
        attention_masks = []
        for text in texts:
            encoded = tokenizer.encode_plus(
                text,
                add_special_tokens=True, # [CLS], [SEP]
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])

        
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels_tensor = torch.tensor(labels)
        return input_ids, attention_masks, labels_tensor

    
    train_input_ids, train_attention_masks, train_labels_tensor = encode_texts(train_texts, train_labels, tokenizer)
    val_input_ids, val_attention_masks, val_labels_tensor = encode_texts(val_texts, val_labels, tokenizer)
    test_input_ids, test_attention_masks, test_labels_tensor = encode_texts(test_texts, test_labels, tokenizer)

    
    class TextDataset(Dataset):
        def __init__(self, input_ids, attention_masks, labels):
            self.input_ids = input_ids
            self.attention_masks = attention_masks
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.input_ids[idx], self.attention_masks[idx], self.labels[idx]

    train_dataset = TextDataset(train_input_ids, train_attention_masks, train_labels_tensor)
    val_dataset = TextDataset(val_input_ids, val_attention_masks, val_labels_tensor)
    test_dataset = TextDataset(test_input_ids, test_attention_masks, test_labels_tensor)

    
    class DistilBERTClassifier(nn.Module):
        def __init__(self, model_name='distilbert-base-uncased', num_labels=2):
            super(DistilBERTClassifier, self).__init__()
            self.distilbert = DistilBertModel.from_pretrained(model_name)
            self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_labels)
            self.dropout = nn.Dropout(self.distilbert.config.seq_classif_dropout)

        def forward(self, input_ids, attention_mask):
            outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
            hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
            pooled_output = hidden_state[:, 0] # Take the [CLS] token representation
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            return logits

    model = DistilBERTClassifier()

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8) 
    batch_size = 16 
    epochs = 2 

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    def evaluate(model, dataloader, device):
        model.eval()
        predictions = []
        true_labels = []
        total_loss = 0
        for batch in dataloader:
            b_input_ids = batch[0].to(device)
            b_attention_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                outputs = model(b_input_ids, b_attention_mask)
                loss = nn.CrossEntropyLoss()(outputs, b_labels)
                total_loss += loss.item()

            logits = outputs.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            predictions.extend(np.argmax(logits, axis=1))
            true_labels.extend(label_ids)
        avg_loss = total_loss / len(dataloader)
        acc = accuracy_score(true_labels, predictions)
        return acc, avg_loss, predictions, true_labels

    start_time = time.time()

    
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1} / {epochs}")
        model.train()
        total_train_loss = 0
        for step, batch in enumerate(train_dataloader):
            if step % 100 == 0: # Print every 100 steps
                print(f"  Batch {step} of {len(train_dataloader)}")

            b_input_ids = batch[0].to(device)
            b_attention_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()
            outputs = model(b_input_ids, b_attention_mask)
            loss = nn.CrossEntropyLoss()(outputs, b_labels)
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        val_acc, val_loss, _, _ = evaluate(model, val_dataloader, device)
        print(f"  Average Training Loss: {avg_train_loss:.4f}")
        print(f"  Validation Accuracy: {val_acc:.4f}, Validation Loss: {val_loss:.4f}")
        
        
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

    distilbert_train_time = time.time() - start_time

    
    test_acc, test_loss, test_predictions, test_true_labels = evaluate(model, test_dataloader, device)
    print(f"\nDistilBERT Test Accuracy: {test_acc:.4f}")
    print(f"DistilBERT Training Time: {distilbert_train_time:.2f} seconds")
    print("\nDistilBERT Test Classification Report:")
    print(classification_report(test_true_labels, test_predictions, target_names=['Negative', 'Positive']))

    
    print("\n--- Step 4: Generating Visualizations ---")                                                                                                                                                                    #搞论文aigc比我他妈的自己写代码都累，真他妈傻逼，                                                                              #
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.title('DistilBERT Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_plot_path = os.path.join(results_dir, "distilbert_loss_curve.png")
    plt.savefig(loss_plot_path)
    print(f"Loss curve saved to {loss_plot_path}")
    plt.close() 

   
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy', marker='d', color='orange')
    plt.title('DistilBERT Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    acc_plot_path = os.path.join(results_dir, "distilbert_accuracy_curve.png")
    plt.savefig(acc_plot_path)
    print(f"Accuracy curve saved to {acc_plot_path}")
    plt.close()

   
    cm = confusion_matrix(test_true_labels, test_predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title('DistilBERT Confusion Matrix on Test Set')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    cm_plot_path = os.path.join(results_dir, "distilbert_confusion_matrix.png")
    plt.savefig(cm_plot_path)
    print(f"Confusion matrix saved to {cm_plot_path}")
    plt.close()

    
    print("\n--- Final Results Summary ---")
    print(f"TF-IDF + Logistic Regression - Test Acc: {test_acc_lr:.4f}, Train Time: {lr_train_time:.2f}s")
    print(f"DistilBERT - Test Acc: {test_acc:.4f}, Train Time: {distilbert_train_time:.2f}s")

    
    with open(os.path.join(results_dir, "experiment_log.txt"), "w") as f:
        f.write("--- Final Results Summary ---\n")
        f.write(f"TF-IDF + Logistic Regression - Test Acc: {test_acc_lr:.4f}, Train Time: {lr_train_time:.2f}s\n")
        f.write(f"DistilBERT - Test Acc: {test_acc:.4f}, Train Time: {distilbert_train_time:.2f}s\n")
        f.write("\nDistilBERT Test Classification Report:\n")
        f.write(classification_report(test_true_labels, test_predictions, target_names=['Negative', 'Positive']))
        f.write("\n\nVisualization files saved in 'results/' directory.")

    print(f"\nResults also saved to {os.path.join(results_dir, 'experiment_log.txt')}")
    print(f"Visualizations saved in the '{results_dir}' directory.")

if __name__ == "__main__":
    main()