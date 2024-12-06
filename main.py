import xml.etree.ElementTree as ET
import pandas as pd
import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torch.optim as optim
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from ABSAModels import SentimentDatasetBOW, SentimentDatasetRoBERTa, RoBERTa, NN3BOW
from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertTokenizer, DistilBertTokenizer
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau,CosineAnnealingWarmRestarts
# from ignite.handlers import EarlyStopping
from sklearn import preprocessing
from transformers import RobertaTokenizer
from sklearn.metrics import classification_report, confusion_matrix



torch.manual_seed(42)

class_weights = []

sentiment_map = {'positive':0, 'negative':1, 'neutral':2}
reverse_sentiment_map = {v: k for k, v in sentiment_map.items()}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def xml_to_csv(file_path):
    # Parse the XML

    with open(file_path, 'r') as d:
        data = d.read()

    root = ET.fromstring(data)
    csv_data = []


    for sentence in root.findall('sentence'):

      text = sentence.find('text').text

      # Extract aspect terms
      if(sentence.find('aspectTerms') is not None):

        aspect_terms = sentence.find('aspectTerms')

        # Process each aspect term
        for aspect_term in (aspect_terms.findall('aspectTerm') or aspect_terms.findall('aspectCategory')):
          csv_data.append({
              'text': text,
              'aspect': aspect_term.get('category') if aspect_terms.findall('aspectTerm') is None else aspect_term.get('term'),
              'polarity_from': aspect_term.get('from'),
              'polarity_to': aspect_term.get('to'),
              'sentiment': aspect_term.get('polarity')
          })

    df = pd.DataFrame(csv_data)

    return df




def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y, aspect_idx) in enumerate(data_loader):

        # Compute prediction error
        pred = model(X, aspect_idx)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss

def eval_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y, aspect_idx) in enumerate(data_loader):

        # Compute prediction error
        pred = model(X, aspect_idx)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss

def experiment(model, train_loader, valid_loader, class_weights):

    loss_fn = nn.CrossEntropyLoss(weight = class_weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )

    all_train_accuracy = []
    all_valid_accuracy = []
    best_valid_accuracy = 0.0
    best_model_state = None

    for epoch in range(100):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)

        valid_accuracy, test_loss = eval_epoch(valid_loader, model, loss_fn, optimizer)
        all_valid_accuracy.append(valid_accuracy)

        scheduler.step(valid_accuracy)

        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            best_model_state = model.state_dict()

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, valid accuracy {valid_accuracy:.3f}')
    
    model.load_state_dict(best_model_state)
    
    return model, all_train_accuracy, all_valid_accuracy



def detailed_evaluation_bow(model, test_loader, test_df):
    model.eval()
    all_preds = []
    all_labels = []
    misclassified_examples = []

    
    with torch.no_grad():
        for batch_idx, (X, y, aspect) in enumerate(test_loader):
            pred = model(X, aspect)
            all_preds.extend(pred.argmax(1).numpy())
            all_labels.extend(y.numpy())

            # Store misclassified examples
            for i, (pred, true) in enumerate(zip(all_preds, all_labels)):
                if pred != true:
                    idx = batch_idx * test_loader.batch_size + i
                    misclassified_examples.append({
                        'text': test_df.iloc[idx]['text'],
                        'aspect': test_df.iloc[idx]['aspect'],
                        'true_label': reverse_sentiment_map[true.item()],
                        'predicted_label': reverse_sentiment_map[pred.item()]
                    })
                    if len(misclassified_examples) >= 5:
                        break
    
    # print("\nMisclassified Examples:")
    # for idx, example in enumerate(misclassified_examples[:5]):
    #     print(f"\nExample {idx + 1}:")
    #     print(f"Text: {example['text']}")
    #     print(f"Aspect: {example['aspect']}")
    #     print(f"True Label: {example['true_label']}")
    #     print(f"Predicted Label: {example['predicted_label']}")
    

    print("\nClassification Report:")
    print(classification_report(
        all_labels, 
        all_preds, 
        target_names=[
            reverse_sentiment_map[0], 
            reverse_sentiment_map[1], 
            reverse_sentiment_map[2]
        ]
    ))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print(conf_matrix)
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = [
        reverse_sentiment_map[0], 
        reverse_sentiment_map[1], 
        reverse_sentiment_map[2]
    ]
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations to the confusion matrix
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('bow_confusion_matrix.png')
    plt.close()


def detailed_evaluation_roberta(model, test_loader):

    model.eval()
    all_preds = []
    all_labels = []
    
    # Disable gradient computation for evaluation
    with torch.no_grad():

        for batch in test_loader:

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            preds = outputs.argmax(1)
            
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(
        all_labels, 
        all_preds, 
        target_names=[
            reverse_sentiment_map[0], 
            reverse_sentiment_map[1], 
            reverse_sentiment_map[2]
        ]
    ))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    conf_matrix = confusion_matrix(all_labels, all_preds)
    print(conf_matrix)
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = [
        reverse_sentiment_map[0], 
        reverse_sentiment_map[1], 
        reverse_sentiment_map[2]
    ]
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations to the confusion matrix
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('roberta_confusion_matrix.png')
    plt.close()



def calculate_class_weights(df):
    sentiment_counts = df['sentiment'].value_counts()
    total_samples = len(df)
    
    # Inverse frequency with power scaling 
    class_weights = torch.tensor([
        (total_samples / sentiment_counts['negative']) ** 0.5,
        (total_samples / sentiment_counts['neutral']) ** 0.5,
        (total_samples / sentiment_counts['positive']) ** 0.5
    ], dtype=torch.float)
    
    return class_weights / class_weights.mean()


def data_preprocess(train_df):

    train_df.dropna(inplace = True)

    print(train_df['sentiment'].value_counts())
    
    return train_df




def main():
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW, BERT)')
    parser.add_argument('--quick_experiment', action='store_true', 
                        help='Use a smaller dataset for rapid experimentation')
    parser.add_argument('--lr_finder', action='store_true', 
                        help='Run learning rate range test')
    args = parser.parse_args()

    train_file = 'train.xml'
    test_file = 'test.xml'
    valid_file = 'val.xml'

    train_df = xml_to_csv(train_file)
    valid_df = xml_to_csv(valid_file)
    test_df = xml_to_csv(test_file)

    train_df = data_preprocess(train_df)

    class_weights = calculate_class_weights(train_df)


    all_aspects = pd.concat([train_df])['aspect'].unique()
    aspect_vocab = {aspect: idx for idx, aspect in enumerate(all_aspects)}


    

    if args.model == 'BOW':

        train_data = SentimentDatasetBOW(train_df, aspect_vocab=aspect_vocab)
        valid_data = SentimentDatasetBOW(valid_df, vectorizer=train_data.vectorizer, train=False, aspect_vocab=aspect_vocab)
        test_data = SentimentDatasetBOW(test_df, vectorizer=train_data.vectorizer, train=False, aspect_vocab=aspect_vocab)

        train_loader = DataLoader(train_data,   batch_size=16, shuffle=True)
        valid_loader = DataLoader(valid_data,  batch_size=16, shuffle = False)
        test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

        aspect_vocab_size = len(aspect_vocab)
        model = NN3BOW(input_size=512, hidden_size=100, aspect_vocab_size=aspect_vocab_size, aspect_embedding_dim=20, dropout_rate = 0.5)

        print('\n3 layers:')
        model, nn2_train_accuracy, nn2_valid_accuracy = experiment(model, train_loader, valid_loader,class_weights)


        detailed_evaluation_bow(model,test_loader, test_df)

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_train_accuracy, label='3 layers')
        
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_valid_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Valid Accuracy')
        plt.title('Valid Accuracy for 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'valid_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"test accuracy plot saved as {testing_accuracy_file}\n\n")

    elif args.model == 'RoBERTa':

         
        tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')

        model = AspectSentimentRoBERTa().to(device)

        new_tokens = ['<aspect>', '</aspect>']
        tokenizer.add_tokens(new_tokens)
        model.roberta.resize_token_embeddings(len(tokenizer))

        # Create datasets
        train_dataset = AspectSentimentDataset(train_df, tokenizer)
        valid_dataset = AspectSentimentDataset(valid_df, tokenizer)
        test_dataset = AspectSentimentDataset(test_df, tokenizer)

        # Create data loaders
        train_loader = DataLoader(train_dataset,batch_size=32, shuffle=True)
        valid_loader = DataLoader(valid_dataset,batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset,batch_size=32, shuffle=False)

        
        

        # Prepare loss function and optimizer
        loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.27)

        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-3)

        # optimizer = torch.optim.Adam(model.parameters(), lr=1.199e-5, weight_decay=0.0034)

        # optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.001)

        optimizer = torch.optim.AdamW(model.parameters(), lr=4.92e-5, weight_decay=0.00065)

        # Training loop
        all_train_accuracy = []
        all_valid_accuracy = []
        best_valid_accuracy = 0.0

        # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=5,      
            T_mult=2,
            eta_min=1e-6  
        )

        
        # early_stopping = EarlyStopping(patience=3)

        for epoch in tqdm(range(10), desc="Training Epochs", position=0):
            # Training phase
            model.train()
            train_loss, train_correct, train_total = 0, 0, 0

            train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", position=1, leave=False)
            for batch in train_progress:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()

                outputs = model(input_ids, attention_mask)
                loss = loss_fn(outputs, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
                optimizer.step()

                train_loss += loss.item()
                train_correct += (outputs.argmax(1) == labels).sum().item()
                train_total += labels.size(0)

                train_progress.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{(train_correct / train_total) * 100:.2f}%'
                })

            # Validation phase
            model.eval()
            valid_loss, valid_correct, valid_total = 0, 0, 0

            valid_progress = tqdm(valid_loader, desc=f"Epoch {epoch+1} Validation", position=1, leave=False)
            with torch.no_grad():
                for batch in valid_progress:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    outputs = model(input_ids, attention_mask)
                    loss = loss_fn(outputs, labels)

                    valid_loss += loss.item()
                    valid_correct += (outputs.argmax(1) == labels).sum().item()
                    valid_total += labels.size(0)

                    # Update tqdm progress bar
                    valid_progress.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{(valid_correct / valid_total) * 100:.2f}%'
                    })

            # Calculate accuracies
            train_accuracy = train_correct / train_total
            valid_accuracy = valid_correct / valid_total
            all_train_accuracy.append(train_accuracy)
            all_valid_accuracy.append(valid_accuracy)

            if valid_accuracy > best_valid_accuracy:
                best_valid_accuracy = valid_accuracy
                best_model_state = model.state_dict()
            
            print(f'\nEpoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, valid accuracy {valid_accuracy:.3f}')
            scheduler.step(epoch)
            # early_stopping(test_accuracy, model)
            # if early_stopping.early_stop:
            #     print("Early stopping triggered")
            #     break

        model.load_state_dict(best_model_state)

        detailed_evaluation_roberta(model,test_loader)

        
        # Plot training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(all_train_accuracy, label='BERT Training')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for BERT')
        plt.legend()
        plt.grid()
        training_accuracy_file = 'bert_train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot valid accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(all_valid_accuracy, label='BERT Valid')
        plt.xlabel('Epochs')
        plt.ylabel('Valid Accuracy')
        plt.title('Valid Accuracy for BERT')
        plt.legend()
        plt.grid()
        testing_accuracy_file = 'bert_valid_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Test accuracy plot saved as {testing_accuracy_file}\n")






if __name__ == "__main__":
    main()

