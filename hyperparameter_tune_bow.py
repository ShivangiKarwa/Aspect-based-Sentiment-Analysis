import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import optuna
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import logging
from ABSAModels import SentimentDatasetBOW, AspectSentimentDataset, AspectSentimentRoBERTa, NN3BOW
import itertools
from torch.utils.data import DataLoader
from main import xml_to_csv

# # Import necessary modules
# from transformers import (
#     RobertaTokenizer, 
#     get_linear_schedule_with_warmup
# )
# from sklearn.model_selection import StratifiedKFold

# # Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# def xml_to_csv(file_path):
#     # Parse the XML

#     with open(file_path, 'r') as d:
#         data = d.read()

#     root = ET.fromstring(data)
#     csv_data = []


#     for sentence in root.findall('sentence'):

#       # Extract full text
#       text = sentence.find('text').text

#       # Extract aspect terms
#       if(sentence.find('aspectTerms') is not None):

#         aspect_terms = sentence.find('aspectTerms')

#         # Process each aspect term
#         for aspect_term in (aspect_terms.findall('aspectTerm') or aspect_terms.findall('aspectCategory')):
#           csv_data.append({
#               'text': text,
#               'aspect': aspect_term.get('category') if aspect_terms.findall('aspectTerm') is None else aspect_term.get('term'),
#               'polarity_from': aspect_term.get('from'),
#               'polarity_to': aspect_term.get('to'),
#               'sentiment': aspect_term.get('polarity')
#           })

#     # Create DataFrame
#     df = pd.DataFrame(csv_data)

#     return df


# def detailed_evaluation(model, test_loader):
#     model.eval()
#     all_preds = []
#     all_labels = []
    
#     with torch.no_grad():
#         for X, y, aspect in test_loader:
#             pred = model(X, aspect)
#             all_preds.extend(pred.argmax(1).numpy())
#             all_labels.extend(y.numpy())
    
#     print(classification_report(all_labels, all_preds))
#     print(confusion_matrix(all_labels, all_preds))




def calculate_class_weights(df):
    sentiment_counts = df['sentiment'].value_counts()
    total_samples = len(df)
    
    # Inverse frequency with power scaling to prevent over-compensation
    class_weights = torch.tensor([
        (total_samples / sentiment_counts['negative']) ** 0.5,
        (total_samples / sentiment_counts['neutral']) ** 0.5,
        (total_samples / sentiment_counts['positive']) ** 0.5
    ], dtype=torch.float)
    
    return class_weights / class_weights.mean()


# def data_preprocess(train_df):



#     train_df.dropna(inplace = True)

#     print(train_df['sentiment'].value_counts())
    

#     return train_df

# def subset_dataset(dataset, fraction=0.5, stratify=True):
#     """
#     Create a subset of the dataset for rapid experimentation
    
#     Args:
#         dataset: Original PyTorch dataset
#         fraction: Fraction of dataset to sample
#         stratify: Whether to maintain class distribution
    
#     Returns:
#         Subset of the original dataset
#     """
#     total_samples = len(dataset)
#     subset_size = int(total_samples * fraction)
    
#     if stratify:
#         print(dataset)
#         # Extract labels for stratification
#         # labels = [dataset[i]['labels'].item() for i in range(len(dataset))]

#         labels = dataset.labels.tolist()
        
#         # Use StratifiedKFold to maintain class distribution
#         skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#         train_idx, _ = next(skf.split(np.zeros(len(dataset)), labels))
        
#         # Select first subset_size indices
#         subset_indices = train_idx[:subset_size]
#     else:
#         # Random sampling
#         subset_indices = np.random.choice(total_samples, subset_size, replace=False)
    
#     return Subset(dataset, subset_indices)

def train_and_evaluate(
    model,
    train_loader,
    valid_loader,
    optimizer,
    loss_fn,
    scheduler,
    device,
    num_epochs=100
):
    best_valid_accuracy = 0
    best_model_state = None
    train_accuracies = []
    valid_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training")):
            X, y, aspect_idx = batch 
            
            pred = model(X, aspect_idx)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (pred.argmax(1) == y).sum().item()
            train_total += y.size(0)

        # Validation Phase
        model.eval()
        valid_loss, valid_correct, valid_total = 0, 0, 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(valid_loader, desc=f"Epoch {epoch+1} Validation")):
                X, y, aspect_idx = batch
                
                pred = model(X, aspect_idx)
                loss = loss_fn(pred, y)
                
                valid_loss += loss.item()
                valid_correct += (pred.argmax(1) == y).sum().item()
                valid_total += y.size(0)

        # Calculate accuracies
        train_accuracy = train_correct / train_total
        valid_accuracy = valid_correct / valid_total
        
        train_accuracies.append(train_accuracy)
        valid_accuracies.append(valid_accuracy)
        
        scheduler.step(valid_loss)
        
        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            best_model_state = model.state_dict()

        if epoch%10 ==9 :
            logger.info(f'Epoch {epoch+1}: Train Acc {train_accuracy:.4f}, Valid Acc {valid_accuracy:.4f}')

    if best_model_state:
        model.load_state_dict(best_model_state)
        
    return {
        'train_accuracies': train_accuracies,
        'valid_accuracies': valid_accuracies,
        'best_valid_accuracy': best_valid_accuracy
    }


# def xml_to_csv(file_path):
#     # Parse the XML

#     with open(file_path, 'r') as d:
#         data = d.read()

#     root = ET.fromstring(data)
#     csv_data = []


#     for sentence in root.findall('sentence'):

#       # Extract full text
#       text = sentence.find('text').text

#       # Extract aspect terms
#       if(sentence.find('aspectTerms') is not None):

#         aspect_terms = sentence.find('aspectTerms')

#         # Process each aspect term
#         for aspect_term in (aspect_terms.findall('aspectTerm') or aspect_terms.findall('aspectCategory')):
#           csv_data.append({
#               'text': text,
#               'aspect': aspect_term.get('category') if aspect_terms.findall('aspectTerm') is None else aspect_term.get('term'),
#               'polarity_from': aspect_term.get('from'),
#               'polarity_to': aspect_term.get('to'),
#               'sentiment': aspect_term.get('polarity')
#           })

#     # Create DataFrame
#     df = pd.DataFrame(csv_data)

#     return df


def grid_search_cv(model_class, train_dataset, valid_dataset, param_grid, cv=5):
    best_score = 0
    best_params = None
    results = []  # Initialize as a list
    
    # Create all possible combinations of parameters
    param_combinations = [dict(zip(param_grid.keys(), v)) 
                         for v in itertools.product(*param_grid.values())]

    class_weights = calculate_class_weights(train_df).to(device)
    
    for params in tqdm(param_combinations, desc="Parameter Combinations"):
        # Initialize model with current parameters
        model = model_class(
            input_size=512,
            hidden_size=params['hidden_size'],
            aspect_vocab_size=len(train_dataset.aspect_vocab),
            aspect_embedding_dim=params['aspect_embedding_dim'],
            dropout_rate=params['dropout_rate']
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=params['batch_size'], 
            shuffle=True
        )
        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=params['batch_size'], 
            shuffle=False
        )
        
        # Initialize optimizer and loss function
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params['lr'],
            weight_decay=params['weight_decay']
        )
        
        loss_fn = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=params['label_smoothing']
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Train and evaluate
        training_results = train_and_evaluate(
            model,
            train_loader,
            valid_loader,
            optimizer,
            loss_fn,
            scheduler,
            device,
            num_epochs=50
        )
        
        # Track best parameters
        current_score = training_results['best_valid_accuracy']
        if current_score > best_score:
            best_score = current_score
            best_params = params
        
        # Append results to the list
        results.append({
            'params': params,
            'valid_accuracy': current_score
        })
    
    return best_params, best_score, results


# Define parameter grid
param_grid = {
    'lr': [1e-4, 1e-3, 1e-2],
    'weight_decay': [1e-4, 1e-3],
    'dropout_rate': [0.1, 0.3, 0.5],
    'batch_size': [16, 32],
    'label_smoothing': [0.0, 0.1, 0.2],
    'aspect_embedding_dim': [20, 50],
    'hidden_size': [100, 200]
}


train_file = 'train.xml'
valid_file = 'val.xml'

train_df = xml_to_csv(train_file)
valid_df = xml_to_csv(valid_file)

all_aspects = pd.concat([train_df, valid_df])['aspect'].unique()
aspect_vocab = {aspect: idx for idx, aspect in enumerate(all_aspects)}

train_dataset = SentimentDatasetBOW(train_df, aspect_vocab=aspect_vocab)
valid_dataset = SentimentDatasetBOW(valid_df, vectorizer=train_dataset.vectorizer, train=False, aspect_vocab=aspect_vocab)

best_params, best_score, all_results = grid_search_cv(
    NN3BOW,
    train_dataset,
    valid_dataset,
    param_grid
)

print(f"Best parameters: {best_params}")
print(f"Best validation accuracy: {best_score:.4f}")

# Plot results
plt.figure(figsize=(10, 6))
accuracies = [r['valid_accuracy'] for r in all_results]
plt.plot(accuracies)
plt.title('Validation Accuracy for Different Parameter Combinations')
plt.xlabel('Combination Index')
plt.ylabel('Validation Accuracy')
plt.savefig('grid_search_results.png')
plt.close()