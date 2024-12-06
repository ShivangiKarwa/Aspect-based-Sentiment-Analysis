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
from transformers import (
    RobertaTokenizer, 
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import StratifiedKFold
from main import xml_to_csv, calculate_class_weights

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def subset_dataset(dataset, fraction=0.5, stratify=True):
    
    total_samples = len(dataset)
    subset_size = int(total_samples * fraction)
    
    if stratify:

        labels = [dataset[i]['labels'].item() for i in range(len(dataset))]
        
        # Use StratifiedKFold to maintain class distribution
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, _ = next(skf.split(np.zeros(len(dataset)), labels))
        
        
        subset_indices = train_idx[:subset_size]
    else:
        
        subset_indices = np.random.choice(total_samples, subset_size, replace=False)
    
    return Subset(dataset, subset_indices)

def train_and_evaluate(
    model, 
    train_loader, 
    valid_loader, 
    optimizer, 
    loss_fn, 
    scheduler, 
    device, 
    num_epochs=10
):
   
    best_valid_accuracy = 0
    best_model_state = None
    
    train_accuracies = []
    valid_accuracies = []
    
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
        
        # Validation Phase
        model.eval()
        valid_loss, valid_correct, valid_total = 0, 0, 0
        
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"Epoch {epoch+1} Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = loss_fn(outputs, labels)
                
                valid_loss += loss.item()
                valid_correct += (outputs.argmax(1) == labels).sum().item()
                valid_total += labels.size(0)
        
        # Calculate accuracies
        train_accuracy = train_correct / train_total
        valid_accuracy = valid_correct / valid_total
        
        train_accuracies.append(train_accuracy)
        valid_accuracies.append(valid_accuracy)
        
        scheduler.step(valid_accuracy)
        
        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            best_model_state = model.state_dict()
        
        logger.info(f'Epoch {epoch+1}: Train Acc {train_accuracy:.4f}, Valid Acc {valid_accuracy:.4f}')
    
    # Restore best model state
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return {
        'train_accuracies': train_accuracies,
        'valid_accuracies': valid_accuracies,
        'best_valid_accuracy': best_valid_accuracy
    }

def objective(trial):
   
    # Hyperparameters to optimize
    lr = trial.suggest_loguniform('lr', 1e-6, 1e-4)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
    dropout_rate = trial.suggest_categorical('dropout_rate', [0.1, 0.2, 0.3, 0.4])
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    label_smoothing = trial.suggest_uniform('label_smoothing', 0.0, 0.3)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Running on device: {device}')
    
    train_file = 'train.xml'
    valid_file = 'val.xml'
    
    train_df = xml_to_csv(train_file)
    valid_df = xml_to_csv(valid_file)
    
    tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')
    
    # Add custom tokens
    new_tokens = ['<aspect>', '</aspect>']
    tokenizer.add_tokens(new_tokens)
    
    train_dataset = AspectSentimentDataset(train_df, tokenizer)
    valid_dataset = AspectSentimentDataset(valid_df, tokenizer)
    
    train_dataset = subset_dataset(train_dataset, fraction=0.5)
    valid_dataset = subset_dataset(valid_dataset, fraction=0.5)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    model = AspectSentimentRoBERTa(dropout_rate=dropout_rate).to(device)
    model.roberta.resize_token_embeddings(len(tokenizer))
    
    class_weights = calculate_class_weights(train_df).to(device)
    
    loss_fn = nn.CrossEntropyLoss(
        weight=class_weights, 
        label_smoothing=label_smoothing
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay
    )
    
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 
    #     mode='max', 
    #     factor=0.5, 
    #     patience=2
    # )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=5,      
            T_mult=2,   
            eta_min=1e-6 )
    # Train and evaluate
    results = train_and_evaluate(
        model, 
        train_loader, 
        valid_loader, 
        optimizer, 
        loss_fn, 
        scheduler, 
        device
    )
    
    return results['best_valid_accuracy']

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization for Aspect Sentiment Analysis')
    parser.add_argument('--n_trials', type=int, default=25, help='Number of Optuna trials')
    args = parser.parse_args()
    
    # Create a study object and optimize the objective function
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.n_trials)
    
    print('Number of finished trials:', len(study.trials))
    print('Best trial:')
    trial = study.best_trial
    
    print('  Value: ', trial.value)
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))
    
    import optuna.visualization as vis
    
    # Parallel Coordinates Plot
    fig = vis.plot_parallel_coordinate(study)
    fig.write_image("hyperparameter_parallel_coordinate.png")
    
    # Contour Plot
    fig = vis.plot_contour(study)
    fig.write_image("hyperparameter_contour.png")

if __name__ == "__main__":
    main()