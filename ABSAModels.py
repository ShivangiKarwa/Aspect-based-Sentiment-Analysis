import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertForMaskedLM
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import DistilBertModel, DistilBertTokenizer
from transformers import RobertaTokenizer, RobertaModel
from sklearn import preprocessing
import re



sentiment_map = {'positive':0, 'negative':1, 'neutral':2}
inverse_sentiment_map = {v: k for k, v in sentiment_map.items()}


class SentimentDatasetRoBERTa(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.texts = dataframe['text'].tolist()
        self.aspects = dataframe['aspect'].tolist()
        
        # Convert span values to integers, handling potential string inputs
        self.spans = list(zip(
            pd.to_numeric(dataframe['polarity_from'], errors='coerce').fillna(0).astype(int),
            pd.to_numeric(dataframe['polarity_to'], errors='coerce').fillna(len(dataframe['text'])).astype(int)
        ))
        
        self.sentiments = [sentiment_map[s] for s in dataframe['sentiment']]
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        aspect = self.aspects[idx]
        span_from, span_to = self.spans[idx]
        
        span_from = max(0, span_from)
        span_to = min(len(text), span_to)
        
        # Create a marker for the aspect span
        marked_text = text[:span_from] + " <aspect> " + text[span_from:span_to] + " </aspect> " + text[span_to:]
        
        encoding = self.tokenizer(
            marked_text + "</s></s>" + aspect,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.sentiments[idx], dtype=torch.long)
        }



class RoBERTa(nn.Module):
    def __init__(self, num_labels=3, dropout_rate=0.4):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('distilroberta-base')
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels)
        )
        # self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class SentimentDatasetBOW(Dataset):
    def __init__(self, dataFrame, vectorizer=None, train=True, aspect_vocab=None):
        def preprocess_text_with_aspect_and_polarity(text, aspect, span_from, span_to):
            # Mark the polarity span in the text
            marked_text = text[:span_from] + " <polarity> " + text[span_from:span_to] + " </polarity> " + text[span_to:]
            return f"{aspect} {marked_text}"

        # Create or use provided aspect vocabulary
        if aspect_vocab is None:
            self.aspect_vocab = {aspect: idx for idx, aspect in enumerate(dataFrame['aspect'].unique())}
        else:
            self.aspect_vocab = aspect_vocab

        # Add "UNKNOWN" token for missing aspects
        if "UNKNOWN" not in self.aspect_vocab:
            self.aspect_vocab["UNKNOWN"] = len(self.aspect_vocab)

        self.aspect_idxs = dataFrame['aspect'].map(self.aspect_vocab).fillna(self.aspect_vocab["UNKNOWN"]).astype(int).tolist()

        # Preprocess sentences with aspect terms and polarity spans
        self.sentences = dataFrame.apply(
            lambda row: preprocess_text_with_aspect_and_polarity(
                row['text'], 
                row['aspect'], 
                int(row['polarity_from']), 
                int(row['polarity_to'])
            ),
            axis=1
        )
        self.labels = dataFrame['sentiment'].map({'positive': 0, 'negative': 1, 'neutral': 2}).tolist()

        if train or vectorizer is None:
            self.vectorizer = CountVectorizer(max_features=512, ngram_range=(1, 3), stop_words='english')
            self.embeddings = self.vectorizer.fit_transform(self.sentences).toarray()
        else:
            self.vectorizer = vectorizer
            self.embeddings = self.vectorizer.transform(self.sentences).toarray()

        # Normalize embeddings
        self.embeddings = preprocessing.normalize(self.embeddings)

        # Convert embeddings and labels to tensors
        self.embeddings = torch.tensor(self.embeddings, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.aspect_idxs = torch.tensor(self.aspect_idxs, dtype=torch.long)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], self.aspect_idxs[idx]



class NN3BOW(nn.Module):
    def __init__(self, input_size, hidden_size, aspect_vocab_size, aspect_embedding_dim, dropout_rate):
        super().__init__()
        # Aspect embedding layer
        self.aspect_embedding = nn.Embedding(aspect_vocab_size, aspect_embedding_dim)

        # Fully connected layers
        self.fc1 = nn.Linear(input_size + aspect_embedding_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 3)  # Output layer for 3 classes

       
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, aspect_idx):
        
        aspect_embed = self.aspect_embedding(aspect_idx)

        x = torch.cat([x, aspect_embed], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

