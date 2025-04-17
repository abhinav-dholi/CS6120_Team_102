import os
import re
import json
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from model import precision, recall, f1_score
from data_utils import split_text, clean_word

def load_model_artifacts(models_dir, processed_dir):
    """
    Loads the model and its artifacts.
    """
    # loads model
    custom_objects = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }
    
    model = load_model(os.path.join(models_dir, 'ner_model.h5'), custom_objects=custom_objects)
    
    # loads tokenizer
    with open(os.path.join(models_dir, 'tokenizer.pickle'), 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    # loads model metadata
    with open(os.path.join(models_dir, 'model_metadata.json'), 'r') as f:
        model_metadata = json.load(f)
        
    max_length = model_metadata['max_length']
    label_to_index = model_metadata['label_to_index']
    index_to_label = {int(k): v for k, v in model_metadata['index_to_label'].items()}
    
    # loads entity mappings
    with open(os.path.join(processed_dir, 'entity_mappings.json'), 'r') as f:
        entity_mappings = json.load(f)
        
    acronyms_to_entities = entity_mappings['acronyms_to_entities']
    
    return model, tokenizer, max_length, label_to_index, index_to_label, acronyms_to_entities

def generate_classification_report(model, X_test, y_test, index_to_label, label_to_index, output_path=None):
    """
    Generates a classification report for the model's performance.
    """
    # gets model predictions
    y_pred = model.predict(X_test)
    
    # converts one-hot encoded predictions and labels to class indices
    y_pred_indices = np.argmax(y_pred, axis=2).flatten()
    y_true_indices = np.argmax(y_test, axis=2).flatten()
    
    # creates a mask to filter out padding
    mask = y_true_indices != label_to_index['<PAD>']
    
    # applies mask
    y_pred_indices = y_pred_indices[mask]
    y_true_indices = y_true_indices[mask]
    
    # converts indices to labels
    y_pred_labels = [index_to_label[idx] for idx in y_pred_indices]
    y_true_labels = [index_to_label[idx] for idx in y_true_indices]
    
    # generates and print the classification report
    report = classification_report(y_true_labels, y_pred_labels, zero_division=0)
    print(report)
    
    # saves report as text file if output path is provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
    
    return y_true_labels, y_pred_labels

def preprocess_text_for_prediction(text, tokenizer, max_length):
    """
    Preprocesses text for making predictions with the model.
    """
    # splits text into tokens and get their ranges
    tokens, token_ranges, sentence_breaks = split_text(text)
    
    # clean tokens
    cleaned_tokens = [clean_word(token) for token in tokens]
    
    # convert tokens to indices
    token_indices = []
    for token in cleaned_tokens:
        idx = tokenizer.word_index.get(token, 1)  # 1 is OOV
        token_indices.append(idx)
    
    # pads sequences
    X_padded = pad_sequences([token_indices], maxlen=max_length, padding='post', truncating='post')
    
    return tokens, token_indices, X_padded, token_ranges, sentence_breaks

def predict_entities(text, model, tokenizer, max_length, index_to_label, acronyms_to_entities):
    """
    Predicts entities in text using the trained NER model.
    """
    # preprocesses text
    original_tokens, token_indices, X_padded, token_ranges, sentence_breaks = preprocess_text_for_prediction(
        text, tokenizer, max_length)
    
    # makes predictions
    y_pred = model.predict(X_padded)[0]
    
    # gets the predicted label indices
    pred_indices = np.argmax(y_pred, axis=1)
    
    # converts indices to labels
    predicted_labels = []
    predicted_entities = []
    
    for i in range(min(len(original_tokens), max_length)):
        label = index_to_label[pred_indices[i]]
        predicted_labels.append(label)
        
        # extracts entity type from label (B-XXX or I-XXX)
        if label.startswith('B-') or label.startswith('I-'):
            entity_acronym = label[2:]  # removes B- or I- prefix
            entity_type = acronyms_to_entities.get(entity_acronym, entity_acronym)
            predicted_entities.append(entity_type)
        else:
            predicted_entities.append('O')
    
    return original_tokens, predicted_labels, predicted_entities, token_ranges, sentence_breaks

class ClinicalNERPredictor:
    """
    Predictor class for deploying the clinical NER model.
    """
    
    def __init__(self, model_path, tokenizer_path, metadata_path, entity_mappings_path):
        """
        Initializes the predictor with necessary files.
        """
        # loads model
        self.model = load_model(model_path, custom_objects={
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        })
        
        # loads tokenizer
        with open(tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        
        # loads metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            self.max_length = metadata['max_length']
            self.label_to_index = metadata['label_to_index']
            self.index_to_label = {int(k): v for k, v in metadata['index_to_label'].items()}
        
        # loads entity mappings
        with open(entity_mappings_path, 'r') as f:
            entity_mappings = json.load(f)
            self.acronyms_to_entities = entity_mappings['acronyms_to_entities']
        
        # loads spaCy
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Downloading spaCy model...")
            os.system('python -m spacy download en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
    
    def predict(self, text):
        """
        Predicts entities in the given text.
        """
        # preprocesses and predict
        tokens, labels, entities, token_ranges, sentence_breaks = predict_entities(
            text, self.model, self.tokenizer, self.max_length, 
            self.index_to_label, self.acronyms_to_entities)
        
        # creates list of identified entities with positions
        entity_list = []
        current_entity = None
        start_idx = None
        entity_text = []
        
        for i, (token, label, entity, (start, end)) in enumerate(zip(tokens, labels, entities, token_ranges)):
            if label.startswith('B-'):
                # if we were tracking an entity, add it to our list
                if current_entity is not None:
                    entity_list.append({
                        'text': ' '.join(entity_text),
                        'entity_type': current_entity,
                        'start': start_idx,
                        'end': token_ranges[i-1][1]
                    })
                
                # start tracking a new entity
                current_entity = entity
                start_idx = start
                entity_text = [token]
                
            elif label.startswith('I-'):
                # continue with current entity
                if current_entity is not None:
                    entity_text.append(token)
            
            elif label == 'O':
                # if we were tracking an entity, add it to our list
                if current_entity is not None:
                    entity_list.append({
                        'text': ' '.join(entity_text),
                        'entity_type': current_entity,
                        'start': start_idx,
                        'end': token_ranges[i-1][1]
                    })
                    current_entity = None
                    start_idx = None
                    entity_text = []
        
        # add the last entity if there is one
        if current_entity is not None:
            entity_list.append({
                'text': ' '.join(entity_text),
                'entity_type': current_entity,
                'start': start_idx,
                'end': token_ranges[-1][1]
            })
        
        # count entities by type
        entity_counts = Counter([e['entity_type'] for e in entity_list])
        
        return {
            'tokens': tokens,
            'labels': labels,
            'entity_list': entity_list,
            'entity_counts': dict(entity_counts),
            'token_ranges': token_ranges
        }
    
    def save(self, output_dir):
        """
        Saves the predictor for deployment.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # saves model
        self.model.save(os.path.join(output_dir, 'model.h5'))
        
        # saves tokenizer
        with open(os.path.join(output_dir, 'tokenizer.pickle'), 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # saves metadata
        metadata = {
            'max_length': self.max_length,
            'label_to_index': self.label_to_index,
            'index_to_label': {str(k): v for k, v in self.index_to_label.items()}
        }
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        
        # saves entity mappings
        entity_mappings = {
            'acronyms_to_entities': self.acronyms_to_entities
        }
        with open(os.path.join(output_dir, 'entity_mappings.json'), 'w') as f:
            json.dump(entity_mappings, f)
            
        print(f"Predictor saved to {output_dir}")
    
    @classmethod
    def load(cls, input_dir):
        """
        Loads a saved predictor.
        """
        model_path = os.path.join(input_dir, 'model.h5')
        tokenizer_path = os.path.join(input_dir, 'tokenizer.pickle')
        metadata_path = os.path.join(input_dir, 'metadata.json')
        entity_mappings_path = os.path.join(input_dir, 'entity_mappings.json')
        
        return cls(model_path, tokenizer_path, metadata_path, entity_mappings_path)

def batch_process_clinical_notes(file_paths, predictor):
    """
    Processes multiple clinical notes and compile statistics.
    """
    all_entities = []
    file_results = {}
    
    for file_path in file_paths:
        try:
            # reads the file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            # gets file name
            file_name = os.path.basename(file_path)
            print(f"Processing {file_name}...")
            
            # gets predictions
            results = predictor.predict(text)
            
            # extracts entities
            entity_list = results['entity_list']
            
            # adds to overall entity list
            all_entities.extend([e['entity_type'] for e in entity_list])
            
            # stores results for this file
            file_results[file_name] = {
                'entity_counts': results['entity_counts'],
                'total_entities': len(entity_list),
                'word_count': len(results['tokens'])
            }
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # compiles statistics
    overall_entity_counts = Counter(all_entities)
    total_entities = sum(overall_entity_counts.values())
    
    return {
        'file_results': file_results,
        'overall_entity_counts': dict(overall_entity_counts),
        'total_entities': total_entities
    }