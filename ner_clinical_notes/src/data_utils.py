import os
import re
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy
from spacy.tokens.doc import Doc
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from IPython.display import HTML, display

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy model...")
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

def remove_trailing_punctuation(token):
    """
    Removes trailing punctuation (ie. periods and commas) from each token.
    """
    while token and re.search(r'[^\w\s\']', token[-1]):
        token = token[:-1]
        
    return token

def split_text(text):
    """
    Splits note line by line, extracts tokens/characters and collects sentence break info.
    """
    regex_match = r'[^\s\u200a\-\u2010-\u2015\u2212\uff0d]+'
    
    tokens = []
    start_end_ranges = []
    sentence_breaks = []
    
    start_idx = 0
    
    for sentence in text.split('\n'):
        # finds words using regex & remove punctuation from them
        words = [match.group(0) for match in re.finditer(regex_match, sentence)]

        # tracks start & end positions of each token
        processed_words = list(map(remove_trailing_punctuation, words))
        sentence_indices = [(match.start(), match.start() + len(token)) for match, token in
                            zip(re.finditer(regex_match, sentence), processed_words)]
        sentence_indices = [(start_idx + start, start_idx + end) for start, end in sentence_indices]
        
        # appends current sentence's tokens & position
        start_end_ranges.extend(sentence_indices)
        tokens.extend(processed_words)
        sentence_breaks.append(len(tokens))
        start_idx += len(sentence) + 1

    return tokens, start_end_ranges, sentence_breaks

def clean_word(word):
    """
    Strips non-alphanumeric characters, converts to lowercase and lemmatizes using spaCy ("running" -> "run).
    """
    word = re.sub(r'[^\w\s]','',word)
    word = re.sub(r'\s+',' ',word)
    word = word.lower()
    lemma = nlp(word)[0].lemma_
    return lemma if lemma.strip() else word

def preprocess_text_for_prediction(text, tokenizer, max_length):
    """
    Converts tokens into padded sequences, applies tokenizer lookups and returns everything needed for
    NER to make predictions.
    """
    # prepares text to be fed into NER models
    tokens, token_ranges, sentence_breaks = split_text(text)
    cleaned_tokens = [clean_word(token) for token in tokens]
    
    # converts each token into integer index using tokenizer
    token_indices = []
    for token in cleaned_tokens:
        idx = tokenizer.word_index.get(token, 1)  # 1 is OOV
        token_indices.append(idx)
    
    # pads to match model's max input length
    X_padded = pad_sequences([token_indices], maxlen=max_length, padding='post', truncating='post')
    
    return tokens, token_indices, X_padded, token_ranges, sentence_breaks

def predict(text, model, index_to_label, acronyms_to_entities, max_length):
    """
    Predicts named entities in a given text.
    """
    # loads tokenizer
    with open('../data/models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    # preprocesses the text
    tokens, _, X_padded, token_ranges, _ = preprocess_text_for_prediction(text, tokenizer, max_length)
    
    # predicts label for token
    y_pred = model.predict(X_padded)[0]
    pred_indices = np.argmax(y_pred, axis=1)
    
    predicted_labels = []
    predicted_entities = []
    
    # translates indices into readable entity labels
    for i in range(min(len(tokens), max_length)):
        label = index_to_label[pred_indices[i]]
        predicted_labels.append(label)
        
        # extracts entity type from label (B-XXX or I-XXX)
        if label.startswith('B-') or label.startswith('I-'):
            entity_acronym = label[2:]  # removes B- or I- prefix
            entity_type = acronyms_to_entities.get(entity_acronym, entity_acronym)
            predicted_entities.append(entity_type)
        else:
            predicted_entities.append('O')
    
    # prints predictions
    print("Predicted Named Entities:")
    for token, entity in zip(tokens[:len(predicted_entities)], predicted_entities):
        print(f"{token}: {entity}")
        
    # prepare entities
    ents = []
    last_entity = None
    start_char = None
    
    for i, (token, entity, (start, end)) in enumerate(zip(tokens, predicted_entities, token_ranges)):
        if entity != 'O':
            if entity != last_entity:
                # if we have stored an entity, adds it to our entities list
                if last_entity is not None and start_char is not None:
                    ents.append({
                        "start": start_char,
                        "end": token_ranges[i-1][1],
                        "label": last_entity
                    })
                # starts a new entity
                start_char = start
                last_entity = entity
        else:
            # if we have stored an entity, adds it to our entities list
            if last_entity is not None and start_char is not None:
                ents.append({
                    "start": start_char,
                    "end": token_ranges[i-1][1],
                    "label": last_entity
                })
                start_char = None
                last_entity = None
    
    # adds the last entity if there is one
    if last_entity is not None and start_char is not None:
        ents.append({
            "start": start_char,
            "end": token_ranges[len(predicted_entities)-1][1],
            "label": last_entity
        })
        
    # displays the entities
    return tokens, predicted_labels, predicted_entities, token_ranges

def load_data(data_dir):
    """
    Loads processed data for training, validation, and testing.
    """
    # loads processed data
    data = np.load(os.path.join(data_dir, 'processed', 'processed_data.npz'))
    
    train_sequences_padded = data['X_train']
    train_labels = data['y_train']
    val_sequences_padded = data['X_val']
    val_labels = data['y_val']
    test_sequences_padded = data['X_test']
    test_labels = data['y_test']
    
    # loads label mappings
    with open(os.path.join(data_dir, 'processed', 'label_mappings.json'), 'r') as f:
        import json
        label_mappings = json.load(f)
    
    label_to_index = label_mappings["label_to_index"]
    index_to_label = {int(k): v for k, v in label_mappings["index_to_label"].items()}
    
    return (train_sequences_padded, train_labels), (val_sequences_padded, val_labels), (test_sequences_padded, test_labels), label_to_index, index_to_label

def parse_bio_files(bio_files):
    """
    Parse multiple BIO files and extract sentences and labels.
    """
    all_sentences = []
    all_labels = []
    
    for idx, bio_file in enumerate(bio_files):
        try:
            sentences, labels = parse_data_from_file(bio_file)
            
            if sentences:  # only add if not empty
                all_sentences.extend(sentences)
                all_labels.extend(labels)
                
            if (idx+1) % 20 == 0:
                print(f'{idx+1} files processed')
                
        except Exception as e:
            print(f"Error processing {bio_file}: {e}")

    return all_sentences, all_labels

def parse_data_from_file(bio_file):
    """
    Reads a file in BIO format and extracts sentences and labels.
    """
    sentences = []
    labels = []
    
    with open(bio_file, "r", encoding='utf-8') as f:
        
        current_sentence = []
        current_labels = []
        
        for line in f:
            line = line.strip()
            
            if line == '':
                # if we encounter a blank line, it means we've reached the end of a sentence
                if current_sentence:  # only add if not empty
                    sentences.append(current_sentence)
                    labels.append(current_labels)
                    current_sentence = []
                    current_labels = []
                continue
                    
            parts = line.split('\t')
            if len(parts) >= 2:  # make sure we have both token and tag
                word, tag = parts
                
                # cleans the word
                word = clean_word(word)
                
                if word.strip():
                    current_sentence.append(word)
                    
                    # make sures consistent BIO tagging
                    if current_labels and tag.startswith("B-") and tag[2:] == current_labels[-1][2:] and len(current_sentence) > 1:
                        tag = f"I-{tag[2:]}"
                        
                    current_labels.append(tag)
        
        # adds the last sentence if it's not empty
        if current_sentence:
            sentences.append(current_sentence)
            labels.append(current_labels)
                
    return sentences, labels