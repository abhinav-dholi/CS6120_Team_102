import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM, Dense, 
    TimeDistributed, Dropout, SpatialDropout1D
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def precision(y_true, y_pred):
    """
    Precision metric for multi-class classification.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """
    Recall metric for multi-class classification.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1_score(y_true, y_pred):
    """
    F1 score for multi-class classification.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))

def build_bilstm_model(vocab_size, max_length, num_classes, embedding_dim=200, embedding_matrix=None):
    """
    Builds a BiLSTM model for NER.
    """
    model = Sequential()
    
    # embedding layer
    if embedding_matrix is not None:
        embedding_layer = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
            input_length=max_length,
            trainable=False
        )
    else:
        embedding_layer = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_length,
            mask_zero=True
        )
        
    model.add(embedding_layer)
    model.add(SpatialDropout1D(0.2))
    
    # BiLSTM layers
    model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
    
    # output layer
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(num_classes, activation='softmax')))
    
    # compiles the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', precision, recall, f1_score])
    
    return model

def create_callbacks(models_dir):
    """
    Creates callbacks for model training
    """
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(models_dir, 'best_model.h5'),
        monitor='val_f1_score',
        mode='max',
        save_best_only=True,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_f1_score',
        mode='max',
        patience=5,
        verbose=1,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_f1_score',
        mode='max',
        factor=0.5,
        patience=2,
        verbose=1,
        min_lr=0.0001
    )
    
    return [model_checkpoint, early_stopping, reduce_lr]

def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=20, callbacks=None):
    """
    Trains the model on the given data.
    """
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history, model

def save_model_with_metadata(model, tokenizer, label_to_index, index_to_label, max_length, vocab_size, 
                            num_classes, test_metrics, models_dir):
    """
    Saves the model and its metadata.
    """
    # saves model
    model.save(os.path.join(models_dir, 'ner_model.h5'))
    
    # saves model metadata
    model_metadata = {
        'max_length': max_length,
        'vocab_size': vocab_size,
        'num_classes': num_classes,
        'label_to_index': label_to_index,
        'index_to_label': {str(k): v for k, v in index_to_label.items()},
        'performance': {
            'test_accuracy': float(test_metrics[1]),  # accuracy is usually the second metric
            'test_precision': float(test_metrics[2]),  # precision is usually the third metric
            'test_recall': float(test_metrics[3]),  # recall is usually the fourth metric
            'test_f1': float(test_metrics[4])  # f1 is usually the fifth metric
        }
    }
    
    import json
    with open(os.path.join(models_dir, 'model_metadata.json'), 'w') as f:
        json.dump(model_metadata, f)
    
    # saves tokenizer
    import pickle
    with open(os.path.join(models_dir, 'tokenizer.pickle'), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Model and metadata saved to {models_dir}.")