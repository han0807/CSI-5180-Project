import sys
sys.path.append("../")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import utils

from sklearn.metrics.pairwise import cosine_similarity
import spacy

from spacy.lang.en import English
import en_core_web_md
nlp = en_core_web_md.load()
en = English()

# Loading the data
NEWS_STORIES = utils.open_pickle('../data/news_stories.pkl')
data = pd.read_csv('../data/newsqa-dataset-cleaned.csv')
total_examples = len(data)

def simple_tokenizer(doc, model=en):
    # a simple tokenizer for individual documents
    parsed = model(doc)
    return([t.lower_ for t in parsed if (t.is_alpha)&(not t.like_url)])

def get_doc_embedding(tokens, model = nlp):
    embeddings = []
    for t in tokens:
        embeddings.append(model.vocab[t].vector)
    
    embeddings = np.array(embeddings)
    if embeddings.ndim == 1:
        return embeddings
    else:
        return np.mean(embeddings, axis = 0)

def predict_answer(text,question):
    sentence_to_char_idx = [0]     
    sentences = []
    start_idx = 0
    
    for idx, char in enumerate(text):
        # If the chracter is a punctuation, we append the sentence
        if utils.is_punct(char):
            sentences.append(text[start_idx:idx])
            start_idx = idx + 1
            sentence_to_char_idx.append(start_idx)
    
    # Getting embeddings for each sentence
    sentence_embeddings = []
    for s in sentences:
        tokens = simple_tokenizer(s)
        embd = get_doc_embedding(tokens)
        if embd.shape == (300,):
            sentence_embeddings.append(embd)
    
    sentence_embeddings = np.stack(sentence_embeddings)
    
    # Getting the embedding for the question
    question_embedding = get_doc_embedding(simple_tokenizer(question))
    question_embedding = np.expand_dims(question_embedding, axis = 0)
    
    #print(sentence_embeddings.shape)
    # Get the cosine similarity of each sentence with the question
    similarity = cosine_similarity(sentence_embeddings, question_embedding)
    
    # Get the sentence with the most similarity
    best_idx = np.argmax(similarity)
    
    # Get the sentence start and end index
    pred_start = sentence_to_char_idx[best_idx]
    pred_end = sentence_to_char_idx[best_idx + 1] - 1
    
    return pred_start, pred_end

def calculate_metrics(pred_start, pred_end, true_start, true_end):
    '''
    Calculates the f1 score and if the predicted answer overlaps 
    with the correct one

    Parameters
    -----------
    pred_start, pred_end: int
                          The predicted start and end indices

    true_start, true_end: int
                          The actual indices
    '''
    # Get the overlap
    overlap = set(range(true_start, true_end)).intersection(range(pred_start, pred_end))
    overlap = len(overlap)

    # If either of them have no answer
    if true_end == 0 or pred_end == 0:
        f1_score = int(true_end == pred_end)
        is_correct = int(end_idx == pred_end)
        return f1_score, is_correct
    
    # If they don't overlap at all
    if overlap == 0 or pred_start >= pred_end:
        f1_score = 0
        is_correct = 0
        return f1_score, is_correct

    # If there is an overlap, we consider it correct
    is_correct = 1

    precision = overlap / (pred_end - pred_start)
    recall = overlap / (true_end - true_start)
    f1_score = (2 * precision * recall) / (precision + recall)

    return f1_score, is_correct

# Evaluate the performance of this approach on the data
correct = 0
total_f1 = 0

for idx, row in data.iterrows():
    text = NEWS_STORIES[row['story_id']]
    question = row['question']
    
    # Get the predictions
    pred_start, pred_end = predict_answer(text, question)
    f1, is_correct = calculate_metrics(pred_start, pred_end, row['start_idx'], row['end_idx'])
    
    total_f1 += f1
    correct += is_correct
    
    # Print progress
    utils.drawProgressBar(idx + 1, total_examples)
    
acc = correct/total_examples
f1_score = total_f1/total_examples

print("F1 score: {:.4f}".format(f1_score))
print("Accuracy: {:.4f}".format(acc))
