import sys
sys.path.append("../")

import numpy as np
import pandas as pd
import random
import utils
import re

# Reading the data
data = pd.read_csv('../data/newsqa-data-formatted.csv')
data.head()


# We don't need is_answer_absent and is_question_bad columns
data = data.drop(['is_answer_absent', 'is_question_bad'], axis = 1)

# Reading all stories as dictionary
STORY_PATH = '../cnn/stories/'
STORY_EXT = '.story'

story_ids = list(set(data['story_id']))
NEWS_STORIES = {}
cnt = 0
tot = len(story_ids)

for sid in story_ids:
    with open(STORY_PATH + sid + STORY_EXT, 'r') as file:
        NEWS_STORIES[sid] = file.read()
    cnt += 1
    utils.drawProgressBar(cnt, tot)



def adjust_answer_range(story_id,answer_range):
    # If answer is not available, denote it as -1
    if answer_range == 'None':
        return [-1, -1]
    
    story = NEWS_STORIES[story_id]
    
    # Check for errors in answer
    if len(answer_range.split(':')) == 1:
        return [-1, -1]
    
    start_idx, end_idx = answer_range.split(':')
    start_idx, end_idx = int(start_idx), int(end_idx)
    
    # Moves back start_idx to the start of a word
    while start_idx != 0 and not utils.is_whitespace(story[start_idx - 1]) and not utils.is_punct(story[start_idx - 1]):
        start_idx = start_idx - 1
    
    # Some ranges end with a punctuation or a whitespace
    if utils.is_whitespace(story[end_idx - 1]) or utils.is_punct(story[end_idx - 1]):
        end_idx = end_idx - 1
    
    # Moves end_idx to the end of a word
    while not utils.is_whitespace(story[end_idx]) and not utils.is_punct(story[end_idx + 1]):
        end_idx = end_idx + 1
        
    # There are some answers with \n at the end followed by a letter
    # The answer will not be in two different paragraphs
    answer_text = story[start_idx:end_idx]
    answer_para = re.split('\n', answer_text)
    
    if len(answer_para[-1]) > len(answer_para[0]):
        start_idx = end_idx - len(answer_para[-1])
        answer_text = answer_para[-1]
    else:
        end_idx = start_idx + len(answer_para[0])
        answer_text = answer_para[0]
    
    return [start_idx, end_idx]

def get_answer(qa_details):
     # If validated answers are available, select the one with most votes
    if qa_details['validated_answers'] is not np.nan:
        validated_answers = eval(qa_details['validated_answers'])
        
        # Get the answers with maximum votes
        max_vote_ans = utils.get_max_keys(validated_answers)
        
        # Check for ties
        if len(max_vote_ans) == 1:
            return adjust_answer_range(qa_details['story_id'], max_vote_ans[0])
    
    # If validated answers are not available or if there is a tie in validated answers
    # Get all available answers
    answers = re.split(',|\|', qa_details['answer_char_ranges'])
    
    # If there is just one answer
    if len(answers) == 1:
        return adjust_answer_range(qa_details['story_id'], answers[0])
    
    # Get counts of each answer
    answer_freq = utils.get_frequency(answers)
    max_vote_ans = utils.get_max_keys(answer_freq)
    
    if len(max_vote_ans) == 1:
        return adjust_answer_range(qa_details['story_id'], max_vote_ans[0])
    
    # If there is a tie for multiple answers, return a random answer
    return adjust_answer_range(qa_details['story_id'], random.choice(answers))


# Save the cleaned data
data.to_csv('../data/newsqa-dataset-cleaned.csv', index = False)
utils.save_pickle('../data/news_stories.pkl', NEWS_STORIES)
