{\rtf1\ansi\ansicpg1252\cocoartf2706
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 HelveticaNeue;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\deftab560
\pard\pardeftab560\slleading20\pardirnatural\partightenfactor0

\f0\fs26 \cf0 # NLP Language Models Project\
\
## Overview\
\
Welcome to my Natural Language Processing (NLP) Language Models project! In this project, I'll be working with Language Models (LMs) to analyze a dataset. The main goal is to explore the capabilities of LMs and understand their performance in the context of NLP.\
\
## Dataset\
\
The dataset chosen for this project is the "Toxic Comment Dataset." It's important to note that this dataset contains toxic comments, so please exercise caution when reviewing the data. You can access both the training and test sets [here](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data).\
\
- **Training Set (train.csv):** This set will be used for training our Language Models.\
- **Test Set (test.csv) and Labels (test_labels.csv):** These files will be used for testing and analyzing our models. It's crucial to understand how the training and test sets are structured.\
\
## Language Model Tasks\
\
As part of this project, I'll be implementing two essential functions for Language Models:\
\
1. **train_LM(path_to_train_file):** This function will train a Language Model (up to a bigram model) using the provided training data file. The training data file should follow the same format as the training data.\
\
2. **test_LM(path_to_test_file, LM_model):** This function will test the Language Model on a test file. For each instance in the test file, it will assign a Maximum Likelihood Estimation (MLE) score for each test text. The function will generate an output file in the same format as the test file, including a new column with the MLE scores.\
\
## Project Tasks\
\
During this project, I'll complete the following tasks:\
\
1. Create three distinct Language Models:\
   - **LM_full:** Trained on the entire training dataset.\
   - **LM_not:** Trained on the training data with toxic labels set to 0.\
   - **LM_toxic:** Trained on the training data with toxic labels set to 1.\
\
2. Test each Language Model on the following:\
   - The full test set.\
   - A subset of data with non-toxic labels.\
   - A subset of data with toxic labels.\
\
3. Analyze and make observations based on the model performances. I'll calculate averages of scores for LM_full and compare them to LM_not to determine if the models are effectively capturing the language they were trained on.\
\
4. Document all observations and findings in a report document. Please refere to the word document\
\
Feel free to explore the code and report to see the insights and results of this NLP project!\
}