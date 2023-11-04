#We have five parts to this code:
#1. train_LM
#2. test_LM
#3. Exporting datasets(Please use this function if unable to download CSV files)
#4. LM_toxic
#5. LM_not
#6. LM_full
#Steps:
#1. Run the train_LM and test_LM
#2. Run the  export_datasets function if unable to download the csv files provided
#3. The CSV files will be input to the language models (LM_toxic, LM_not and LM_full) 
#4. While calling the LM functions please first enter local path locations
#5. The local path locations will be stored in variables which will be used to call the respective language models
#6. Please store the test, test_labels and train datasets in the same folder as this python file
#7. Otherwise enter the location of the local files manually in the respective places where location is required
import csv
import pandas as pd
from nltk.lm import MLE
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from nltk.util import pad_sequence
from nltk.lm.preprocessing import pad_both_ends
import csv
import nltk
from nltk.util import ngrams
from nltk.probability import MLEProbDist
#1. train_LM function
def train_LM(path_to_train_file):
#We initialise a list called comments. We require a list format to train our bigram model. 
#We take the csv file and extract the second column with index 1 from each row as a list of comments. The comments
#are split into individual words and stored in a list of list called comments
    comments = []
    with open(path_to_train_file) as file:
        reader = csv.reader(file)
        for row in reader:
            comment = row[1].split()
            comments.append(comment)
#We use the padded_everygram_pipeline() function from the nltk library(link shared by professor)
#to create a padded training set of bigrams from the comments
    train, vocab = padded_everygram_pipeline(2, comments)
#We craete a maximum likelihood estimation langauage model. This is of order 2 for bigrams.
#We then fit the model to the training data
    lm = MLE(2)
    lm.fit(train, vocab)
    return lm


#2. test_LM function
def test_LM(path_to_test_file, lm):
    test_data = pd.read_csv(path_to_test_file)
#We first assign MLE_score to 0
    MLE_scores = []
    for index, row in test_data.iterrows():
#We also initialise score and tolerances for smoothening out 0s and very small values
        score = 0.0
        tolerance = 1e-6
# split the string into bigrams
        bigrams = list(nltk.bigrams(row[1].split()))
# loop through the bigrams
        for bigram in bigrams:
# calculate the MLE score for the bigram
            lg_scr = lm.score(bigram[1], [bigram[0]])
            if lg_scr == 0:
#if lm.score is 0 we add the value as 0
                score += 0
            elif abs(lg_scr) < tolerance:
#if lm.score is less than the tolerance we add the value as a 0
                score += 0
            else:
                score += lg_scr
# append the MLE score to the list of scores
        MLE_scores.append(score)
    test_data['MLE_score_bigram'] = MLE_scores
    return test_data


#3. Exporting datasets
#Use the following function to get the csv files if unable to download the csv files I have shared
def export_datasets(train_file, test_file, test_labels_file):
# We are creating complete train dataset for creation of toxic train and non_toxic train datasets
    df_complete_train = pd.read_csv(train_file)
    df_complete_train = df_complete_train[['id','comment_text','toxic']]
    
# We are creating toxic and non-toxic train datasets
    df_toxic_train = df_complete_train[df_complete_train['toxic']==1]
    df_non_toxic_train = df_complete_train[df_complete_train['toxic']==0]
    
# We are create complete test dataset. We are joining the test dataset with the test labels dataset to get the 
#lables for toxic and non toxic
    df_test = pd.read_csv(test_file)
    df_test_labels = pd.read_csv(test_labels_file)
    df_test_complete = df_test.merge(df_test_labels, on='id')
    df_test_complete = df_test_complete[['id','comment_text','toxic']]
    
# We are creating toxic and non-toxic test datasets
    df_test_toxic = df_test_complete[df_test_complete['toxic']==1]
    df_test_non_toxic = df_test_complete[df_test_complete['toxic']==0]
    
# We are exporting datasets to local destination
    df_toxic_train.to_csv('toxic_train.csv', index=False)
    df_non_toxic_train.to_csv('non_toxic_train.csv', index=False)
    df_test_complete.to_csv('test_complete.csv', index=False)
    df_test_toxic.to_csv('test_toxic.csv', index=False)
    df_test_non_toxic.to_csv('test_non_toxic.csv', index=False)

#While calling the function I have kept my local destnation as refrence. Please change the local destination to get output
#We need 3 inputs here the train dataset, test dataset and test labels dataset from the kaggle link provided by the professor
#Below line of code is for user refrence only

export_datasets("train.csv", "test.csv", "test_labels.csv")

#Language models:

# How to use the below functions:
#The below functions will give the output for LM_toxic , LM_not and LM_full respectively
#Run the function
#I have shared the subsets of data which need to be inputed to these functions. The maupulations have been done by
#me to get toxic_train_path, test_complete_path,test_non_toxic_path,test_toxic_path, respectively.I have also named
#the files in the same naming convention as the path varible names.
#The only change which needs to be made is the local destination of the file . I have kept my local destination as it is 
#in my laptop so that it serves as a reference. Please make the required change to see the output
#The export_datasets function above can also be run to get the respective datasets on your local system if unable to download the csv files I have shared.

#4. LM_toxic
def LM_toxic(toxic_train_path, test_complete_path, test_toxic_path, test_non_toxic_path):
#We Train the language model
    lm = train_LM(toxic_train_path)
    
# We Test on the complete test set
    df_toxic_train_complete_test = test_LM(test_complete_path, lm)
    average_toxic_complete_MLE = df_toxic_train_complete_test['MLE_score_bigram'].mean()
    
# We Test on the toxic test set
    df_toxic_train_toxic_test = test_LM(test_toxic_path, lm)
    average_toxic_train_toxic_test_MLE = df_toxic_train_toxic_test['MLE_score_bigram'].mean()
    
# We Test on the non-toxic test set
    df_toxic_train_non_toxic_test = test_LM(test_non_toxic_path, lm)
    average_toxic_train_non_toxic_test_MLE = df_toxic_train_non_toxic_test['MLE_score_bigram'].mean()
    
# We Return the three dataframes and three values
# The three values are the average MLE scores tained on the bigram model
    return (df_toxic_train_complete_test, df_toxic_train_toxic_test, df_toxic_train_non_toxic_test,
            average_toxic_complete_MLE, average_toxic_train_toxic_test_MLE, average_toxic_train_non_toxic_test_MLE)

#Replace the location with users local location
#Below code is for user reference 
#If stored in the same location as python file then keep the below code same
toxic_train_path = "toxic_train.csv"
test_complete_path = "test_complete.csv"
test_toxic_path = "test_toxic.csv"
test_non_toxic_path = "test_non_toxic.csv"

#Calling the function LM_toxic
#Below code is for user reference only
(df_toxic_train_complete_test, df_toxic_train_toxic_test, df_toxic_train_non_toxic_test,
 average_toxic_complete_MLE, average_toxic_train_toxic_test_MLE, average_toxic_train_non_toxic_test_MLE) = LM_toxic(toxic_train_path, test_complete_path, test_toxic_path, test_non_toxic_path)
print(df_toxic_train_complete_test)
print(df_toxic_train_toxic_test)
print(df_toxic_train_non_toxic_test)
print(average_toxic_complete_MLE)
print(average_toxic_train_toxic_test_MLE)
print(average_toxic_train_non_toxic_test_MLE)


#5. LM_not
def LM_not(non_toxic_train_path, test_complete_path, test_toxic_path, test_non_toxic_path):
# We train the language model
    lm = train_LM(non_toxic_train_path)
    
# We test on the complete test set
    df_non_toxic_train_complete_test = test_LM(test_complete_path, lm)
    average_non_toxic_complete_MLE = df_non_toxic_train_complete_test['MLE_score_bigram'].mean()
    
# We test on the non-toxic test set
    df_non_toxic_train_non_toxic_test = test_LM(test_non_toxic_path, lm)
    average_non_toxic_train_non_toxic_test_MLE = df_non_toxic_train_non_toxic_test['MLE_score_bigram'].mean()
    
# We test on the toxic test set
    df_non_toxic_train_toxic_test = test_LM(test_toxic_path, lm)
    average_non_toxic_train_toxic_test_MLE = df_non_toxic_train_toxic_test['MLE_score_bigram'].mean()
    
# We Return the three dataframes and three values
# The three values are the average MLE scores tained on the bigram model
    return (df_non_toxic_train_complete_test, df_non_toxic_train_toxic_test, df_non_toxic_train_non_toxic_test,
            average_non_toxic_complete_MLE, average_non_toxic_train_toxic_test_MLE, average_non_toxic_train_non_toxic_test_MLE)
#Below code is for user reference only
#Replace the location with users local location
#If stored in the same location as python file then keep the below code same
non_toxic_train_path = "non_toxic_train.csv"
test_complete_path = "test_complete.csv"
test_toxic_path = "test_toxic.csv"
test_non_toxic_path = "test_non_toxic.csv"
#Below code is for user reference only
#Calling the function LM_not
(df_non_toxic_train_complete_test, df_non_toxic_train_toxic_test, df_non_toxic_train_non_toxic_test,
 average_non_toxic_complete_MLE, average_non_toxic_train_toxic_test_MLE, average_non_toxic_train_non_toxic_test_MLE) = LM_not(non_toxic_train_path, test_complete_path, test_toxic_path, test_non_toxic_path)
print(df_non_toxic_train_complete_test)
print(df_non_toxic_train_toxic_test)
print(df_non_toxic_train_non_toxic_test)
print(f"Average MLE LM_not for complete test{average_non_toxic_complete_MLE}")
print(f"Average MLE LM not for toxic test{average_non_toxic_train_toxic_test_MLE}")
print(f"Average MLE LM not for non toxic test{average_non_toxic_train_non_toxic_test_MLE}")

#6. LM_full
def LM_full(train_path, test_complete_path, test_toxic_path, test_non_toxic_path):
# We train the language model
    lm = train_LM(train_path)
    
# We test on the complete test set
    df_complete_train_complete_test = test_LM(test_complete_path, lm)
    average_complete_train_complete_test_MLE = df_complete_train_complete_test['MLE_score_bigram'].mean()

# We test on the toxic test set
    df_complete_train_toxic_test = test_LM(test_toxic_path, lm)
    average_complete_train_toxic_test_MLE = df_complete_train_toxic_test['MLE_score_bigram'].mean()

# We test on the non-toxic test set
    df_complete_train_non_toxic_test = test_LM(test_non_toxic_path, lm)
    average_complete_train_non_toxic_test_MLE = df_complete_train_non_toxic_test['MLE_score_bigram'].mean()

# We Return the three dataframes and three values
# The three values are the average MLE scores tained on the bigram model
    return (df_complete_train_complete_test, df_complete_train_toxic_test, df_complete_train_non_toxic_test, 
            average_complete_train_complete_test_MLE, average_complete_train_toxic_test_MLE, 
            average_complete_train_non_toxic_test_MLE)

#Below code is for user reference only
#Replace the location with users local location
#If stored in the same location as python file then keep the below code same
train_path = "train.csv"
test_complete_path = "test_complete.csv"
test_toxic_path = "test_toxic.csv"
test_non_toxic_path = "test_non_toxic.csv"

#Below code is for user reference only
#Calling the function LM_full
(df_complete_train_complete_test, df_complete_train_toxic_test, df_complete_train_non_toxic_test,
 average_complete_train_complete_test_MLE, average_complete_train_toxic_test_MLE, average_complete_train_non_toxic_test_MLE) = LM_full(train_path, test_complete_path, test_toxic_path, test_non_toxic_path)
print(df_complete_train_complete_test)
print(df_complete_train_toxic_test)
print(df_complete_train_non_toxic_test)
print(f"Average MLE LM full for complete test{average_complete_train_complete_test_MLE}")
print(f"Average MLE LM full for toxic test{average_complete_train_toxic_test_MLE}")
print(f"Average MLE LM full for non toxic test{average_complete_train_non_toxic_test_MLE}")

