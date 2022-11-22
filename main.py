import os
# Data manipulation libarys
import numpy as np
import pandas as pd
import csv
import re
# Models used to traning
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC

# Natural Language
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

porter = PorterStemmer()
lancaster=LancasterStemmer()

def clean_email(email, hdrs=True):

    # if ture remove headers from email
    if hdrs:
        hdrstart = email.find('\n\n')
        email = email[hdrstart:]
    email = email.lower()

    # Removes html tags
    rx = re.compile('<[^<>]+>|\n')
    email= rx.sub(' ', email)

    # makes any number, number
    rx = re.compile('[0-9]+')
    email = rx.sub('number ', email)

    # Convetrs urls to httpaddr
    rx = re.compile('(http|https)://[^\s]*')
    email = rx.sub('httpaddr ', email)

    # Converst any email into emailaddr
    rx = re.compile('[^\s]+@[^\s]+')
    email = rx.sub('emailaddr ', email)

    # $
    rx = re.compile('[$]+')
    email = rx.sub('dollar ', email)

    # Removes all non alpha numeric
    rx = re.compile('[^a-zA-Z0-9 ]')
    email = rx.sub('', email)

    #Tokenise Email
    token_words=word_tokenize(email)
    stem_sentence = []

    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

def vectorise_email(email, hdrs=True):

    # Load the dictionary
    d = {}
    with open('vocab.txt') as f:
        for line in f:
            (key, val) = line.split()
            d[key] = int(val)

    vector = np.zeros([1, len(d)], dtype=int)

    # Get the stemmed email
    email = clean_email(email, hdrs)

    # Vectorise the email
    for word in email.split():
        if word in d:
            vector[0, d[word]] = 1
    return vector

def create_vocab(path, length):
    vocab = {}
    file_list = os.listdir(path)
    # For every file log the stemmed words into vocab with the count
    for file in file_list:
        f = open(path+'/'+file, 'r')
        try:
            email = f.read()
            cleaned_email = clean_email(email)
            for word in cleaned_email.split():
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
            f.close()
        except UnicodeDecodeError:
            f.close()
    # Output the vocab of the desired length into a file
    f = open('vocab.txt', 'w')
    for i in range(length):
        highest_occurance = max(vocab, key=vocab.get)
        f.write(highest_occurance+' '+str(i)+'\n')
        del vocab[highest_occurance]
    f.close()

def create_data_file(paths):
    spam_paths = paths['spam']
    ham_paths = paths['ham']
    data_csv = open('vectorized_data.csv', 'w')
    writer = csv.writer(data_csv, delimiter=',')
    for spam_path in spam_paths:
        files = os.listdir(spam_path)
        for file in files:
            f = open(spam_path+'/'+file, 'r')
            try:
                email = f.read()
                data = vectorise_email(email)
                data = np.append(data, 1)
                writer.writerow(data)
                f.close()
            except UnicodeDecodeError:
                f.close()
    for ham_path in ham_paths:
        files = os.listdir(ham_path)
        for file in files:
            f = open(ham_path+'/'+file, 'r')
            try:
                email = f.read()
                data = vectorise_email(email)
                data = np.append(data, 0)
                writer.writerow(data)
                f.close()
            except UnicodeDecodeError:
                f.close()
    data_csv.close()

def find_model_svm():
    data = pd.read_csv('vectorized_data.csv', header=None)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

    svc = SVC(kernel='linear')
    svc.fit(X_train, y_train)

    y_pred = svc.predict(X_val)
    print("Validation Data Set")
    print(confusion_matrix(y_val,y_pred))
    print(classification_report(y_val,y_pred))
    print('\n\n\n')

    y_pred_test = svc.predict(X_test)
    print("Testset Data Set")
    print(confusion_matrix(y_test,y_pred_test))
    print(classification_report(y_test,y_pred_test))

    return svc


def predict_spam(svc, email, hdrs=False):
    email = vectorise_email(email, hdrs)
    email = pd.DataFrame(email)
    prediction = int(svc.predict(email))
    print("The email was predicted:", prediction, "(1 is spam, 0 is not spam)")
    return prediction

def main():
    path = 'Spam Data/spam' # Paths to spam emails
    create_vocab(path, 2000) # creates vocab list

    # Adding paths that have data we are using for the data set
    options = {'spam': ['Spam Data/spam'],
               'ham': ['Spam Data/easy_ham_2',
                       'Spam Data/hard_ham']}
    create_data_file(options)

    # Create a SVM instance
    svc = find_model_svm()

    with open('test_email.txt', 'r') as f:
        email = f.read()
        predict_spam(svc, email)


if __name__ == "__main__":
    main()