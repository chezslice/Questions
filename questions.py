import os
import string
import math

import nltk
import sys


FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    # Creating a dictionary contents variable set to dict().
    dictionary_contents = dict()

    # For loop, to interate over file in the files and opening a path.
    for i, j, files in os.walk(directory):
        for file in files:
            f_open = open(os.path.join(i, file), 'r')
            # Contents of the files will equal to f_open variable.
            dictionary_contents[file] = f_open.read()

    return dictionary_contents

def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # Assigning variable with string of characters.
    punctuation = string.punctuation
    # Assigning variable to the english corpus y using a stop function.
    stop_function = nltk.corpus.stopwords.words("english")

    # Tokenize the document and lowercase the letters. 
    # Internate word by word thru the punctuation and stop_function variables.
    words_token = nltk.word_tokenize(document.lower())
    words_token = [word for word in words_token if word not in punctuation and word not in stop_function]

    return words_token

def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # Assigning idfs variable to the dictionary set.
    idfs_values = dict()
    # Number of documents are assigned by the len of the documents.
    num_of_documents = len(documents)

    # Create a words variable and assign sublist into the documents file.
    words = set(word for sublist in documents.values() for word in sublist)

    # For loop, to interate over the word in words and create a counter variable.
    for word in words:
        word_counter = 0
        # Internate over the documents and increase the counter variable if the word is found in the dictionary.
        for document in documents.values():
            if word in document:
                word_counter += 1

        # Idf variables are assigned to their apporiate variables.
        # Math.log is used to use the formula to divide and get idf results.
        idf_value = math.log(num_of_documents / word_counter)
        idfs_values = idf_value

    return idfs_values

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # Create a dictionary variable.
    scores = dict()

    # Internate thru the file and words inside the files.
    for files, words in files.items():
        # Create a total counter of tf idfs.
        total = 0
        # Interate word in the query and increase total counter using the formula.
        for word in query:
            total += word.count(word) * idfs[word]
        scores[file] = total

    # Return a sorted list of ranked_idfs.
    ranked_idfs = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)

    ranked_idfs = [x[0] for x in ranked_idfs]

    return ranked_idfs[:n]
    
def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # Create a dictionary variable.
    scores = dict()

    for sentence, words in sentence.items():
        query_words = query.intersection(words)

        # Value of the idf inside the sentence.
        idf_value = 0
        
        # Interate thru word inside the query.
        for word in query_words:
            idf_value += idfs[word]

        # Density of the sentence based on the query.
        query_numbers = sum(map(lambda x: x in query_words, words))
        density = query_numbers / len(words)

        # Sentence is then update with the idf and query density.
        scores[sentence] = {'idf': idf, 'qtd': density}

    # Return a sorted list of ranked_idfs inside the sentences.
    ranked_idfs_sentence = sorted(scores.items(), key=lambda x: (x[1]['idf'], x[1]['qtd']), reverse=True)
    ranked_idfs_sentence = [x[0] for x in ranked_idfs_sentence]

    return ranked_idfs_sentence[:n]

if __name__ == "__main__":
    main()

