import nltk
import sys
import os
import string
import math
import operator

"""
Shuyan Liu
CS50's Intro to AI
08/08/2020
"""

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
    fileDict = dict()
    files = [f for f in os.listdir(directory)]
    for fileName in files:
        f = open(os.path.join(directory, fileName), "r", encoding="utf8")
        fileDict.update({fileName: f.read()})
        f.close()
    return fileDict


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words = nltk.word_tokenize(document)
    for i in range(len(words)-1, -1, -1):
        words[i] = words[i].lower()
        if words[i] in string.punctuation or words[i] in nltk.corpus.stopwords.words("english"):
            words.remove(words[i])
    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # Copied from the source code (tfidf.py)
    # Get all words in corpus
    words = set()
    for filename in documents:
        words.update(documents[filename])
    # Calculate IDFs
    idfs = dict()
    for word in words:
        f = sum(word in documents[filename] for filename in documents)
        idf = math.log(len(documents) / f)
        idfs[word] = idf
    return idfs
                        

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tfidfs = []
    total = 0
    for f in files:
        for word in query:
            frequency = files[f].count(word)
            if idfs[word] != 0:
                total += frequency / idfs[word]
        tfidfs.append((f, total))
        total = 0
    # Sort
    tfidfs.sort(key=lambda tfidf: tfidf[1], reverse=True)
    # Keep only the top n items
    tfidfs = tfidfs[:n]
    # Remove the idf value
    for i in range(len(tfidfs)):
        tfidfs[i] = tfidfs[i][0]
    return tfidfs


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    topSentences = []
    total = 0
    count = 0
    for s in sentences:
        for word in query:
            if word in sentences[s]:
                total += idfs[word]
                count += sentences[s].count(word)
        qtd = count / len(sentences[s])
        topSentences.append((s, total, qtd))
        total = 0
        count = 0
    topSentences.sort(key = operator.itemgetter(1, 2), reverse=True)
    topSentences = topSentences[:n]
    for i in range(len(topSentences)):
        topSentences[i] = topSentences[i][0]
    return topSentences


if __name__ == "__main__":
    main()
