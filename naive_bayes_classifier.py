
# NLTK imports
from nltk.corpus import genesis, stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import FreqDist, NaiveBayesClassifier

# SKLEARN imports
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

# OTHER imports
import matplotlib.pyplot as plt
from random import seed,shuffle
import string

def process_corpora(corpora_name: str,corpora_language: str):
    """
    Preprocess the corpora in order to:
        1 - Tokenize, remove punctuation, remove numbers, filter stopwords,
            stemmarize and lemmatize each word
        2 - Create a structure of sentences with filtered_word
    :param corpora_name: the name of the corpora
    :param corpora_language: the language in which the corpora had written
    :return: the corpora divided in sentences of tokens processed
    """

    # read the corpora
    corpora = genesis.raw(corpora_name)
    # split corpora in sentences
    sentences = sent_tokenize(corpora)
    # tokenize the sentences
    sentences_tokenized = [word_tokenize(sentence) for sentence in sentences]
    # create stopwords array
    stopwords_current = stopwords.words(corpora_language)

    senteces_filtered = []

    # iterate over each sentence
    for i in range(len(sentences_tokenized)):
        sentence = sentences_tokenized[i]
        new_sentence = []
        # iterate over each word
        for j in range(len(sentence)):
            word = sentence[j]
            # remove punctuation and numbers from words
            # and check if word is not in stopwords
            if (word not in string.punctuation) and (not word.isnumeric()) \
                    and (word.casefold() not in stopwords_current):
                # stemmarize every word
                word_stemmarized = stemmer.stem(word)
                # lemmatize every word
                word_lemmatized = lemmatizer.lemmatize(word_stemmarized)
                # add word processed to the new sentence
                new_sentence.append(word_lemmatized)
                # add word to all word
                all_words.append(word_lemmatized)
        # append each new sentence to the final return
        senteces_filtered.append(new_sentence)
    return senteces_filtered



def sentence_feature(sentence: list[str]):
    """
    Determines if the most 2000 common words are present in sentence
    :param sentence: the sentence to process
    :return: each words in the most 2000 words associated
            with True of False value, depending if they are present
            or not in sentence
    """
    # create a dictionary which indicates wether each word are
    # in the most 2000 common words
    return {f'contains({word})': (word in set(sentence)) for word in top_freqDist}

def labelization(corpora: list[str], label: int):
    """
    Return each sentence in each corpora processed and labelized

    :param corpora: list of sentences in corpora
    :param label: 1 for 'ENGLISH' or 0 for 'NON ENGLISH'
    :return: labelized sentences
    """


    # instantiate label array for each sentence
    label_list = [label for _ in range(len(corpora))]
    # create an array which contains each most 2000 common words presence
    sentence_featured = [sentence_feature(sentence) for sentence in corpora]
    # return all sentences labelized with 1 ('ENGLISH') or 0 ('NON ENGLISH')
    return list(zip(sentence_featured,label_list))

def split_train_test(set: list[str], perc_train: float):
    """
    Create train and test array given a percentage for train

    :param set: set of words with label
    :param perc_train: percentage of set to use for train array
    :return: two lists of words labeled one train and one test
    """

    index = int((perc_train/100) * (len(set)))

    return set[:index], set[index:]


if __name__ == "__main__":

    # language = ['english', 'finnish', 'french', 'german', 'portuguese', 'swedish']
    language = ['english','english','portuguese','swedish']
    corpora_name = [f'{language[0]}-kjv.txt', f'{language[1]}-web.txt',  f'{language[2]}.txt', f'{language[3]}.txt']

    # init stemmer object
    stemmer = PorterStemmer()
    # init lemmatizer object
    lemmatizer = WordNetLemmatizer()


    # all words that are in all corpus
    all_words = []
    # contains 4 arrays (each for every language)
    # and every array contains lists of sentences
    sentences_set = []
    # init the dataset
    dataset = []

    # iterate over corpus in order to preprocess data
    for i, value in enumerate(corpora_name):
        sentences_set.append(process_corpora(value,language[i]))

    # compute the frequency distribution of each word
    # encountered during preprocessing
    freq_dist = FreqDist(all_words)
    # select the top 2000 most common words
    top_freqDist = list(freq_dist)[:2000]

    # iterate over each corpora
    for i, corpora in enumerate(sentences_set):
        # the first 2 corpus are in 'ENGLISH' language
        if i < 2:
            # ENGLISH labelization
            dataset.extend(labelization(corpora,1))
        # the other 2 corpus are in 'NON ENGLISH' language
        else:
            # NON ENGLISH labelization
            dataset.extend(labelization(corpora,0))

    seed(1)
    # shuffle feature in order to randomize 'ENGLISH' and 'NON ENGLISH' words
    shuffle(dataset)
    # split dataset in train (70%) and test (30%)
    train_set, test_set = split_train_test(dataset, 70)
    # train the data
    classifier = NaiveBayesClassifier.train(train_set)

    # init real label
    label_real = []
    # init label classified
    label_predicted = []

    # iterate on sentences in test_set
    for i, (sentence, label) in enumerate(test_set):
        # save real label
        label_real.append(label)
        # save predicted label
        label_predicted.append(classifier.classify(sentence))

    # compute the confusion matrix
    conf_matrix = confusion_matrix(y_true=label_real, y_pred=label_predicted)

    # compute ACCURACY, PRECISION, RECALL and F1
    print('Accuracy: %.8f' % accuracy_score(label_real, label_predicted))
    print('Precision: %.8f' % precision_score(label_real, label_predicted))
    print('Recall: %.8f' % recall_score(label_real, label_predicted))
    print('F1: %.8f' % f1_score(label_real, label_predicted))

    # plot the confusion matrix
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Greens, alpha=0.3)
    # change ticks' name
    ax.set_xticks(ticks=[0, 1], labels=['NON ENGLISH', 'ENGLISH'])
    ax.set_yticks(ticks=[0, 1], labels=['NON ENGLISH', 'ENGLISH'], rotation=90, va='center')

    # add confusion matrix value to the plot
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    # set title and label for plotting
    plt.title('CONFUSION MATRIX', fontsize=25)
    plt.xlabel('Predict', fontsize=18)
    plt.ylabel('Real', fontsize=18)
    plt.show()
