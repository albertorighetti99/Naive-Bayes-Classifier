# Naïve Bayes Classifier NLTK implementation

## The experience

### Goal
* * *
This program is an implementation of the Naïve Bayes Classifier created in order to distinguish ENGLISH from NON ENGLISH words.<br/>
First I decided to analyse **Genesis** corpora in *two different **english*** : World English Bible (2000) and King James Version (1611);
and *two different **non english** language*: German (1534, Luther Translation ) and Swedish (1917, Gamla och Nya Testamentet).<br/>
The dataset has to be more balanced as possible therefore I tried to keep 50% of data in *english* and the remaining *non english*.

### Work on words
* * *
I extract each corpus and applied tokenization on it. Were removed the stopwords including punctuation and numbers. Every word was stemmarized,
labelizzed corresponding to the language it belongs and added to the dataset. <br/>

```python
words = word_tokenize(corpus)
words = [word for word in words if (word not in string.punctuation) and (not word.isnumeric())]
stopwords_current = stopwords.words(language[i])
filtered_words = filter_stop_words(words,stopwords_current)
```


### Train and Test
* * *
After that the dataset was shuffled to allow a fair dataset splitting between train and test. <br/> 
The function **nltk.NaiveBayesClassifier.train** was used to train the dataset, instead **nltk.NaiveBayesClassifier.test** was used to classify the test set. <br/>

```python
classifier = nltk.NaiveBayesClassifier.train(train_set)
predicted_label = classifier.classify(word)
```

### Performance indicators
* * *
The performance indicators compute were: <br/>
* **ACCURACY** $$ Accuracy = \dfrac {{TP}+{TN}} {{TP}+{TN}+{FP}+{FN}} $$ <br/>
 # this BELOW to check
 which indicates how many predictions were correct over all the real label
 # this ABOVE to check

* **PRECISION** $$ Precision = \dfrac {{TP}} {{TP}+{FP}} $$ <br/>
 that determines the correct true predictions over the true real label 
* **RECALL** $$ Recall = \dfrac {{TP}} {{TP}+{FN}} $$ <br/>
 whatever establish the correct true predictions over all the real label
<br/><br/>
All these were computed using **sklearn.metrics** library.
<br/><br/><br/>

*The confusion matrix produced.*
![confusion_matrix](confusion_matrix.png)

### Conclusion
This program reveals to be able to predict in a satisfactory way the *english* words,
while it seems to don't be so performant in *non english* words prediction.<br/>
Using a more wide set of corpus probably will improve the performance of the program. <br/>
But only wheter the balancement will be respected and to achieve this I suggest to use <br/>
the same corpus translated in more languages or to pay attention to use an enough mixed language content of words.

