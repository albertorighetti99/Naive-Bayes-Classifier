# Naïve Bayes Classifier NLTK implementation

### Goal
* * *
This program is an implementation of the Naïve Bayes Classifier created in order to distinguish ENGLISH from NON ENGLISH words.<br/>
First I decided to analyse **Genesis** corpora in *two different **english*** : World English Bible (2000) and King James Version (1611);
and *two different **non english** language*: Portuguese (Brazilian Portuguese version) and Swedish (1917, Gamla och Nya Testamentet).<br/>
The dataset has to be more balanced as possible therefore I tried to keep 50% of data in *english* and the remaining *non english*.

### Work on words
* * *
I extract sentences from each corpus and applied tokenization on it. Were removed the stopwords, including punctuation and numbers. Every word was stemmarized and
lemmatized too. After that, words were ricomposed in a new filtered sentence.


### Train and Test
* * *
 Was computed the frequency distribution of each word in every corpora and were stored the
first 2000 common words. Then every sentece previously created was elaborated in order to have a same input shape for each sentence. The elaboration consisted in transform the sentence in a dictionary where there are all 2000 commons words associated to True or False, depending on presence or absence of word in sentence.
Every of this dictionary was labelized with corresponding **'ENGLISH'** or **'NON ENGLISH'** label and appended to a dataset that was shuffled and splitted 70% in train and 30% in test.

### Performance indicators
* * *
The performance indicators compute were: <br/>
<img alt='accuracy' src='/img/accuracy.png' width='30%'/> 
 which indicates how many predictions were correct over all the predictions computed.
 <br/><br/>
<img alt='precision' src='/img/precision.png' width='30%'/> 
 that determines the relevant results.
<br/><br/>
<img alt='recall' src='/img/recall.png' width='30%'/> 
it establish the relevants results that are correctly classified.
<br/><br/>
<img alt='f1' src='/img/f1.png' width='30%'/> 
it is an harmonic average which uses the *precision* and *recall*.

<br/><br/>

All these formulas were computed using **sklearn.metrics** library.
<br/><br/><br/>
```
Accuracy: 0.99601990
Precision: 0.99285076
Recall: 1.00000000
F1: 0.99641256
```

*The confusion matrix produced.*
<img alt='confusion_matrix' src='/img/confusion_matrix.png'/> 


### Conclusion
This program reveals to be able to predict in a satisfactory way the both class. <br/>
Naive Bayes Classifier is now able to classify which is the probability of sentence to belong to an **'ENGLSH'** or to a **'NON ENGLISH'** class. <br/>
During training phase it learns which common words belong to each class and with testing it determines the probability, given a set of words (sentence), that this is in one class instead the other. <br/>
