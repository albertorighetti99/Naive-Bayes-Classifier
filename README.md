# Naïve Bayes Classifier NLTK implementation

### Goal
* * *
This program is an implementation of the Naïve Bayes Classifier created in order to distinguish ENGLISH from NON ENGLISH words.<br/>
First I decided to analyse **Genesis** corpora in *two different **english*** : World English Bible (2000) and King James Version (1611);
and *two different **non english** language*: German (1534, Luther Translation ) and Swedish (1917, Gamla och Nya Testamentet).<br/>
The dataset has to be more balanced as possible therefore I tried to keep 50% of data in *english* and the remaining *non english*.

### Work on words
* * *
I extract sentences from each corpus and applied tokenization on it. Were removed the stopwords, including punctuation and numbers. Every word was stemmarized and
lemmatized too. After that, words were ricomposed in a new filtered sentence.


### Train and Test
* * *
 Was computed the frequency distribution of each word in every corpora and were stored the
first 2000 common words. Then every sentece previously created was elaborated in order to have a same input shape for each sentence. The elaboration consisted in transform the sentence in a dictionary where there are all 2000 commons words associated to True or False depending on presence or absence of word in sentence.
Every of this dictionary was labelized with corresponding **'ENGLISH'** or **'NON ENGLISH'** label and appended to a dataset that was shuffled and splitted 70% in train and 30% in test.

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

