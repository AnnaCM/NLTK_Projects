NLTK projects will be stored in this repository.

The first project is an email-spam-filter.
It is implemented by counting the most frequent words (list markers, modal verbs, adjectives, adverbs and verbs) in all the emails, excluding the stopwords.
Different classifiers have, then, been trained and their accuracies have been calculated.
Next, the user is asked to enter an email to be classified.
In 'classification_method.py', every classifier classifies the email, based on the features, and the mode is returned as the input's classification.

The data sets used for this project can be found in the 'Enron-Spam' folder, or via the following link: http://www2.aueb.gr/users/ion/data/enron-spam/

To avoid training the classifiers every time I need, I have created the 'pickled_files' folder, comprised of all the serialized classifier objects,
that allows me to save time and quickly read the files. To do it, you need to run 'save_objects.py' just one time.

The project is written in Python, using the Natural Language Processing methodology.
So, all you need to do is installing the Natural Language Toolkit, or NLTK (you can do it with pip),
and some of its components, typing 'import nltk' and 'nltk.download()' on your command-line shell.
Then, it should appear a window with red lines: choose to download "all" for all packages, and click "download".
You also want to install scipy (again, using pip, if you have never done so) to train the classifiers, other than the Naive Bayse Algorithm.
