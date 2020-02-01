# NLP: Clustering random Wikipedia articles
By: Ahmed Amine MAJDOUBI

Note: It takes about 5 mins to generate 300 Wikipedia articles. Keep that in mind when playing with the parameters.

The notebook covers the necessary documentation and an explanation of my approach to my submission for the NLP excercise. The goal of my submission was not to produce the best accuracy or implement complex NLP algorithms, but rather to show that my ability in writing a clear, robust and well documented code for data science related tasks. At the end of the notebook, I discussed what could have been done to further improve the segmentation task.

The code functions are divided into four main types:
- <b>Document Extraction:</b> generate random articles of different languages from Wikipedia.
- <b>Language Detection:</b> detect the languages on the extracted documents, and group them by their language.
- <b>Text Processing:</b> process each document by tokenizing, stemming and removing stop words.
- <b>Document Clustering:</b> cluster each language group of document using TF-IDF and K-MEANS.

For this submission, I used the following third party libraries:
- <a href ='https://pypi.org/project/wikipedia/'>wikipedia</a>: Wikipedia API for python to access and parse data from Wikipedia.
- <a href ='https://pypi.org/project/langdetect/'>langdetect</a>: Language detection library ported from Google's language-detection. Supports 55 languages.
- <a href ='https://www.nltk.org/'>nltk</a>: Provides the necessary tools for symbolic and statistical natural language processing.

You can install these libraries using pip by uncommenting the following code:
  - pip install wikipedia
  - pip install langdetect
  - pip install nltk
  
The parameters in the config file are:
- <b>LANGUAGES:</b> list of the languages of the randomly generated articles
- <b>LANG_DICT:</b> dictionnary of the language codes and the full name of the languages (necessary for NLTK library).
- <b>ARTICLES_PER_LANG:</b> number of documents to generate per language.
- <b>NCLUSTERS:</b> number of clusters to generate for each language group.
- <b>NTERMS:</b> number of keywords to generate for each cluster.

The default parameters of the file are 100 random Wikipedia paer language. The languages are English, French and Spanish. The number of clusters per language group is 3, and the number of keywords to generate is 7.
