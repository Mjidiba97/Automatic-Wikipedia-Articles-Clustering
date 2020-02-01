# Dathena Hands-on NLP Technical Exercise
By: Ahmed Amine MAJDOUBI

The notebook covers the necessary documentation and an explanation of my approach to my submission for the NLP excercise. The goal of my submission was not to produce the best accuracy or implement complex NLP algorithms, but rather to show that my ability in writing a clear, robust and well documented code for data science related tasks. At the end of the notebook, I will discuss what could be done to further improve the segmentation task.

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
  
The parameters of the code are:
- <b>languages:</b> list of the languages of the randomly generated articles. 
- <b>lang_dict:</b> dictionnary of the language codes and the full name of the languages (necessary for NLTK library).
- <b>articles_per_lang:</b> number of documents to generate per language.
- <b>nclusters:</b> number of clusters to generate for each language group.

The default parameters of the file are 100 random Wikipedia paer language. The languages are English, French and Spanish. The number of clusters per language group is 3. The <b>'nclusters'</b> parameter can be later changed just after looking at the results of the elbow method generated with my function <b>optimal_clusters()</b>.
