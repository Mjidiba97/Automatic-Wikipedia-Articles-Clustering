# Set the model parameters

class PARAMS:
    
    # list of the languages of the randomly generated articles
    LANGUAGES = ["en", "fr", "es"]

    # dictionnary of the language codes and the full name of the languages
    LANG_DICT = dict({'fr':'french', 'en': 'english', 'es':'spanish'})
    
    # number of documents to generate per language
    ARTICLES_PER_LANG = 10
    
    # number of clusters to generate for each language group
    NCLUSTERS = dict({'fr':3, 'en': 3, 'es':3})
    
    # number of keywords per cluster
    NTERMS = 7