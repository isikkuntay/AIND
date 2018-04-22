import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    for word_id in test_set.get_all_sequences().keys():
        proba_dict = {}   #empty dict
        best_log = float("-inf")   #initialize
        best_guess = ""
        X, lengths = test_set.get_item_Xlengths(word_id)
        for someword in models.keys():
            hmm_model = models[someword]
            try:
                logL = hmm_model.score(X, lengths)
                proba_dict[someword] = logL
                if logL > best_log:
                    best_guess = someword
                    best_log = logL
            except:
                pass
        probabilities.append(proba_dict)        
        guesses.append(best_guess)
    
    return probabilities, guesses
    
