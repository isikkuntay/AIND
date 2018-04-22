import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                n_constant=3, min_n_components=2, max_n_components=10,
                    random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value
        
        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        BIC_min = np.float('inf') #initialize
        BIC_calc = np.float('inf') #initialize
        best_num_components = 0 #Initialize
        for n in range(self.min_n_components, self.max_n_components):
            BIC_model = self.base_model(n)
            try:
                BIC_logL = BIC_model.score(self.X, self.lengths)
                BIC_calc = -2 * BIC_logL + (n**2+2*n*len(self.X[0])-1)* np.log(sum(self.lengths))
            except:
                pass
            if BIC_calc < BIC_min:
                best_num_components = n
                BIC_min = BIC_calc
        return self.base_model(best_num_components)

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        
        DIC_max = float("-inf")    #initialize
        best_num_components = 0    #initialize
        for n in range(self.min_n_components, self.max_n_components):

            M = len(self.words)
            DIC_calc = 0    #initialize
            this_logL = 0   #initialize
            DIC_model = self.base_model(n)
            try:
                this_logL = DIC_model.score(self.X, self.lengths)

            except:
                pass

            total_logL = 0   #initialize

            for other_word in self.words.keys():
                if other_word != self.this_word:

                    other_X, other_Len = self.hwords[other_word]
                        
                    try:
                        other_logL = DIC_model.score(other_X, other_Len)
                        total_logL += other_logL
                    except:
                        M -= 1

            if total_logL != 0:  #we want the net effect of total_logL to be negative 
                DIC_calc = this_logL - 1/(M - 1)*abs(total_logL)

            if DIC_calc > DIC_max and this_logL != 0:
                DIC_max = DIC_calc
                best_num_components = n
                        
        return self.base_model(best_num_components)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_num_components = 0 #initialize
        split_method = KFold()
        test_scores ={}
        
        for n in range(self.min_n_components, self.max_n_components):
            test_scores[n] = 0 #Initialize

        word_sequences = self.sequences
            
        if len(self.sequences) < 3:
            return self.base_model(self.n_constant) #If there aren't enough samples for CV, return arbitrary number of states (3)

        log_count = {}  #Since the main loop is over combinations of data, we need to keep track of those that can return a score.
        for cv_train_idx, cv_test_idx in split_method.split(word_sequences):
            #first concatenate into single X and lengths; create another dictionary
            trainingX, trainingLen = combine_sequences(cv_train_idx, word_sequences)
            testingX, testingLen = combine_sequences(cv_test_idx, word_sequences)
            for n in range(self.min_n_components, self.max_n_components):
                log_count[n] = 0
                cv_model = GaussianHMM(n, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(trainingX, trainingLen)
                try:
                    #Improvement opportunity: The log_count could be eliminated if the n-loop were outside the split-loop.
                    log_count[n] += 1
                    cv_logL = cv_model.score(testingX, testingLen)
                    test_scores[n] = test_scores[n] + cv_logL
                except:
                    pass
        #Correction made on the following loop: in previous version it was supposed to be outside the scoring loop. 
        for n in range(self.min_n_components, self.max_n_components):
            test_scores[n] = test_scores[n] / log_count[n] 
        best_num_components = max(test_scores, key = test_scores.get)
        return self.base_model(best_num_components)
