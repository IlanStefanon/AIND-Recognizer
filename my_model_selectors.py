import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences

import copy



class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
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
        warnings.filterwarnings("ignore", category=RuntimeWarning)
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

    def selected_model(self, num_states, x, lengths):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(x, lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            logL = hmm_model.score(x, lengths)
            return hmm_model, logL
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
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # model selection based on BIC scores

        # number of states :
        #n = np.sum(self.lengths)
        n = len((self.lengths))
        logN = np.log(n)

        min_bic = 1000000
        best_hmm_model = None
        for nb_hidden_state in range(self.min_n_components, self.max_n_components + 1):
            # model, logL = self.base_model(nb_hidden_state)
            try:
                hmm_model = GaussianHMM(n_components=nb_hidden_state, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                logL = hmm_model.score(self.X, self.lengths)

                # number of free states
                p = n*n + 2 * nb_hidden_state * n - 1
                bic = -2 * logL + p * logN
                if bic < min_bic:
                    min_bic = bic
                    best_hmm_model = hmm_model
            except:
                continue
        return best_hmm_model

dict_logL = dict()
current_sequence = dict()

def is_same_dict(dict1, dict2):
    if not dict1 or not dict2:
        return False
    for k in dict1:
        if not k in dict2:
            return False
        else:
            for i,j in zip(dict1[k],dict2[k]):
                #print(i)
                for sub_i,sub_j in zip(i,j):
                    #print(sub_i, type(sub_i))
                    if not np.array_equal(sub_i, sub_j):
                        #print(sub_i, sub_j)
                        return False
                    #for sub_sub_i,sub_sub_j in zip(sub_i,sub_j) :
                    #    if sub_sub_i != sub_sub_j :
                    #        return False
                #if i != j:
                    #return False
            #if not np.array_equal(dict1[k], dict2[k]):
                #print(dict1[k], dict2[k])
            #if dict1[k] != dict2[k]:
                #return False
    return True

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)


    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        ModelSelector.__init__(self, all_word_sequences, all_word_Xlengths, this_word,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False)

        self.compute_logL()

    def compute_logL(self):
        global current_sequence
        if not dict_logL  or current_sequence:

            #if current_sequence and current_sequence == self.hwords:
            if is_same_dict(current_sequence, self.hwords):
                return
                #print(current_sequence, self.hwords)
            print('first generation of dict_logL')
            for word, value in self.hwords.items():
                dict_logL[word] = dict()
                X, lengths = value
                for nb_hidden_state in range(self.min_n_components, self.max_n_components + 1):
                    try:
                        hmm_model = GaussianHMM(n_components=nb_hidden_state, covariance_type="diag", n_iter=1000,
                                                random_state=self.random_state, verbose=False).fit(X, lengths)
                        logL = hmm_model.score(X, lengths)
                        dict_logL[word][nb_hidden_state] = logL
                        #print(word, nb_hidden_state, logL)
                    except:
                        continue
            current_sequence = copy.deepcopy(self.hwords)

    def average_not_word(self, not_this_word, n_components):
        average_logL = 0
        size = 0
        for word, value in dict_logL.items():
            #print('word', word)
            #print('value', value)
            if word != not_this_word:
                #if not_this_word == 'MARY':
                #    print(not_this_word, word, n_components)
                    #print(value[n_components])
                try:
                    if n_components in value:
                        average_logL += value[n_components]
                        #if not_this_word == 'MARY':
                        #    print('average_logL', average_logL)
                        size += 1
                except:
                    print('error raise by in value')
        if size > 0:
            average_logL /= size
        #print('average', average_logL)
        return average_logL

    def select(self):
        max_dic = -1000000
        dic = max_dic
        best_hidden_state = 0
        for nb_hidden_state in range(self.min_n_components, self.max_n_components + 1):
            if nb_hidden_state in dict_logL[self.this_word]:
                dic = dict_logL[self.this_word][nb_hidden_state] - self.average_not_word(self.this_word, nb_hidden_state)
            if dic > max_dic:
                max_dic = dic
                best_hidden_state = nb_hidden_state
        return self.base_model(best_hidden_state)

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):

        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # model selection using CV
        n_splits = min(3, len(self.lengths))
        max_mean_logL = -1000000
        best_hidden_state = 0
        for nb_hidden_state in range(self.min_n_components, self.max_n_components+1):
            #print('nb_hidden_state', nb_hidden_state)
            sum_log = 0
            n_splits_done = 0
            mean_logL = -1000000
            if n_splits == 1 :
                try :
                    hmm_model = GaussianHMM(n_components=nb_hidden_state, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                    mean_logL = hmm_model.score(self.X, self.lengths)
                except:
                    continue

            else :
                split_method = KFold(n_splits=n_splits)
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                    try:
                        model, logL =  self.selected_model(nb_hidden_state, X_train, lengths_train)
                        sum_log += logL
                        n_splits_done += 1
                    except:
                        continue
                if n_splits_done != 0 :
                    mean_logL = sum_log / float(n_splits_done)
                else:
                    mean_logL = -1000000
            #print('mean_logL', mean_logL)
            if(mean_logL> max_mean_logL):
                max_mean_logL = mean_logL
                best_hidden_state = nb_hidden_state
        #print('max_mean_logL', max_mean_logL)
        if best_hidden_state == 0:
            return self.base_model(self.n_constant)
        else:
            return self.base_model(best_hidden_state)