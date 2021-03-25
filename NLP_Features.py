from __future__ import print_function

import re
import os
import pandas as pd
import numpy as np
import spacy
from spacy import displacy
import nltk
from pathlib import Path
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from collections import OrderedDict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from plotly.offline import plot
import plotly.graph_objs as go
from textblob_de import TextBlobDE
from stop_words import get_stop_words
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
# import de_core_news_sm


# ===========================================================================================================================

class NLP_Features():

    def __init__(self):

        self.df = pd.DataFrame()
        self.stemmer = SnowballStemmer("german")

    # ============================================================================================================#
    ''' 
    Functions to extract nlp features from the requirements 

    '''

    """ selects the number of sentences in one requirement and takes the greater number from nltk or spacy """
    def select_sentences(self, row, nb="n"):
        # check if nltk detects less sentences than spacy
        if (row['sentence_nb_by_nltk'] <= row['sentence_nb_by_nlp']):
            if nb == "y":
                return row['sentence_nb_by_nltk']
            else:
                return row['sentences_by_nltk']
        # if the module nltk detects more sentences than spacy module
        else:
            if nb == "y":
                return row['sentence_nb_by_nlp']
            else:
                return row['sentences_by_nlp']

    """ POS Tagging of requirements """
    def tag_sentence(self, nlp, sentence):
        return [(w.text, w.tag_, w.pos_) for w in nlp(sentence)]

    """ function that returns the number of syllables per words. """

    # def _syllables(word):
    #      syllable_count = 0
    #      vowels = 'aeiouy'
    #      if word[0] in vowels:
    #           syllable_count += 1
    #      for index in range(1, len(word)):
    #           if word[index] in vowels and word[index - 1] not in vowels:
    #                syllable_count += 1
    #      if word.endswith('e'):
    #           syllable_count -= 1
    #      if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
    #           syllable_count += 1
    #      if syllable_count == 0:
    #           syllable_count += 1

    #      return syllable_count

    def compute_SPW(self, text):
        syllable = 0
        vowels = ['a', 'e', 'i', 'o', 'u', 'y', 'ä', 'ü', 'ö', '-']
        diphtong = ['eu', 'au', 'ei', 'äu', 'io', 'ai', 'oi', 'ui']
        for word in text.split():
            count = 0
        for vowels in ['a', 'e', 'i', 'o', 'y', 'ä', 'ü', 'ö', '-']:
            count += word.count(vowels)
        for diphtong in ['eu', 'au', 'ei', 'äu']:
            count -= word.count(diphtong)
        if count == 0:
            count += 1
            syllable += count

        # calculate the syllable mean (total syllable divided by number of words in the sentence)
        return syllable / (len(text.split()) * 1.0)

    def count_punctuation(self, attribute):
        i = 0
        for w in range(len(attribute)):
            # if it is tagged with "$(" (<=> punctuation tag)
            if attribute[w][1] == "$(":
                i += 1
        return i

    def count_comma(self, attribute):
        i = 0
        for w in range(len(attribute)):
            if attribute[w][1] == "$,":
                i += 1
        return i

    def count_weird_words(self, attribute):
        i = 0
        for w in range(len(attribute)):
            if attribute[w][2] == "X":
                i += 1
        return i

    """ Search for specific words in the requirements. When word is beispiel or circa, take abbreviation into account """
    def search_specific_words(self, attribute, search_word):
        if search_word == "beispiel":
            word = ["z.b.", "zb", "zum beispiel", "zb."]
        elif search_word == "circa":
            word = ["circa", "ca", "ca."]
        else:
            word = [search_word.lower()]
        i = 0
        for w in range(len(attribute)):
            if attribute[w][0].lower() in word:
                i += 1
        return i

    """ function that checks the presence of "und" or "oder" in the requirement (double counting with CONJ?)"""
    def count_Copulative_Disjunctive_terms(self, requirement):
        disj_list = ["und", "oder"]
        i = 0
        for w in requirement.split():
            if w.lower() in disj_list:
                i += 1
        return i

    """ Checks for the presence of "minimum" or "maximum" and their derivated forms"""
    def check_max_min_presence(self, requirement):
        # list of derivated forms of minimum and maximum
        maxmin_list = ["max", "maximum", "min", "minimum", "max.", "min.", "min-/max", "maximal", "maximale",
                       "maximalen", "maximaler", "minimal", "minimale", "minimalen", "minimaler"]
        presence = "no"
        for w in requirement.split():
            if w.lower() in maxmin_list:
                presence = "yes"

        return presence

    """ function that check if some logical conjunction are present in the requirement """
    def time_logical_conj(self, attribute):
        # list of logical conjunction taken into account
        conj_list = ["während", "sobald", "bis", "innerhalb", "bei", "wenn", "gemäß", "falls", "bzw."]
        i = 0
        for w in range(len(attribute)):
            if attribute[w][0].lower() in conj_list:
                i += 1
        return i

    """ function that check if measurement indicators are present in the requirement """
    def search_measurements_indicators(self, attribute):

        scale_1 = ["sec", "sekund", "stunde", "h", "minut", "Grad", "%", "n", "km", "km/h", "pa", "rad/sec", "/s",
                   "°/s", "cm", "m/s", "m/s^2"]
        # scale_2 = ["sec","sekund","stunde","h","minut","min","Grad","%","n","km","km/h","pa","rad/sec","/s","°/s","cm","m/s","m/s^2"]
        scale_2 = ["%", "rad/sec", "/s", "°/s", "cm", "m/s", "m/s^2"]
        presence = False
        for w in range(len(attribute)):
            if attribute[w][0].lower() in scale_1:
                presence = True
        for s in scale_2:
            if s in attribute[w][0].lower():
                presence = True

        if presence:
            return "yes"
        else:
            return "no"

    def search_numerical_value(self, attribute):
        i = 0
        for w in range(len(attribute)):
            if attribute[w][2] == "NUM":
                i += 1
        return i

    """ Function to determine if a sentence is written in passive form
         # if at least one past participe and one auxiliary ("werden", "wird", "worden" "wurde") = passive sentence
         attribute corresponds to the requirements processed by nlp = (word_, tag_, pos_)
    """
    def passive_detection(self, attribute):
        answer = "no"
        # suppose first there is no past participe
        PP = False
        werden_forms = ["werden", "wird", "worden", "wurde"]
        for w in range(len(attribute)):
            if attribute[w][1] == "VVPP":
                PP = True
        for w in range(len(attribute)):
            # for each word, look at the ones that belongs to the werden_forms list and that have a "AUX" position
            if (attribute[w][2] == "AUX" and (attribute[w][0] in werden_forms) and PP):
                # if an AUX werden and a past participe...then passive form
                answer = "yes"
        return answer

    """ checks if the first word of the requirement is an auxiliary """
    def aux_1st(self, attribute):
        if attribute[0][2] == "AUX":
            return "yes"
        else:
            return "no"

    """ function that counts how many subordinate conjunctions: 
        (e.g. after, although, as, because, before, even if, even though, if, in order that, once, provided that, 
        rather than, since, so that, than, that, though, unless, until, when, whenever, where, whereas, wherever, 
        whether, while, why) are present in the requirement 
    """
    def count_subordinate_conjunction(self, attribute):
        nb_sc = sum(attribute[x][2] == "SCONJ" for x in range(len(attribute)))
        return nb_sc

    """ function that counts how many coordination conjunctions or comparison conjunctions 
        (und, oder, als, bzw., bis, oder) are present in the requirement
    """
    def count_comp_coor_conjunction(self, attribute):
        nb_cc = sum(attribute[x][2] == "CONJ" for x in range(len(attribute)))
        return nb_cc

    """ counts how many verbs are present in requirement """
    def count_verb(self, attribute):
        nb_vb = sum(attribute[x][2] == "VERB" for x in range(len(attribute)))
        return nb_vb

    """ function that counts how many auxiliaries are present in the requirement """
    def count_aux(self, attribute):
        nb_aux = sum(attribute[x][2] == "AUX" for x in range(len(attribute)))
        return nb_aux

    """ function that counts how many times the word "werden" or its conjugate forms appear in a requirement """
    def count_werden(self, text):
        count = 0
        if re.search("wird", text, re.I):
            count += len(re.findall("wird", text, re.I))
        if re.search("werden", text, re.I):
            count += len(re.findall("werden", text, re.I))
        return count

    """ Finds requirements containing mussen, darfen """
    def contain_Muss_Darf_nicht(self, ps, attribute):
        # possible forms for müssen
        muessen = ['muss', 'musst']
        # possible forms for dürfen
        duerfen = ['darf', 'durf', 'durft']
        # stem each word in requirement
        tokens = [ps.stem(w) for w in attribute.split()]
        presence = "no"
        for i in range(len(tokens) - 1):
            if ((tokens[i] in muessen) or (tokens[i] in duerfen) or (tokens[i] in duerfen and (
                    tokens[i + 1] == "nicht" or tokens[i + 1] == "maximal" or tokens[i + 1] == "höchstens"))):
                presence = "yes"
                return presence

    ''' Finds named entities, phrases and concepts '''
    def entities_label(self, text):
        if len(text.ents) != 0:
            return [(x.text, x.label_) for x in text.ents]
        else:
            return text.ents

    # ============================================================================================================== #

    '''
    A function to extract features from requirements and create a dataframe
    '''

    def extract_features(self, Req_list, score_target, export=True, corpal=True):

        nlp = spacy.load('de')
        stemmer = SnowballStemmer("german")
        stop = stopwords.words('german')
        features = pd.DataFrame()
        # create first column of dataframe by allocating requirement list to it; one requirement per line
        features['req'] = Req_list
        # get text, tag_ and pos_ attributes for each word
        features['req_nlp'] = features['req'].apply(lambda x: nlp(x))
        features['tags'] = features['req_nlp'].apply(lambda x: [(w.text, w.tag_, w.pos_) for w in x])

        # Analysis using NLTK
        # Split sentences then count number in each requirement
        features['sentences_by_nltk'] = features['req'].apply(lambda x: nltk.sent_tokenize(x, 'german'))
        features['sentence_nb_by_nltk'] = features['req'].apply(lambda x: len(nltk.sent_tokenize(x, 'german')))
        # analysis with spacy
        features['sentences_by_nlp'] = features['req_nlp'].apply(lambda x: [sent.string.strip() for sent in x.sents])
        features['sentence_nb_by_nlp'] = features['req_nlp'].apply(
            lambda x: len([sent.string.strip() for sent in x.sents]))

        # number of sentences per requirement
        features['sentences'] = features.apply(lambda x: self.select_sentences(x), axis=1)
        features['sentences_nb'] = features.apply(lambda x: self.select_sentences(x, "y"), axis=1)
        features['sentences_tagged'] = features['sentences'].apply(lambda x: [self.tag_sentence(nlp, w) for w in x])

        # Calculating Readability-Index
        # words in requirement
        features['words_nb'] = features['req'].apply(lambda x: len(x.split()))
        # words per sentence
        features['WPS'] = features['words_nb'] / features['sentences_nb']
        # syllables per word
        features['SPW'] = features['req'].apply(lambda x: self.compute_SPW(x))
        # flesch index
        features['Flesch_Index'] = features.apply(lambda x: round((180 - x['WPS'] - (58.5 * x['SPW']))), axis=1)
        # Analyzing punctuation
        features['internal_punctuation'] = features['tags'].apply(lambda x: self.count_punctuation(x))
        features['comma'] = features['tags'].apply(lambda x: self.count_comma(x))
        features['weird_words'] = features['tags'].apply(lambda x: self.count_weird_words(x))

        # Analyzing and counting specific words and list containing words
        features['beispiel'] = features['tags'].apply(lambda x: self.search_specific_words(x, 'beispiel'))
        features['circa'] = features['tags'].apply(lambda x: self.search_specific_words(x, 'circa'))
        features['wenn'] = features['tags'].apply(lambda x: self.search_specific_words(x, 'wenn'))
        features['aber'] = features['tags'].apply(lambda x: self.search_specific_words(x, 'aber'))
        features['max_min_presence'] = features['req'].apply(lambda x: self.check_max_min_presence(x))
        features['Nb_of_Umsetzbarkeit_conj'] = features['tags'].apply(lambda x: self.time_logical_conj(x))
        features['measurement_values'] = features['tags'].apply(lambda x: self.search_measurements_indicators(x))
        features['numerical_values'] = features['tags'].apply(lambda x: self.search_numerical_value(x))
        features['polarity'] = features['req'].map(lambda text: TextBlobDE(text).sentiment.polarity)

        # Analyzing passive and active and auxiliary attributes at the beginning of a requirement
        features['passive_global'] = features['tags'].apply(lambda x: self.passive_detection(x))
        features['passive_per_sentence'] = features['sentences_tagged'].apply(
            lambda x: [self.passive_detection(s) for s in x])
        features['passive_percent'] = features['passive_per_sentence'].apply(
            lambda x: (sum([y == "yes" for y in x]) / len(x)))
        features['Aux_Start'] = features['tags'].apply(lambda x: self.aux_1st(x))
        features['Aux_Start_per_sentence'] = features['sentences_tagged'].apply(lambda x: [self.aux_1st(s) for s in x])

        # Analyzing conjunctions, verbs and auxiliaries
        features['Sub_Conj'] = features['tags'].apply(lambda x: self.count_subordinate_conjunction(x))
        features['Comp_conj'] = features['tags'].apply(lambda x: self.count_comp_coor_conjunction(x))
        features['Nb_of_verbs'] = features['tags'].apply(lambda x: self.count_verb(x))
        features['Nb_of_auxiliary'] = features['tags'].apply(lambda x: self.count_aux(x))
        features['werden'] = features['req'].apply(lambda x: self.count_werden(x))

        # same functions as previous block but analysis made for each sentence on one requirement
        features['Sub_Conj_pro_sentece'] = features['sentences_tagged'].apply(
            lambda x: [self.count_subordinate_conjunction(s) for s in x])
        features['Comp_conj_pro_sentence'] = features['sentences_tagged'].apply(
            lambda x: [self.count_comp_coor_conjunction(s) for s in x])
        features['Nb_of_verbs_pro_sentence'] = features['sentences_tagged'].apply(
            lambda x: [self.count_verb(s) for s in x])
        features['Nb_of_auxiliary_pro_sentence'] = features['sentences_tagged'].apply(
            lambda x: [self.count_aux(s) for s in x])
        features['werden_pro_sentence'] = features['sentences'].apply(lambda x: [self.count_werden(s) for s in x])

        features['formal_global'] = features['req'].apply(lambda x: self.contain_Muss_Darf_nicht(stemmer, x))
        features['formal_per_sentence'] = features['sentences'].apply(
            lambda x: [self.contain_Muss_Darf_nicht(stemmer, s) for s in x])
        features['formal_percent'] = features['formal_per_sentence'].apply(
            lambda x: (sum([y == "yes" for y in x]) / len(x)))
        features['entities'] = features['req_nlp'].apply(lambda x: self.entities_label(x))

        # Graphical representation of the vocabulary of requirements corpus
        if corpal:
            self.Corpus_Analysis(Req_list, stop)

        if export:
            my_path = Path(u"/Users/selina/Code/Python/Thesis/src/Features/" + 'export_features')
            # my_path = Path(u"/Users/selina/Documents/UNI/Thesis/Code/Features/" + 'export_features')
            g_Dirpath = os.path.abspath(my_path)
            dataFile = g_Dirpath + '\\' + 'Features_Export.xlsx'
            print("Create Excel export file: %s" % (dataFile))
            features[0:5000].to_excel(dataFile, index=False)
            print("\nFeatures_Export XLS-file created and data copied.")

        return features, features.sentences_tagged

    # ======================================================================================================= #

    """ Reads the "requirement" column of an excel file and returns it as a dataframe """

    def readData_csv(self, address):
        self.df = pd.read_csv(address, sep=";")
        self.Req = self.df['requirement']
        print('**********************')
        return self.Req

    """ Reads the "requirement" column of an excel file and returns it as a dataframe """

    def readData_excel(self, address, worksheet):
        self.df = pd.read_excel(address, worksheet, encoding="utf-8")
        self.Req = self.df['requirement']
        return self.Req