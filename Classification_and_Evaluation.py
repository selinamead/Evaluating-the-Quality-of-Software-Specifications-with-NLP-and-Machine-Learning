from sklearn.feature_selection import chi2
from sklearn import metrics

from Utils import *
from NLP_Features import *

import warnings

import sys
import re
import os
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, \
    classification_report
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import cross_val_score

from imblearn.over_sampling import SMOTE
from collections import Counter
from matplotlib import pyplot

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

import sys
for p in sys.path:
    print(p)

CV = 10
min_df = 10
idx = 3
# classifier = 'RFC'
# classifier = 'SVM'
# classifier = 'LogReg'
classifier = 'MNB'
ger_stopws = set(stopwords.words('german'))  # move here so can use for tfidf


class Specification_Evaluation():

    def __init__(self, file_path):
        warnings.filterwarnings('ignore')
        self.data_location = file_path
        print('Specification_Evaluation')

    ''' Convert requirement evaluation spreadsheet into dataframe'''
    def get_data(self, score_target):
        print('Converting requirement evaluation spreadsheet into dataframe')

        self.df = pd.read_excel(self.data_location)
        # self.df = pd.read_excel(self.data_location, encoding="UTF-8")

        print(self.df.shape)
        # print(self.df.head())

        # Remove rows where type is not evaluated
        self.df_type = pd.DataFrame(self.df[self.df["score"].notnull()].values, columns=self.df.columns)
        self.df_type["score"] = self.df_type["score"].astype('category')
        # print(self.df_type.head())
        print(self.df_type.shape)

        self.df_type["score_cat"] = self.df_type["score"].cat.codes
        # print(self.df_type.head())
        # print(self.df_type.shape)

        self.data = pd.DataFrame(self.df_type[['requirement', score_target]].values,
                                 columns=['requirement', score_target])
        # print(self.data.head())
        # Convert target to string
        self.data['target'] = self.data[score_target].astype(str)
        # print(self.data.head())

        # Remove string na's
        self.data = self.data[~self.data.target.str.contains("na")]

        print('\nNo of Requirements in dataset:', len(self.data))

        print('\nProcessing Requirements and Extracting Features...\n')

        return self.data

    ''' Create feature matrix using NLP features created in NLP_Features.py '''
    def get_features(self, score_target, metric_create=True, hotenc=False, export_dfm=True):
        print('\nCreating Feature Matrix...')

        if metric_create:
            parser = NLP_Features()
            pos_tag_req = []
            # get the features for all the requirements selected with the features() method
            self.features, pos_tag_req = parser.extract_features(self.data['requirement'], score_target, export=True,
                                                                 corpal=False)
            # Shift to Metrics.py to analyze requirements
            self.dfm = self.features
        # print(self.dfm.head())

        else:
            self.dfm = pd.read_excel(u'/Users/selina/Code/Python/Thesis/src/Generated_Files/'+ str(score_target)+'/Features_Export.xlsx',encoding = "utf-8")
            # self.dfm = pd.read_excel(u'/Users/selina/Documents/UNI/Thesis/Code/Generated_Files/' + str(score_target) + '/Features_Export.xlsx', encoding="utf-8")

        # The following cols will be dropped as not useful
        cols_to_drop = ['req',
                        'req_nlp',
                        'tags',
                        'sentences_by_nlp',
                        'sentence_nb_by_nlp',
                        'sentences_by_nltk',
                        'sentence_nb_by_nltk',
                        'sentences',
                        'sentences_tagged',
                        # 'weakwords_nb2',
                        # 'weakwords_nb2_lemma',
                        # 'difference',
                        'passive_per_sentence',
                        'passive_percent',
                        'Aux_Start_per_sentence',
                        'Sub_Conj_pro_sentece',
                        'Comp_conj_pro_sentence',
                        'Nb_of_verbs_pro_sentence',
                        'Nb_of_auxiliary_pro_sentence',
                        'werden_pro_sentence',
                        'formal_percent',
                        'formal_per_sentence',
                        'entities',
                        ]

        self.dfm.drop(cols_to_drop, axis=1, inplace=True)
        # print(self.dfm)

        # hotenc
        if hotenc:
            self.dfm["max_min_presence"] = self.dfm["max_min_presence"].astype('category')
            self.dfm["max_min_presence"] = self.dfm["max_min_presence"].cat.codes

            self.dfm["measurement_values"] = self.dfm["measurement_values"].astype('category')
            self.dfm["measurement_values"] = self.dfm["measurement_values"].cat.codes

            self.dfm["passive_global"] = self.dfm["passive_global"].astype('category')
            self.dfm["passive_global"] = self.dfm["passive_global"].cat.codes

            self.dfm["Aux_Start"] = self.dfm["Aux_Start"].astype('category')
            self.dfm["Aux_Start"] = self.dfm["Aux_Start"].cat.codes

            self.dfm["formal_global"] = self.dfm["formal_global"].astype('category')
            self.dfm["formal_global"] = self.dfm["formal_global"].cat.codes

        # function to seperate numerical and categorical columns of a input dataframe
        def num_cat_separation(df):
            metrics_num = []
            metrics_cat = []
            for col in df.columns:
                if df[col].dtype == "object":
                    metrics_cat.append(col)
                else:
                    metrics_num.append(col)
            # print("Categorical columns : {}".format(metrics_cat))
            # print("Numerical columns : {}".format(metrics_num))
            return (metrics_cat, metrics_num)

        features_cat, features_num = num_cat_separation(self.dfm)
        # transform categorical columns into 1 and 0
        self.dfm = pd.get_dummies(self.dfm, features_cat)

        # print(self.dfm.head())
        # -------------------------------------------------------------------------------------------------------------
        # Normalise Data (StandardScaler or MinMax?)
        # RFC doesn't need scaled data however others do
        print('Scaling Data...')
        normalise = 'MM'
        # normalise = 'SS'

        if normalise == 'MM':
            dfm_scaled = self.dfm.copy()
            scaler = MinMaxScaler()
            dfm_scale = scaler.fit_transform(dfm_scaled)
            dfm_scaled = pd.DataFrame(dfm_scale, index=dfm_scaled.index, columns=dfm_scaled.columns)
            self.dfm_scaled = pd.DataFrame(dfm_scaled)
        # self.dfm_scaled = scaler.fit_transform(self.dfm)
        # print(self.dfm_scaled)

        if normalise == 'SS':
            dfm_scaled = self.dfm.copy()
            scaler = StandardScaler()
            dfm_scale = scaler.fit_transform(dfm_scaled)
            dfm_scaled = pd.DataFrame(dfm_scale, index=dfm_scaled.index, columns=dfm_scaled.columns)
            self.dfm_scaled = pd.DataFrame(dfm_scaled)
        # self.dfm_scaled = scaler.fit_transform(self.dfm)
        # print(self.dfm_scaled)

        if export_dfm:
            datafile = "./Generated_Files/" + str(score_target) + "/DFM_Export/DFM_Export_" + str(
                score_target) + ".xlsx"
            self.dfm.to_excel(datafile, index=False)
            dfm_scale = self.dfm.copy()
            scaler = MinMaxScaler()
            dfm_scale2 = scaler.fit_transform(dfm_scale)
            dfm_scale = pd.DataFrame(dfm_scale2, index=dfm_scale.index, columns=dfm_scale.columns)
            dfm_scale = pd.DataFrame(dfm_scale)
            datafile_scale = "./Generated_Files/" + str(score_target) + "/DFM_Export/DFM_Export_Scale_" + str(
                score_target) + ".xlsx"
            dfm_scale.to_excel(datafile_scale, index=False)
        # print ("Create Excel export file: %s"%(datafile))
        # print ("Create Excel export file: %s"%(datafile_scale))

        return self.dfm, self.dfm_scaled

    ''' Preprocessing data ready for use in model (stemming, tokenisation)'''
    def preprocessing(self, score_target):
        print('preprocessing uses dfm and converts to features')

        # Function for stopwords removal
        # self.ger_stopws = set(stopwords.words('german'))
        german_sw = set(get_stop_words('german'))

        # Function to take the stop words out of a tokenized list
        def stopword_removal(token_list):
            new_list = []
            for token in token_list:
                if token not in ger_stopws:
                    new_list.append(token)
            return new_list

        # Function to carry out stemming
        stemmer = SnowballStemmer("german")

        def word_stemming(token_list):
            stemming_list = []
            for word in token_list:
                stemming_list.append(stemmer.stem(word))
            return stemming_list

        # Tokenize requirements
        tokenizer = nltk.RegexpTokenizer(pattern='\w+|\$[\d\.]+|\S+')
        text_tokenized = self.data['requirement'].apply(lambda x: tokenizer.tokenize(x))
        # print(text_tokenized)
        # input()

        # Remove stopwords
        text_stopwords = text_tokenized.apply(stopword_removal)

        # stemming requirements
        text_stemmed = text_stopwords.apply(word_stemming)
        self.text_stemmed_string = text_stemmed.apply(lambda x: " ".join(x))

        # print('Problem with req_process')

        # print(text_stemmed)
        # input()
        # print(text_stemmed_string)
        # input()

        # Add a new column to dataframe which contains the pre-processed requirements
        self.data['req_process'] = self.text_stemmed_string
        # print(self.data['req_process'])
        # input()

        export_token = True

        if export_token:
            datafile = "./Generated_Files/" + str(score_target) + "/DFM_Export/Toke_Stop_Stem_Export_" + str(
                score_target) + ".xlsx"
            self.data.to_excel(datafile, index=False)
        # print ("\nCreate Excel export file: %s"%(datafile))

        # Combine preprocessed requirements (dfm) with features
        self.features = self.dfm.copy()
        self.features['req_process'] = self.data['req_process']
        self.features['target'] = self.data['target']
        # Combine scaled dfm with features
        self.features_scaled = self.dfm_scaled.copy()
        self.features_scaled['req_process'] = self.data['req_process']
        self.features_scaled['target'] = self.data['target']

        export_token_features = True

        if export_token_features:
            datafile = "./Generated_Files/" + str(score_target) + "/DFM_Export/DFM_ProcReqs_Export_" + str(
                score_target) + ".xlsx"
            self.features.to_excel(datafile, index=False)
        # print ("Create Excel export file: %s"%(datafile))

        # Remove na's from target for scaled and non-scaled
        self.features = self.features[~self.features.target.str.contains("na")]
        self.features_scaled = self.features_scaled[~self.features_scaled.target.str.contains("na")]

        print('\nFeatures Dataframe data types:\n', self.features.dtypes)

        print('\nPreprocessing Data Complete!!!')

    ''' Combine tfidf and NLP features using Pipeline'''
    def combine_features(self, score_target):
        # dfm = self.features  # Doing this so I can save and load features without preprocessing each time
        from sklearn.compose import ColumnTransformer
        # https://medium.com/vickdata/a-simple-guide-to-scikit-learn-pipelines-4ac0d974bdcf
        # SimpleInputer (changes na values to median)

        ### First pipeline performs tfidf vectorization on dataframe containing processed requirements

        self.tfidf_pipeline = Pipeline([
            ('selector', DataFrameSelector('req_process')),  # working with self.data
            # ('selector', DataFrameSelector('requirement')), # working with stemmed_text_df but should be the same as self.data
            ('vectorizer',
             TfidfVectorizer(stop_words=ger_stopws, min_df=min_df, encoding="utf-8", norm="l2", ngram_range=(1, 2)))
        ])
        # Data to be saved and loaded
        tfidf_features = self.tfidf_pipeline.fit_transform(self.data)
        tfidf_features = tfidf_features.todense()
        self.tfidf_features_df = pd.DataFrame(tfidf_features)
        # print(self.tfidf_features_df)

        ### 2nd pipeline scales scales feature values from a dataframe

        self.features_pipeline = Pipeline([
            ('selector', DataFrameSelector(self.dfm.columns)),
            ('normalizer', MinMaxScaler())
        ])
        # Data to be saved and loaded
        feat_pipe_test_df = pd.DataFrame(self.dfm)
        feat_pipe_test_df = feat_pipe_test_df.loc[:, feat_pipe_test_df.columns != 'req_process']
        feat_pipe_test_df = feat_pipe_test_df.loc[:, feat_pipe_test_df.columns != 'target']
        transformed_features = self.features_pipeline.fit_transform(feat_pipe_test_df)
        self.transformed_features_df = pd.DataFrame(transformed_features, columns=self.dfm.columns)

        ### 3rd Pipeline: Combine pipelines together to have two lots of features (NLP and TFIDF)

        self.combined_pipeline = FeatureUnion(transformer_list=[
            ("tfidf_pipeline", self.tfidf_pipeline),
            ("features_pipeline", self.features_pipeline)
        ])

        ### 4th Pipeline contains a pipeline of features and the classifeir to be used
        self.classifier_pipeline = Pipeline([
            ("combined_pipeline", self.combined_pipeline),
            ("classifier", RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42))
        ])

    ''' Save features matrix to skip preprocessing '''
    def save_features_matrix(self, score_target):

        print('Saving features Matrix to file...')
        self.features.to_csv('features_matrix.csv')
        self.features.to_csv('features_scaled_matrix.csv')
        self.tfidf_features_df.to_csv('tfidf_features.csv')
        self.transformed_features_df.to_csv('transformed_features.csv')

    ''' Lpoad features matrix to skip preprocessing '''
    def load_features_matrix(self, score_target):

        print('Loading features matrix from file...')
        # Load feature matrix (and convert target to int)
        features_matrix = pd.read_csv("features_matrix.csv")
        self.features = features_matrix.copy()
        features_scaled_matrix = pd.read_csv("features_scaled_matrix.csv")
        self.features_scaled = features_scaled_matrix.copy()

        # Pipeline data
        self.tfidf_features = pd.read_csv('tfidf_features.csv')
        self.transformed_features = pd.read_csv('transformed_features.csv')

    ''' Function to compare differing ML models to find most suitable model'''
    def classifier_comparison(self, score_target):
        print('\n======== Model Comparison ========\n')

        '''  
        -----------  different models possibilities   -----------------------------------------------------------------------------
        Benchmark the following models:
            RandomForestClassifier                       # A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and use averaging to improve the predictive accuracy and control over-fitting.
            MultinomialNB                                # The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification). The multinomial distribution normally requires integer feature counts. However, in practice, fractional counts such as tf-idf may also work.     
            Logistic Regression                          # Logistic Regression (aka logit, MaxEnt) classifier.In the multiclass case, the training algorithm uses the one-vs-rest (OvR) scheme if the �multi_class� option is set to �ovr�, and uses the cross- entropy loss if the �multi_class� option is set to �multinomial�.  
       ----------------------------------------------------------------------------------------------------------------------------------     
       '''

        # Assign training_data and labels
        features = self.features_scaled.copy()
        features['target'] = self.data['target']

        # Assigning X, y
        X = features.loc[:, features.columns != 'target']
        y = features.loc[:, features.columns == 'target'].values.ravel()
        # Convert to ints
        y = [int(i) for i in y]

        # Split train and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

        # Models to compare
        models = [
            RandomForestClassifier(n_estimators=200, max_depth=30, random_state=0),
            LogisticRegression(random_state=42, solver='lbfgs', multi_class='multinomial', max_iter=1000),
            SVC(decision_function_shape="ovo", C=1, gamma=0.1, verbose=False, kernel="rbf")
        ]

        # list that will merge the results
        entries = []
        # For each model: get name and accuracy
        for model in models:
            model_name = model.__class__.__name__
            accuracies = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=CV)
            # for each fold get fold num and acc
            for fold_idx, accuracy in enumerate(accuracies):
                entries.append((model_name, fold_idx, accuracy))

        # Df to store results
        cv_df = pd.DataFrame(index=range(CV * len(models)))
        cv_df = pd.DataFrame(entries, columns=['classifier', 'fold_idx', 'accuracy'])
        print(cv_df.groupby('classifier').accuracy.mean())

        sns.boxplot(x='classifier', y='accuracy', data=cv_df)
        sns.stripplot(x='classifier', y='accuracy', data=cv_df, size=5, jitter=True, edgecolor="gray", linewidth=1)
        plt.xticks(rotation=45, ha="right")
        plt.savefig('./Generated_Files/' + str(score_target) + '/Benchmark_Export_' + str(score_target) + '.png',
                    dpi=300, format='png', bbox_inches='tight')  # use format='svg' or 'pdf' for vectorial pictures
        plt.show(block=False)
        plt.pause(2)
        plt.clf()
        plt.close()

    ''' Function to find the best params for RFC '''
    def grid_search(self, score_target):
        print('======== Grid Search ========')

        # Function to find accuracy score of each classifier
        def evaluate(model, X_test, y_test):
            predictions = model.predict(X_test)
            errors = abs(predictions - y_test)
            mape = 100 * np.mean(errors / y_test)
            accuracy = 100 - mape
            print('\n+++ Model Performance +++')
            print('Model: ', model)  # Check model
            print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
            print('Accuracy = {:0.2f}%.'.format(accuracy))
            return accuracy

        # Assigning training and test data #

        nlp_feature_matrix = pd.DataFrame(self.features_scaled)
        nlp_feature_matrix = nlp_feature_matrix.loc[:, nlp_feature_matrix.columns != 'req_process']
        # Convert to numpy
        y = np.array(nlp_feature_matrix['target'])
        # axis 1 refers to the columns
        X = nlp_feature_matrix.drop('target', axis=1)
        # Saving feature names
        feature_names = list(X.columns)
        # print(feature_names)
        # Convert to numpy array
        X = np.array(X)
        print(X.shape, y.shape)

        # Split train and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        # ### Choose Classifier ###
        # # classifier = "RFC"
        # # classifier = "SVM"
        # classifier = "LogReg"
        # # classifier = "NB"

        if classifier == "RFC":
            clf = RandomForestClassifier()
            print('\n+++ Parameters currently in use +++\n')
            print(clf.get_params())

            # Number of trees in random forest
            n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
            # Number of features to consider at every split
            max_features = ['auto', 'sqrt']
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(start=10, stop=100, num=10)]
            max_depth.append(None)
            # Minimum number of samples required to split a node
            min_samples_split = [2, 5, 10]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 2, 4]
            # Method of selecting samples for training each tree
            bootstrap = [True, False]
            # Create the random grid
            random_grid = {'n_estimators': n_estimators,
                           'max_features': max_features,
                           'max_depth': max_depth,
                           'min_samples_split': min_samples_split,
                           'min_samples_leaf': min_samples_leaf,
                           'bootstrap': bootstrap}
            # print('\n+++ Random Grid +++\n', random_grid)
            '''
            {'bootstrap': [True, False],
             'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
             'max_features': ['auto', 'sqrt'],
             'min_samples_leaf': [1, 2, 4],
             'min_samples_split': [2, 5, 10],
             'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
            '''

            # Use the random grid to search for best hyperparameters

            # Random search of parameters, using 3 fold cross validation
            rf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, n_iter=100, cv=CV, verbose=2,
                                           random_state=42, n_jobs=-1)
            # Fit the random search model
            rf_random.fit(X_train, y_train)

            # View the best parameters from fitting the Random search
            print('\n+++ Best Params +++\n', rf_random.best_params_)

            base_model = clf
            base_model.fit(X_train, y_train)
            base_accuracy = evaluate(base_model, X_test, y_test)

            best_random = rf_random.best_estimator_
            random_accuracy = evaluate(best_random, X_test, y_test)

            print('\nImprovement of {:0.2f}%.'.format(100 * (random_accuracy - base_accuracy) / base_accuracy))

        if classifier == "SVM":
            clf = SVC()
            print('\n+++ Parameters for SVM +++\n')
            print(clf.get_params())

            # Dict of possible parameters
            params_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],  # 10
                           'gamma': [0.0001, 0.001, 0.01, 0.1],  # 0.1
                           # 'kernel':['rbf', 'linear'] } # rbf
                           'kernel': ['rbf']}  # rbf

            grid_SVM = GridSearchCV(SVC(), params_grid, refit=True, verbose=3, cv=CV, n_jobs=-1)
            # Fit the data
            grid_SVM.fit(X_train, y_train)

            # View the best parameters from fitting the Random search
            print('\n+++ Best Params +++\n', grid_SVM.best_params_)

            base_model = clf
            base_model.fit(X_train, y_train)
            base_accuracy = evaluate(base_model, X_test, y_test)

            best_random = grid_SVM.best_estimator_
            random_accuracy = evaluate(best_random, X_test, y_test)

            print('\nImprovement of {:0.2f}%.'.format(100 * (random_accuracy - base_accuracy) / base_accuracy))

        if classifier == "LogReg":
            # variations with respect to type of regularization, size of penalty, and type of solver used
            clf = LogisticRegression()
            print('\n+++ Parameters for Logistic Regression +++\n')
            print(clf.get_params())

            # Dict of possible parameters
            params_grid = {'penalty': ['l2'],
                           'C': (0.1, 1.0),
                           # 'C' : np.logspace(-4, 4, 20),
                           'max_iter': [1000, 1500],
                           'solver': ['lbfgs'],  # 'newton-cg',
                           'multi_class': ['ovr']},  # 'multinomial'

            grid_LogReg = GridSearchCV(LogisticRegression(), params_grid, verbose=3, cv=CV, n_jobs=-1)
            # Fit the data
            grid_LogReg.fit(X_train, y_train)

            # View the best parameters from fitting the Random search
            print('\n+++ Best Params +++\n', grid_LogReg.best_params_)

            base_model = clf
            base_model.fit(X_train, y_train)
            base_accuracy = evaluate(base_model, X_test, y_test)

            best_random = grid_LogReg.best_estimator_
            random_accuracy = evaluate(best_random, X_test, y_test)

            print('\nImprovement of {:0.2f}%.'.format(100 * (random_accuracy - base_accuracy) / base_accuracy))

    ' Build a model to classify the requirements into 5 categories (1, 2, 3, 4, 5) '
    def multi_class_classification(self, score_target, smote, save_model, combine_features, nlp_features, tfidf, display):
        print('\t\t\t\t *** MULTICLASS CLASSIFICATION ***\n')
        # -------------------------------------------------------------------------------------------------------------------------------------------- #
        """ Option to choose at beginning of code - move to main"""

        if classifier == "RFC":
            clf = RandomForestClassifier(n_estimators=200, max_depth=30, random_state=0)
            print('\n+++ RANDOM FOREST +++\n')
        if classifier == "LogReg":
            clf = LogisticRegression(C=1.0, random_state=42, solver='lbfgs', multi_class='ovr', max_iter=1000)
            print('\n+++ LOGISTIC REGRESSION +++\n')
        if classifier == "SVM":
            clf = svm.SVC(decision_function_shape="ovo", C=10, gamma=0.1, verbose=False, kernel="rbf")
            print('\n+++ SUPPORT VECTOR MACHINE +++\n')
        if classifier == "MNB":
            clf = GaussianNB()
            print('\n+++ MULTINOMIAL NAIVE BAYES +++\n')

        # ------------------------------------------------------------------------------------------------------------ #
        ### Use Pipeline to combine both tfidf and nlp features and build classifier model ###
        if combine_features:

            ## Assigning training and test data ##
            feature_matrix = pd.DataFrame(self.features)
            # print('Feature Matrix DF:\n', feature_matrix)
            # Assigning X, y
            X = feature_matrix.loc[:, feature_matrix.columns != 'target']
            y = feature_matrix.loc[:, feature_matrix.columns == 'target'].values.ravel()
            # Convert to ints
            y = [int(i) for i in y]
            # Split train and test data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
            # print('X_Train Data:\n', X_train)

            ## Build classifier pipeline model - combines NLP features with TFIDF features ##
            self.classifier_pipeline.fit(X_train, y_train)
            pred_y = self.classifier_pipeline.predict(X_test)
            # print('\nPIPE Predictions\n', pred_y)
            cv_score_pipe = cross_val_score(self.classifier_pipeline, X_train, y_train, cv=CV)
            acc_pipe = accuracy_score(y_test, pred_y)
            prec_rec_f1_pipe = precision_recall_fscore_support(y_test, pred_y, average='weighted')
            cv_accuracy_pipe = cv_score_pipe.mean()

            # -------------------------------------------------------------------------------------------------------------------------------------------- #
            # Display results #
            if display:
                print('\n+++ RESULTS OF COMBINED FEATURES +++')
                print('\nModel Accuracy:', '{:.1%}'.format(accuracy_score(y_test, pred_y).round(3)))
                print('\nPrecision: ', '{:.1%}'.format(prec_rec_f1_pipe[0].round(3)), '\nRecall: ',
                      '{:.1%}'.format(prec_rec_f1_pipe[1].round(3)), '\nFscore: ', '{:.1%}'.format(prec_rec_f1_pipe[2].round(3)))
                print("\n=== Confusion Matrix === \n", confusion_matrix(y_test, pred_y))
                print("\n=== Classification Report ===\n", classification_report(y_test, pred_y))
                print('\n=== 10 Fold Cross Validation Scores ===')
                num = 0
                for i in cv_score_pipe:
                    num += 1
                    print('CVFold', num, '=', '{:.1%}'.format(i))
                print("\nMean Cross Validation Score: ", '{:.1%}'.format(cv_accuracy_pipe.round(3)))
                print('\n=== Final Model Results ===')
                print('\nAccuracy:', '{:.1%}'.format(accuracy_score(y_test, pred_y).round(3)))
                print('\nPrecision: ', '{:.1%}'.format(prec_rec_f1_pipe[0].round(3)), '\nRecall: ',
                      '{:.1%}'.format(prec_rec_f1_pipe[1].round(3)), '\nFscore: ', '{:.1%}'.format(prec_rec_f1_pipe[2].round(3)))
                # Confusion Matrix
                conf_mat = confusion_matrix(y_test, pred_y)
                fig, ax = plt.subplots(figsize=(8, 8))
                sns.heatmap(conf_mat, annot=True, fmt='d')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                all_sample_title = 'Accuracy Score: {0}'.format(acc_pipe)
                plt.title(all_sample_title, size=15);
                plt.tight_layout()
                plt.savefig('./Generated_Files/' + str(score_target) + '/Conf_Matrix_Multi_' + str(score_target) + '.png',
                            dpi=300, format='png', bbox_inches='tight')  # use format='svg' or 'pdf' for vectorial pictures
                # plt.show(block=False)
                # plt.pause(3)
                plt.clf()
                plt.close()

        # -------------------------------------------------------------------------------------------------------------------------------------------- #
        ### Perform classification using only NLP features ###
        if nlp_features:
            ## Assigning training and test data ##
            nlp_feature_matrix = pd.DataFrame(self.features_scaled)
            nlp_feature_matrix = nlp_feature_matrix.loc[:, nlp_feature_matrix.columns != 'req_process']

            # Convert to numpy
            y = np.array(nlp_feature_matrix['target'])
            # Convert y values from str to int
            y = y.astype(np.int)
            # axis 1 refers to the columns
            X = nlp_feature_matrix.drop('target', axis=1)
            # Saving feature names
            nlp_feature_names = list(X.columns)
            # print(nlp_feature_names)
            # Convert to numpy array
            X = np.array(X)
            # print('Shape of X and y: ', X.shape, y.shape)

            # Split train and test data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
            # print('X_train data:\n', X_train)

            # -------------------------------------------------------------------------------------------------------------------------------------------- #
            ### Implement SMOTE to create new data to balance classes ###
            if smote:
                # summarize class distribution before over sample
                counter = Counter(y_train)
                for k, v in counter.items():
                    per = v / len(y_train) * 100
                    print(k, v, per)
                print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
                oversample = SMOTE()
                X_train, y_train = oversample.fit_resample(X_train, y_train)
                # summarize class distribution after over sample
                counter = Counter(y_train)
                for k, v in counter.items():
                    per = v / len(y_train) * 100
                    # print(k, v, per)
                    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))

            # -------------------------------------------------------------------------------------------------------------------------------------------- #
            ## Build classifier model without tfidf features ##
            clf.fit(X_train, y_train)
            print('Classifier: ', clf)
            # Predict model
            predicted = clf.predict(X_test)

            # Perform cross validation
            # cv_score = cross_val_score(classifier, X_train, y_train.values.ravel(), cv=CV)
            cv_score = cross_val_score(clf, X_train, y_train, cv=CV)

            # Store results
            acc = accuracy_score(y_test, predicted)
            prec_rec_f1 = precision_recall_fscore_support(y_test, predicted, average='weighted')
            cv_accuracy = cv_score.mean()

            # -------------------------------------------------------------------------------------------------------------------------------------------- #
            # Display results #
            if display:
                print('\n+++ RESULTS FOR NLP FEATURES +++')
                print('\nModel Accuracy:', '{:.1%}'.format(accuracy_score(y_test, predicted).round(3)))
                print('\nPrecision: ', '{:.1%}'.format(prec_rec_f1[0].round(3)), '\nRecall: ',
                      '{:.1%}'.format(prec_rec_f1[1].round(3)), '\nFscore: ', '{:.1%}'.format(prec_rec_f1[2].round(3)))
                print("\n=== Confusion Matrix === \n", confusion_matrix(y_test, predicted))
                print("\n=== Classification Report ===\n", classification_report(y_test, predicted))
                print('\n=== 10 Fold Cross Validation Scores ===')
                num = 0
                for i in cv_score:
                    num += 1
                    print('CVFold', num, '=', '{:.1%}'.format(i))
                print("\nMean Cross Validation Score: ", '{:.1%}'.format(cv_accuracy.round(3)))
                print('\n=== Final Model Results ===')
                print('\nAccuracy:', '{:.1%}'.format(accuracy_score(y_test, predicted).round(3)))
                print('\nPrecision: ', '{:.1%}'.format(prec_rec_f1[0].round(3)), '\nRecall: ',
                      '{:.1%}'.format(prec_rec_f1[1].round(3)), '\nFscore: ', '{:.1%}'.format(prec_rec_f1[2].round(3)))
                # Confusion Matrix
                conf_mat = confusion_matrix(y_test, predicted)
                fig, ax = plt.subplots(figsize=(8, 8))
                sns.heatmap(conf_mat, annot=True, fmt='d')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                all_sample_title = 'Accuracy Score: {0}'.format(acc)
                plt.title(all_sample_title, size=15);
                plt.tight_layout()
                plt.savefig('./Generated_Files/' + str(score_target) + '/Conf_Matrix_Multi_' + str(score_target) + '.png',
                            dpi=300, format='png', bbox_inches='tight')  # use format='svg' or 'pdf' for vectorial pictures
                # plt.show(block=False)
                # plt.pause(3)
                plt.clf()
                plt.close()

        # -------------------------------------------------------------------------------------------------------------------------------------------- #
        ### tfidf ###
        if tfidf:
            # Create a dataframe with only single requirements
            df = pd.DataFrame()
            df['Requirement'] = self.features['req_process']  # Should be req_process?
            df['Rating'] = self.features['target']
            df['category_id'] = self.features['target'].values
            # print(df.head)

            category_id_df = df[['Rating', 'category_id']].sort_values('category_id')
            # category_id_df = df[['Requirement', 'category_id']].sort_values('category_id')
            category_to_id = dict(category_id_df.values)
            id_to_category = dict(category_id_df[['category_id', 'Rating']].values)
            # id_to_category = dict(category_id_df[['category_id', 'Requirement']].values)

            tfidf = TfidfVectorizer(sublinear_tf=True, min_df=min_df, norm='l2', encoding='utf-8',
                                    ngram_range=(1, 2),
                                    stop_words=ger_stopws)

            features = tfidf.fit_transform(df.Requirement).toarray()
            labels = df.category_id
            print(
                "Each of the %d requirements are represented by %d features (TF-IDF score of unigrams and bigrams)"
                % (features.shape))
            # print('\nShape of tfidf features matrix: ', features.shape)
            # print('\ntfidf features matrix: \n', features)

            # FInd most correlated terms
            N = 5
            # for Rating, category_id in sorted(category_to_id.items()):
            for rating, category_id in sorted(category_to_id.items()):
                features_chi2 = chi2(features, labels == category_id)
                indices = np.argsort(features_chi2[0])
                feature_names = np.array(tfidf.get_feature_names())[indices]
                unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
                bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
                print("\n==> %s:" % (rating))
                print("  * Most Correlated Unigrams are: %s" % (', '.join(unigrams[-N:])))
                print("  * Most Correlated Bigrams are: %s" % (', '.join(bigrams[-N:])))

            # Split test, train and perform tfidf vectorizer
            X_train, X_test, y_train, y_test = train_test_split(df['Requirement'], df['Rating'], random_state=0)

            count_vect = CountVectorizer()
            X_train_counts = count_vect.fit_transform(X_train)
            tfidf_transformer = TfidfTransformer()
            X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

            # Test different classifiers
            models = [
                RandomForestClassifier(n_estimators=200, max_depth=30, random_state=0, class_weight='balanced'),
                svm.SVC(decision_function_shape="ovo", C=10, gamma=0.1, verbose=False, kernel="rbf"),
                MultinomialNB(),
                LogisticRegression(C=1.0, random_state=42, solver='lbfgs', multi_class='ovr', max_iter=1000),
            ]
            # print('Model = RFC \n', models[0] )
            # index = 0
            # Cross Validation
            cv_df = pd.DataFrame(index=range(CV * len(models)))
            entries = []
            # Find best model
            for model in models:
                model_name = model.__class__.__name__
                accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
                for fold_idx, accuracy in enumerate(accuracies):
                    entries.append((model_name, fold_idx, accuracy))
            cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

            print('\nModel Performance: \n', cv_df.groupby('model_name').accuracy.mean().round(3), '\n')

            X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels,
                                                                                             df.index,
                                                                                             test_size=0.20,
                                                                                             random_state=0)
            # Fit the best model
            models[idx].fit(X_train, y_train)
            y_pred = models[idx].predict(X_test)

            # Classification report
            print('\t\t\t\tCLASSIFICATION METRICS\n')
            print(metrics.classification_report(y_test, y_pred,
                                                target_names=df['Rating'].unique()))

            # acc = np.mean(y_test == y_pred)
            acc = accuracy_score(y_pred, y_test)
            prec_rec_f1 = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            print('\n--- TFIDF Results ---')
            print('Accuracy: ', '{:.1%}'.format(accuracy_score(y_test.values.ravel(), y_pred).round(3)))
            # print("\nPrec_Recall_FScore: \n", precision_recall_fscore_support(y_test, y_pred, average='weighted'))
            print('\nPrecision: ', '{:.1%}'.format(prec_rec_f1[0].round(3)), '\nRecall: ',
                  '{:.1%}'.format(prec_rec_f1[1].round(3)), '\nFscore: ', '{:.1%}'.format(prec_rec_f1[2].round(3)))

            print("\n=== Confusion Matrix === \n", confusion_matrix(y_test, y_pred))

            # Feature Importance when using Random Forest
            if idx == 0:
                feat_importance = models[0].feature_importances_
                # print(feat_importance)
                feat_names = tfidf.get_feature_names()
                # print(feat_names)
                df_feat_importance = pd.DataFrame()
                df_feat_importance['feat'] = feat_names
                df_feat_importance['importance'] = feat_importance
                print('\n--- Top TFIDF features ---\n',
                      df_feat_importance.sort_values(by='importance', ascending=False).head(), '\n')

            # Display
            conf_mat = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 8))
            sns.heatmap(conf_mat, annot=True, fmt='d')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            all_sample_title = 'Accuracy Score: {0}'.format(acc)
            plt.title(all_sample_title, size=15)
            plt.tight_layout()
            plt.savefig(
                './Generated_Files/' + str(score_target) + '/Conf_Matrix_TFIDF_Multi_' + str(score_target) + '.png',
                dpi=300, format='png', bbox_inches='tight')  # use format='svg' or 'pdf' for vectorial pictures
            plt.show(block=False)
            plt.pause(5)
            plt.clf()
            plt.close()

    ''' Build a model to classify the requirements into 2 categories (1, 0) '''
    def binary_classification(self, score_target, combine_features, nlp_features, tfidf, smote, display):
        print('\t\t\t\t *** BINARY CLASS CLASSIFICATION ***\n')
        # -------------------------------------------------------------------------------------------------------------------------------------------- #
        ## Choose Classifier ##
        if classifier == "RFC":
            clf = RandomForestClassifier(n_estimators=200, max_depth=30, random_state=0)
            print('\n+++ RANDOM FOREST +++\n')
        if classifier == "LogReg":
            clf = LogisticRegression(C=1.0, random_state=42, solver='lbfgs', multi_class='ovr', max_iter=1000)
            print('\n+++ LOGISTIC REGRESSION +++\n')
        if classifier == "SVM":
            clf = svm.SVC(decision_function_shape="ovo", C=10, gamma=0.1, verbose=False, kernel="rbf")
            print('\n+++ SUPPORT VECTORE MACHINE +++\n')
        if classifier == "MNB":
            clf = GaussianNB()
            print('\n+++ MULTINOMIAL NAIVE BAYES +++\n')
        # -------------------------------------------------------------------------------------------------------------------------------------------- #
        ## Create binary dataframe ##
        binary_df = pd.DataFrame()
        # Create feature matrix df
        nlp_feature_matrix = pd.DataFrame(self.features)
        # nlp_feature_matrix = nlp_feature_matrix.loc[:, nlp_feature_matrix.columns != 'req_process'] # Removed as needed for tfidf (req)
        # print(nlp_feature_matrix)
        # convert target scores to string
        binary_df['target'] = nlp_feature_matrix['target'].apply(str)
        # Replaced good and bad classes
        binary_df['target'] = binary_df.target.replace(['1', '2'], '0')
        binary_df['target'] = binary_df.target.replace(['3', '4', '5'], '1')

        # binary_df['target'] = binary_df.target.replace(['4', '5'], '1')
        # Remove any requirements with target equalling 3
        # binary_df['target'] = binary_df['target'][~binary_df.target.str.contains('3')]

        # Replace target column of nlp_feature_matrix with new binary target column
        nlp_feature_matrix['target'] = binary_df['target']
        nlp_feature_matrix = nlp_feature_matrix.dropna()
        # print(nlp_feature_matrix.shape[0])

        # Find number of IO and NIO requirements
        IO_counts = len(nlp_feature_matrix[nlp_feature_matrix['target'].str.contains('1')])
        NIO_counts = len(nlp_feature_matrix[nlp_feature_matrix['target'].str.contains('0')])
        print('Num of In Ordnung Requirements: ', IO_counts)
        print('Num of Nicht In Ordnung Requirements: ', NIO_counts)
        # ------------------------------------------------------------------------------------------------------------ #
        ### Use Pipeline to combine both tfidf and nlp features and build classifier model ###
        if combine_features:
            ## Assigning training and test data ##
            comb_feature_matrix = pd.DataFrame(nlp_feature_matrix)
            # comb_feature_matrix['req_process'] = self.features['req_process']
            # print('Combined Feature Matrix DF:\n', comb_feature_matrix)
            # Assigning X, y
            X = comb_feature_matrix.loc[:, comb_feature_matrix.columns != 'target']
            y = comb_feature_matrix.loc[:, comb_feature_matrix.columns == 'target'].values.ravel()
            # Convert to ints
            y = [int(i) for i in y]
            # Split train and test data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
            print('X_Train Data:\n', X_train)

            # print(X.columns) # Check for req_process
            ## Build classifier pipeline model - combines NLP features with TFIDF features ##
            self.classifier_pipeline.fit(X_train, y_train)
            pred_y = self.classifier_pipeline.predict(X_test)
            # print('\nPIPE Predictions\n', pred_y)
            cv_score_pipe = cross_val_score(self.classifier_pipeline, X_train, y_train, cv=CV)
            acc_pipe = accuracy_score(y_test, pred_y)
            prec_rec_f1_pipe = precision_recall_fscore_support(y_test, pred_y, average='weighted')
            cv_accuracy_pipe = cv_score_pipe.mean()

            # -------------------------------------------------------------------------------------------------------------------------------------------- #
            # Display results #
            if display:
                print('\n+++ RESULTS OF COMBINED FEATURES +++')
                print('\nModel Accuracy:', '{:.1%}'.format(accuracy_score(y_test, pred_y).round(3)))
                print('\nPrecision: ', '{:.1%}'.format(prec_rec_f1_pipe[0].round(3)), '\nRecall: ',
                      '{:.1%}'.format(prec_rec_f1_pipe[1].round(3)), '\nFscore: ',
                      '{:.1%}'.format(prec_rec_f1_pipe[2].round(3)))
                print("\n=== Confusion Matrix === \n", confusion_matrix(y_test, pred_y))
                print("\n=== Classification Report ===\n", classification_report(y_test, pred_y))
                print('\n=== 10 Fold Cross Validation Scores ===')
                num = 0
                for i in cv_score_pipe:
                    num += 1
                    print('CVFold', num, '=', '{:.1%}'.format(i))
                print("\nMean Cross Validation Score: ", '{:.1%}'.format(cv_accuracy_pipe.round(3)))
                print('\n=== Final Model Results ===')
                print('\nAccuracy:', '{:.1%}'.format(accuracy_score(y_test, pred_y).round(3)))
                print('\nPrecision: ', '{:.1%}'.format(prec_rec_f1_pipe[0].round(3)), '\nRecall: ',
                      '{:.1%}'.format(prec_rec_f1_pipe[1].round(3)), '\nFscore: ',
                      '{:.1%}'.format(prec_rec_f1_pipe[2].round(3)))
                # Confusion Matrix
                conf_mat = confusion_matrix(y_test, pred_y)
                fig, ax = plt.subplots(figsize=(8, 8))
                sns.heatmap(conf_mat, annot=True, fmt='d')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                all_sample_title = 'Accuracy Score: {0}'.format(acc_pipe)
                plt.title(all_sample_title, size=15);
                plt.tight_layout()
                plt.savefig(
                    './Generated_Files/' + str(score_target) + '/Conf_Matrix_Binary_' + str(score_target) + '.png',
                    dpi=300, format='png', bbox_inches='tight')  # use format='svg' or 'pdf' for vectorial pictures
                # plt.show(block=False)
                # plt.pause(3)
                plt.clf()
                plt.close()

        # -------------------------------------------------------------------------------------------------------------------------------------------- #
        ### Perform classification using only NLP features ###
        if nlp_features:
            ## Assigning training and test data ##
            nlp_feature_matrix = pd.DataFrame(self.features_scaled)
            nlp_feature_matrix = nlp_feature_matrix.loc[:, nlp_feature_matrix.columns != 'req_process']

            # Convert to numpy
            y = np.array(nlp_feature_matrix['target'])
            # Convert y values from str to int
            y = y.astype(np.int)
            # axis 1 refers to the columns
            X = nlp_feature_matrix.drop('target', axis=1)
            # Saving feature names
            nlp_feature_names = list(X.columns)
            # print(nlp_feature_names)
            # Convert to numpy array
            X = np.array(X)
            print('Shape of X and y: ', X.shape, y.shape)

            # Split train and test data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
            # print('X_train data:\n', X_train)

            # -------------------------------------------------------------------------------------------------------------------------------------------- #
            ### Implement SMOTE to create new data to balance classes ###
            if smote:
                # summarize class distribution before over sample
                counter = Counter(y_train)
                for k, v in counter.items():
                    per = v / len(y_train) * 100
                    print(k, v, per)
                print('Class=%d, n=%d (%.3f%%)' % (k, v, per))  # NOT WORKING NOW???
                oversample = SMOTE()
                X_train, y_train = oversample.fit_resample(X_train, y_train)
                # summarize class distribution after over sample
                counter = Counter(y_train)
                for k, v in counter.items():
                    per = v / len(y_train) * 100
                    # print(k, v, per)
                    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))

            # -------------------------------------------------------------------------------------------------------------------------------------------- #
            ## Build classifier model without tfidf features ##
            clf.fit(X_train, y_train)
            print('\nClassifier: ', clf)
            # Predict model
            predicted = clf.predict(X_test)
            # Perform cross validation
            cv_score = cross_val_score(clf, X_train, y_train, cv=CV)
            # Store results
            acc = accuracy_score(y_test, predicted)
            prec_rec_f1 = precision_recall_fscore_support(y_test, predicted, average='weighted')
            cv_accuracy = cv_score.mean()
            # -------------------------------------------------------------------------------------------------------------------------------------------- #
            # Display results #
            if display:
                print('\n+++ RESULTS FOR NLP FEATURES +++')
                print('\nModel Accuracy:', '{:.1%}'.format(accuracy_score(y_test, predicted).round(3)))
                print('\nPrecision: ', '{:.1%}'.format(prec_rec_f1[0].round(3)), '\nRecall: ',
                      '{:.1%}'.format(prec_rec_f1[1].round(3)), '\nFscore: ',
                      '{:.1%}'.format(prec_rec_f1[2].round(3)))
                print("\n=== Confusion Matrix === \n", confusion_matrix(y_test, predicted))
                print("\n=== Classification Report ===\n", classification_report(y_test, predicted))
                print('\n=== 10 Fold Cross Validation Scores ===')
                num = 0
                for i in cv_score:
                    num += 1
                    print('CVFold', num, '=', '{:.1%}'.format(i))
                print("\nMean Cross Validation Score: ", '{:.1%}'.format(cv_accuracy.round(3)))
                print('\n=== Final Model Results ===')
                print('\nAccuracy:', '{:.1%}'.format(accuracy_score(y_test, predicted).round(3)))
                print('\nPrecision: ', '{:.1%}'.format(prec_rec_f1[0].round(3)), '\nRecall: ',
                      '{:.1%}'.format(prec_rec_f1[1].round(3)), '\nFscore: ',
                      '{:.1%}'.format(prec_rec_f1[2].round(3)))
                # Confusion Matrix
                conf_mat = confusion_matrix(y_test, predicted)
                fig, ax = plt.subplots(figsize=(8, 8))
                sns.heatmap(conf_mat, annot=True, fmt='d')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                all_sample_title = 'Accuracy Score: {0}'.format(acc)
                plt.title(all_sample_title, size=15);
                plt.tight_layout()
                plt.savefig('./Generated_Files/' + str(score_target) + '/Conf_Matrix_Binary_' + str(
                    score_target) + '.png',
                            dpi=300, format='png',
                            bbox_inches='tight')  # use format='svg' or 'pdf' for vectorial pictures
                # plt.show(block=False)
                # plt.pause(3)
                plt.clf()
                plt.close()

        # -------------------------------------------------------------------------------------------------------------------------------------------- #
        ### TFIDF ###
        if tfidf:
            print('\n===== Tfidf Features =====\n')

            # Create a dataframe with only single requirements
            bi_df = pd.DataFrame()
            bi_df['Requirement'] = self.features['req_process']
            bi_df['Rating'] = binary_df['target']
            bi_df['category_id'] = binary_df['target'].values
            # print(bi_df.head)

            category_id_df = bi_df[['Rating', 'category_id']].sort_values('category_id')
            category_to_id = dict(category_id_df.values)
            id_to_category = dict(category_id_df[['category_id', 'Rating']].values)

            tfidf = TfidfVectorizer(sublinear_tf=True, min_df=min_df, norm='l2', encoding='utf-8', ngram_range=(1, 2), stop_words=ger_stopws)

            features = tfidf.fit_transform(bi_df.Requirement).toarray()
            labels = bi_df.category_id
            print("Each of the %d requirements are represented by %d features (TF-IDF score of unigrams and bigrams)" % (features.shape))
            print('\nShape of tfidf features matrix: ', features.shape)
            # print('\ntfidf features matrix: \n', features)

            # FInd most correlated terms
            N = 5
            # for Rating, category_id in sorted(category_to_id.items()):
            for rating, category_id in sorted(category_to_id.items()):
                features_chi2 = chi2(features, labels == category_id)
                indices = np.argsort(features_chi2[0])
                feature_names = np.array(tfidf.get_feature_names())[indices]
                unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
                bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
                print("\n==> %s:" % (rating))
                print("  * Most Correlated Unigrams are: %s" % (', '.join(unigrams[-N:])))
                print("  * Most Correlated Bigrams are: %s" % (', '.join(bigrams[-N:])))

            # Split test, train and perform tfidf vectorizer
            X_train, X_test, y_train, y_test = train_test_split(bi_df['Requirement'], bi_df['Rating'], random_state=0)

            count_vect = CountVectorizer()
            X_train_counts = count_vect.fit_transform(X_train)
            tfidf_transformer = TfidfTransformer()
            X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

            # Test different classifiers
            models = [
                RandomForestClassifier(n_estimators=200, max_depth=30, random_state=0, class_weight='balanced'),
                svm.SVC(decision_function_shape="ovo", C=10, gamma=0.1, verbose=False, kernel="rbf"),
                LinearSVC(),
                MultinomialNB(),
                LogisticRegression(C=1.0, random_state=42, solver='lbfgs', multi_class='ovr', max_iter=1000),
            ]
            # print('Model = RFC \n', models[0] )

            # index = 0
            # Cross Validation
            cv_df = pd.DataFrame(index=range(CV * len(models)))
            entries = []
            # Find best model
            for model in models:
                model_name = model.__class__.__name__
                accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
                for fold_idx, accuracy in enumerate(accuracies):
                    entries.append((model_name, fold_idx, accuracy))
            cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

            print('\nModel Performance: \n', cv_df.groupby('model_name').accuracy.mean().round(3), '\n')

            X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels,
                                                                                             bi_df.index,
                                                                                             test_size=0.20,
                                                                                             random_state=0)
            # Fit the best model
            models[idx].fit(X_train, y_train)
            y_pred = models[idx].predict(X_test)
            # acc = np.mean(y_test == y_pred)
            acc = accuracy_score(y_pred, y_test)
            prec_rec_f1 = precision_recall_fscore_support(y_test, y_pred, average='weighted')

            # Display
            if display:
                # Classification report
                print('\t\t\t\tCLASSIFICATION METRICS\n')
                print(metrics.classification_report(y_test, y_pred, target_names=bi_df['Rating'].unique()))
                print('\n--- TFIDF Results ---')
                print('Accuracy: ', '{:.1%}'.format(accuracy_score(y_test.values.ravel(), y_pred).round(3)))
                # print("\nPrec_Recall_FScore: \n", precision_recall_fscore_support(y_test, y_pred, average='weighted'))
                print('\nPrecision: ', '{:.1%}'.format(prec_rec_f1[0].round(3)), '\nRecall: ',
                      '{:.1%}'.format(prec_rec_f1[1].round(3)), '\nFscore: ', '{:.1%}'.format(prec_rec_f1[2].round(3)))

                print("\n=== Confusion Matrix === \n", confusion_matrix(y_test, y_pred))

                conf_mat = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(8, 8))
                sns.heatmap(conf_mat, annot=True, fmt='d')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                all_sample_title = 'Accuracy Score: {0}'.format(acc)
                plt.title(all_sample_title, size=15)
                plt.tight_layout()
                plt.savefig(
                    './Generated_Files/' + str(score_target) + '/Conf_Matrix_TFIDF_Binary_' + str(score_target) + '.png',
                    dpi=300, format='png', bbox_inches='tight')  # use format='svg' or 'pdf' for vectorial pictures
                plt.show(block=False)
                plt.pause(5)
                plt.clf()
                plt.close()

            # Feature Importance when using Random Forest
            if idx == 0:
                feat_importance = models[0].feature_importances_
                # print(feat_importance)
                feat_names = tfidf.get_feature_names()
                # print(feat_names)
                df_feat_importance = pd.DataFrame()
                df_feat_importance['feat'] = feat_names
                df_feat_importance['importance'] = feat_importance
                print('\n--- Top TFIDF features ---\n',
                      df_feat_importance.sort_values(by='importance', ascending=False).head(), '\n')
        # -------------------------------------------------------------------------------------------------------------------------------------------- #

    ''' Build a model to classify the requirements into 3 categories (-1, 0, 1) '''
    def tri_classification(self, score_target, combine_features, nlp_features, tfidf, smote, display):
        # global clf
        ''' *** Something is either leaking from binary or not changed in tri as the results are incorrect when run same time as binary '''
        print('\t\t\t\t *** TRI CLASS CLASSIFICATION ***\n')
        # -------------------------------------------------------------------------------------------------------------------------------------------- #
        # Build  Model
        if classifier == "RFC":
            print('\n+++ RANDOM FOREST +++\n')
            clf = RandomForestClassifier(n_estimators=200, max_depth=30, random_state=0)
        if classifier == "LogReg":
            print('\n+++ LOGISTIC REGRESSION +++\n')
            clf = LogisticRegression(C=1.0, random_state=42, solver='lbfgs', multi_class='ovr', max_iter=1000)
        if classifier == "SVM":
            print('\n+++ SUPPORT VECTORE MACHINE +++\n')
            clf = svm.SVC(decision_function_shape="ovo", C=10, gamma=0.1, verbose=False, kernel="rbf")
        if classifier == "MNB":
            clf = GaussianNB()
            print('\n+++ MULTINOMIAL NAIVE BAYES +++\n')
        # -------------------------------------------------------------------------------------------------------------------------------------------- #
        # Convert into 3 Classes
        tri_class_df = self.features.copy()  # use for load/save matrix
        # tri_class_df = tri_class_df.drop('req_process', axis=1)
        # tri_df['target'] = self.data['target']
        tri_class_df['target'] = self.features['target'].astype(str)
        tri_class_df['target'] = tri_class_df.target.replace(['1', '2'], '-1')
        tri_class_df['target'] = tri_class_df.target.replace(['3'], '0')
        tri_class_df['target'] = tri_class_df.target.replace(['4', '5'], '1')

        nlp_feature_matrix = pd.DataFrame(self.features)
        # nlp_feature_matrix = nlp_feature_matrix.loc[:, nlp_feature_matrix.columns != 'req_process']

        nlp_feature_matrix = tri_class_df.copy()
        # features['req_process'] = self.data['req_process']
        print('No of requirements: ', nlp_feature_matrix.shape[0])
        nlp_feature_matrix = nlp_feature_matrix.dropna()
        # print('NLP feature df:\n', nlp_feature_matrix.head())

        # ------------------------------------------------------------------------------------------------------------ #
        ### Use Pipeline to combine both tfidf and nlp features and build classifier model ###
        if combine_features:
            ## Assigning training and test data ##
            comb_feature_matrix = pd.DataFrame(nlp_feature_matrix)
            # comb_feature_matrix['req_process'] = self.features['req_process']
            # print('Combined Feature Matrix DF:\n', comb_feature_matrix)
            # Assigning X, y
            X = comb_feature_matrix.loc[:, comb_feature_matrix.columns != 'target']
            y = comb_feature_matrix.loc[:, comb_feature_matrix.columns == 'target'].values.ravel()
            # Convert to ints
            y = [int(i) for i in y]
            # Split train and test data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
            # print('X_Train Data:\n', X_train)

            # print(X.columns)  # Check for req_process
            ## Build classifier pipeline model - combines NLP features with TFIDF features ##
            self.classifier_pipeline.fit(X_train, y_train)
            pred_y = self.classifier_pipeline.predict(X_test)
            # print('\nPIPE Predictions\n', pred_y)
            cv_score_pipe = cross_val_score(self.classifier_pipeline, X_train, y_train, cv=CV)
            acc_pipe = accuracy_score(y_test, pred_y)
            prec_rec_f1_pipe = precision_recall_fscore_support(y_test, pred_y, average='weighted')
            cv_accuracy_pipe = cv_score_pipe.mean()

            # -------------------------------------------------------------------------------------------------------------------------------------------- #
            # Display results #
            if display:
                print('\n+++ RESULTS OF COMBINED FEATURES +++')
                print('\nModel Accuracy:', '{:.1%}'.format(accuracy_score(y_test, pred_y).round(3)))
                print('\nPrecision: ', '{:.1%}'.format(prec_rec_f1_pipe[0].round(3)), '\nRecall: ',
                      '{:.1%}'.format(prec_rec_f1_pipe[1].round(3)), '\nFscore: ',
                      '{:.1%}'.format(prec_rec_f1_pipe[2].round(3)))
                print("\n=== Confusion Matrix === \n", confusion_matrix(y_test, pred_y))
                print("\n=== Classification Report ===\n", classification_report(y_test, pred_y))
                print('\n=== 10 Fold Cross Validation Scores ===')
                num = 0
                for i in cv_score_pipe:
                    num += 1
                    print('CVFold', num, '=', '{:.1%}'.format(i))
                print("\nMean Cross Validation Score: ", '{:.1%}'.format(cv_accuracy_pipe.round(3)))
                print('\n=== Final Model Results ===')
                print('\nAccuracy:', '{:.1%}'.format(accuracy_score(y_test, pred_y).round(3)))
                print('\nPrecision: ', '{:.1%}'.format(prec_rec_f1_pipe[0].round(3)), '\nRecall: ',
                      '{:.1%}'.format(prec_rec_f1_pipe[1].round(3)), '\nFscore: ',
                      '{:.1%}'.format(prec_rec_f1_pipe[2].round(3)))
                # Confusion Matrix
                conf_mat = confusion_matrix(y_test, pred_y)
                fig, ax = plt.subplots(figsize=(8, 8))
                sns.heatmap(conf_mat, annot=True, fmt='d')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                all_sample_title = 'Accuracy Score: {0}'.format(acc_pipe)
                plt.title(all_sample_title, size=15);
                plt.tight_layout()
                plt.savefig(
                    './Generated_Files/' + str(score_target) + '/Conf_Matrix_Tri_' + str(score_target) + '.png',
                    dpi=300, format='png', bbox_inches='tight')  # use format='svg' or 'pdf' for vectorial pictures
                # plt.show(block=False)
                # plt.pause(3)
                plt.clf()
                plt.close()

        # -------------------------------------------------------------------------------------------------------------------------------------------- #
        ### Perform classification using only NLP features ###
        if nlp_features:
            ## Assigning training and test data ##
            nlp_feature_matrix = pd.DataFrame(self.features_scaled)
            nlp_feature_matrix = nlp_feature_matrix.loc[:, nlp_feature_matrix.columns != 'req_process']

            # Convert to numpy
            y = np.array(nlp_feature_matrix['target'])
            # Convert y values from str to int
            y = y.astype(np.int)
            # axis 1 refers to the columns
            X = nlp_feature_matrix.drop('target', axis=1)
            # Saving feature names
            nlp_feature_names = list(X.columns)
            # print(nlp_feature_names)
            # Convert to numpy array
            X = np.array(X)
            print('Shape of X and y: ', X.shape, y.shape)

            # Split train and test data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
            # print('X_train data:\n', X_train)

            # -------------------------------------------------------------------------------------------------------------------------------------------- #
            ### Implement SMOTE to create new data to balance classes ###
            if smote:
                # summarize class distribution before over sample
                counter = Counter(y_train)
                for k, v in counter.items():
                    per = v / len(y_train) * 100
                    print(k, v, per)
                print('Class=%d, n=%d (%.3f%%)' % (k, v, per))  # NOT WORKING NOW???
                oversample = SMOTE()
                X_train, y_train = oversample.fit_resample(X_train, y_train)
                # summarize class distribution after over sample
                counter = Counter(y_train)
                for k, v in counter.items():
                    per = v / len(y_train) * 100
                    # print(k, v, per)
                    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))

            # -------------------------------------------------------------------------------------------------------------------------------------------- #
            ## Build classifier model without tfidf features ##
            clf.fit(X_train, y_train)
            print('Classifier: ', clf)
            # Predict model
            predicted = clf.predict(X_test)
            # Perform cross validation
            cv_score = cross_val_score(clf, X_train, y_train, cv=CV)
            # Store results
            acc = accuracy_score(y_test, predicted)
            prec_rec_f1 = precision_recall_fscore_support(y_test, predicted, average='weighted')
            cv_accuracy = cv_score.mean()
            # -------------------------------------------------------------------------------------------------------------------------------------------- #
            # Display results #
            if display:
                print('\n+++ RESULTS FOR NLP FEATURES +++')
                print('\nModel Accuracy:', '{:.1%}'.format(accuracy_score(y_test, predicted).round(3)))
                print('\nPrecision: ', '{:.1%}'.format(prec_rec_f1[0].round(3)), '\nRecall: ',
                      '{:.1%}'.format(prec_rec_f1[1].round(3)), '\nFscore: ',
                      '{:.1%}'.format(prec_rec_f1[2].round(3)))
                print("\n=== Confusion Matrix === \n", confusion_matrix(y_test, predicted))
                print("\n=== Classification Report ===\n", classification_report(y_test, predicted))
                print('\n=== 10 Fold Cross Validation Scores ===')
                num = 0
                for i in cv_score:
                    num += 1
                    print('CVFold', num, '=', '{:.1%}'.format(i))
                print("\nMean Cross Validation Score: ", '{:.1%}'.format(cv_accuracy.round(3)))
                print('\n=== Final Model Results ===')
                print('\nAccuracy:', '{:.1%}'.format(accuracy_score(y_test, predicted).round(3)))
                print('\nPrecision: ', '{:.1%}'.format(prec_rec_f1[0].round(3)), '\nRecall: ',
                      '{:.1%}'.format(prec_rec_f1[1].round(3)), '\nFscore: ',
                      '{:.1%}'.format(prec_rec_f1[2].round(3)))
                # Confusion Matrix
                conf_mat = confusion_matrix(y_test, predicted)
                fig, ax = plt.subplots(figsize=(8, 8))
                sns.heatmap(conf_mat, annot=True, fmt='d')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                all_sample_title = 'Accuracy Score: {0}'.format(acc)
                plt.title(all_sample_title, size=15);
                plt.tight_layout()
                plt.savefig('./Generated_Files/' + str(score_target) + '/Conf_Matrix_Binary_' + str(
                    score_target) + '.png',
                            dpi=300, format='png',
                            bbox_inches='tight')  # use format='svg' or 'pdf' for vectorial pictures
                # plt.show(block=False)
                # plt.pause(3)
                plt.clf()
                plt.close()

        # -------------------------------------------------------------------------------------------------------------------------------------------- #
        ### TFIDF ###
        if tfidf:
            print('\n===== Tfidf Features =====\n')

            # Create a dataframe with only single requirements
            tri_df = pd.DataFrame()
            tri_df['Requirement'] = self.features['req_process']
            tri_df['Rating'] = tri_class_df['target']
            tri_df['category_id'] = tri_class_df['target'].values
            # print(tri_df.head)

            category_id_df = tri_df[['Rating', 'category_id']].sort_values('category_id')
            category_to_id = dict(category_id_df.values)
            id_to_category = dict(category_id_df[['category_id', 'Rating']].values)

            tfidf = TfidfVectorizer(sublinear_tf=True, min_df=min_df, norm='l2', encoding='utf-8', ngram_range=(1, 2),
                                    stop_words=ger_stopws)

            features = tfidf.fit_transform(tri_df.Requirement).toarray()
            labels = tri_df.category_id
            print(
                "Each of the %d requirements are represented by %d features (TF-IDF score of unigrams and bigrams)" % (
                    features.shape))
            print('\nShape of tfidf features matrix: ', features.shape)
            # print('\ntfidf features matrix: \n', features)

            # FInd most correlated terms
            N = 5
            # for Rating, category_id in sorted(category_to_id.items()):
            for rating, category_id in sorted(category_to_id.items()):
                features_chi2 = chi2(features, labels == category_id)
                indices = np.argsort(features_chi2[0])
                feature_names = np.array(tfidf.get_feature_names())[indices]
                unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
                bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
                print("\n==> %s:" % (rating))
                print("  * Most Correlated Unigrams are: %s" % (', '.join(unigrams[-N:])))
                print("  * Most Correlated Bigrams are: %s" % (', '.join(bigrams[-N:])))

            # Split test, train and perform tfidf vectorizer
            X_train, X_test, y_train, y_test = train_test_split(tri_df['Requirement'], tri_df['Rating'], random_state=0)

            count_vect = CountVectorizer()
            X_train_counts = count_vect.fit_transform(X_train)
            tfidf_transformer = TfidfTransformer()
            X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

            # Test different classifiers
            models = [
                RandomForestClassifier(n_estimators=200, max_depth=30, random_state=0, class_weight='balanced'),
                svm.SVC(decision_function_shape="ovo", C=10, gamma=0.1, verbose=False, kernel="rbf"),
                MultinomialNB(),
                LogisticRegression(C=1.0, random_state=42, solver='lbfgs', multi_class='ovr', max_iter=1000),
            ]
            print('Model = \n', models[idx])

            # Cross Validation
            cv_df = pd.DataFrame(index=range(CV * len(models)))
            entries = []
            # Find best model
            for model in models:
                model_name = model.__class__.__name__
                accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
                for fold_idx, accuracy in enumerate(accuracies):
                    entries.append((model_name, fold_idx, accuracy))
            cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

            print('\nModel Performance: \n', cv_df.groupby('model_name').accuracy.mean().round(3), '\n')

            X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels,
                                                                                             tri_df.index,
                                                                                             test_size=0.20,
                                                                                             random_state=0)
            # Fit the best model
            models[idx].fit(X_train, y_train)
            y_pred = models[idx].predict(X_test)
            # acc = np.mean(y_test == y_pred)
            acc = accuracy_score(y_pred, y_test)
            prec_rec_f1 = precision_recall_fscore_support(y_test, y_pred, average='weighted')

            # Display
            if display:
                # Classification report
                print('\t\t\t\tCLASSIFICATION METRICS\n')
                print(metrics.classification_report(y_test, y_pred, target_names=tri_df['Rating'].unique()))
                print('\n--- TFIDF Results ---')
                print('Accuracy: ', '{:.1%}'.format(accuracy_score(y_test.values.ravel(), y_pred).round(3)))
                # print("\nPrec_Recall_FScore: \n", precision_recall_fscore_support(y_test, y_pred, average='weighted'))
                print('\nPrecision: ', '{:.1%}'.format(prec_rec_f1[0].round(3)), '\nRecall: ',
                      '{:.1%}'.format(prec_rec_f1[1].round(3)), '\nFscore: ', '{:.1%}'.format(prec_rec_f1[2].round(3)))

                print("\n=== Confusion Matrix === \n", confusion_matrix(y_test, y_pred))

                conf_mat = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(8, 8))
                sns.heatmap(conf_mat, annot=True, fmt='d')
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                all_sample_title = 'Accuracy Score: {0}'.format(acc)
                plt.title(all_sample_title, size=15)
                plt.tight_layout()
                plt.savefig(
                    './Generated_Files/' + str(score_target) + '/Conf_Matrix_TFIDF_Tri_' + str(score_target) + '.png',
                    dpi=300, format='png', bbox_inches='tight')  # use format='svg' or 'pdf' for vectorial pictures
                plt.show(block=False)
                plt.pause(5)
                plt.clf()
                plt.close()

            # Feature Importance when using Random Forest
            if idx == 0:
                feat_importance = models[0].feature_importances_
                # print(feat_importance)
                feat_names = tfidf.get_feature_names()
                # print(feat_names)
                df_feat_importance = pd.DataFrame()
                df_feat_importance['feat'] = feat_names
                df_feat_importance['importance'] = feat_importance
                print('\n--- Top TFIDF features ---\n',
                      df_feat_importance.sort_values(by='importance', ascending=False).head(), '\n')

        # -------------------------------------------------------------------------------------------------------------------------------------------- #

    ''' A function to test model performance using only tfidf features '''
    def tfidf_features(self, score_target):
        # Only works when creating new features matrix (i.e not to be used when loading matrix)
        # https://datascience.stackexchange.com/questions/22813/using-tf-idf-with-other-features-in-sklearn
        # https://stackoverflow.com/questions/48573174/how-to-combine-tfidf-features-with-other-features

        ''' The idea is to test whether the model performs better when using only the natural features found in the data - the tfidf featureds
            compared to using the NLP features.

            RESULTS: There seems to be no improvement in model performance when using only TFIDF features.

            * create a feature matrix using tfidf?
        '''

        print('\n===== Tfidf Features =====\n')

        # Create a dataframe with only single requirements
        df = pd.DataFrame()
        df['Requirement'] = self.features['req_process']  # Should be req_process?
        df['Rating'] = self.features['target']
        df['category_id'] = self.features['target'].values
        print(df.head)

        category_id_df = df[['Rating', 'category_id']].sort_values('category_id')
        # category_id_df = df[['Requirement', 'category_id']].sort_values('category_id')
        category_to_id = dict(category_id_df.values)
        id_to_category = dict(category_id_df[['category_id', 'Rating']].values)
        # id_to_category = dict(category_id_df[['category_id', 'Requirement']].values)

        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=min_df, norm='l2', encoding='utf-8', ngram_range=(1, 2),
                                stop_words=ger_stopws)

        features = tfidf.fit_transform(df.Requirement).toarray()
        labels = df.category_id
        print("Each of the %d requirements are represented by %d features (TF-IDF score of unigrams and bigrams)"
              % (features.shape))
        print('\nShape of tfidf features matrix: ', features.shape)
        print('\ntfidf features matrix: \n', features)

        # FInd most correlated terms
        N = 5
        # for Rating, category_id in sorted(category_to_id.items()):
        for rating, category_id in sorted(category_to_id.items()):
            features_chi2 = chi2(features, labels == category_id)
            indices = np.argsort(features_chi2[0])
            feature_names = np.array(tfidf.get_feature_names())[indices]
            unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
            bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
            print("\n==> %s:" % (rating))
            print("  * Most Correlated Unigrams are: %s" % (', '.join(unigrams[-N:])))
            print("  * Most Correlated Bigrams are: %s" % (', '.join(bigrams[-N:])))

        # Split test, train and perform tfidf vectorizer
        X_train, X_test, y_train, y_test = train_test_split(df['Requirement'], df['Rating'], random_state=0)

        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(X_train)
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

        # Test different classifiers
        models = [
            RandomForestClassifier(n_estimators=200, max_depth=30, random_state=0, class_weight='balanced'),
            svm.SVC(decision_function_shape="ovo", C=10, gamma=0.1, verbose=False, kernel="rbf"),
            LinearSVC(),
            MultinomialNB(),
            LogisticRegression(C=1.0, random_state=42, solver='lbfgs', multi_class='ovr', max_iter=1000),
        ]
        # print('Model = RFC \n', models[0] )
        index = 0
        # Cross Validation
        cv_df = pd.DataFrame(index=range(CV * len(models)))
        entries = []
        # Find best model
        for model in models:
            model_name = model.__class__.__name__
            accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
            for fold_idx, accuracy in enumerate(accuracies):
                entries.append((model_name, fold_idx, accuracy))
        cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

        print('\nModel Performance: \n', cv_df.groupby('model_name').accuracy.mean().round(3), '\n')

        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index,
                                                                                         test_size=0.20, random_state=0)
        # Fit the best model
        models[index].fit(X_train, y_train)
        y_pred = models[index].predict(X_test)

        # Classification report
        print('\t\t\t\tCLASSIFICATION METRICS\n')
        print(metrics.classification_report(y_test, y_pred,
                                            target_names=df['Rating'].unique()))

        # acc = np.mean(y_test == y_pred)
        acc = accuracy_score(y_pred, y_test)
        prec_rec_f1 = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        print('\n--- TFIDF Results ---')
        print('Accuracy: ', '{:.1%}'.format(accuracy_score(y_test.values.ravel(), y_pred).round(3)))
        # print("\nPrec_Recall_FScore: \n", precision_recall_fscore_support(y_test, y_pred, average='weighted'))
        print('\nPrecision: ', '{:.1%}'.format(prec_rec_f1[0].round(3)), '\nRecall: ',
              '{:.1%}'.format(prec_rec_f1[1].round(3)), '\nFscore: ', '{:.1%}'.format(prec_rec_f1[2].round(3)))

        print("\n=== Confusion Matrix === \n", confusion_matrix(y_test, y_pred))

        # Feature Importance when using Random Forest
        if index == 0:
            feat_importance = models[0].feature_importances_
            # print(feat_importance)
            feat_names = tfidf.get_feature_names()
            # print(feat_names)
            df_feat_importance = pd.DataFrame()
            df_feat_importance['feat'] = feat_names
            df_feat_importance['importance'] = feat_importance
            print('\n--- Top TFIDF features ---\n',
                  df_feat_importance.sort_values(by='importance', ascending=False).head(), '\n')

        # Display
        conf_mat = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(conf_mat, annot=True, fmt='d')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        all_sample_title = 'Accuracy Score: {0}'.format(acc)
        plt.title(all_sample_title, size=15)
        plt.tight_layout()
        plt.savefig('./Generated_Files/' + str(score_target) + '/Conf_Matrix_TFIDF_Multi_' + str(score_target) + '.png',
                    dpi=300, format='png', bbox_inches='tight')  # use format='svg' or 'pdf' for vectorial pictures
        plt.show(block=False)
        plt.pause(5)
        plt.clf()
        plt.close()

        '''
        # Create a dataframe with only single requirements
        df = pd.DataFrame()
        df['Requirement'] = self.features['req_process']  # Should be req_process?
        df['Rating'] = self.features['target']
        df['category_id'] = self.features['target'].values
        print('TFIDF Dataframe:\n', df.head)

        category_id_df = df[['Rating', 'category_id']].sort_values('category_id')
        category_to_id = dict(category_id_df.values)
        # print('category ids (dict: req:rating): \n', category_to_id)
        id_to_category = dict(category_id_df[['category_id', 'Rating']].values)
        # print('ID to category (dict: rating:req):\n', id_to_category)

        print(df.head())

        fig = plt.figure(figsize=(8, 6))
        colors = ['grey', 'grey', 'darkblue', 'darkblue', 'darkblue']
        df.groupby('category_id').Requirement.count().sort_values().plot.barh(
            ylim=0, color=colors, title='NUMBER OF Requirements IN EACH rating CATEGORY\n')
        plt.xlabel('Number of ocurrences', fontsize=10)
        plt.show()

        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=min_df, norm='l2', encoding='utf-8', ngram_range=(1, 2),
                                stop_words=ger_stopws)

        # Transform each requirement into a vector
        features = tfidf.fit_transform(df.Requirement).toarray()
        labels = df.category_id
        print("Each of the %d requirements are represented by %d features (TF-IDF score of unigrams and bigrams)" % (
            features.shape))
        print('\nShape of tfidf features matrix: ', features.shape)
        print('\ntfidf features matrix: \n', features)

        # FInd 3 most correlated terms
        N = 5
        # for Rating, category_id in sorted(category_to_id.items()):
        for rating, category_id in sorted(category_to_id.items()):
            features_chi2 = chi2(features, labels == category_id)
            indices = np.argsort(features_chi2[0])
            feature_names = np.array(tfidf.get_feature_names())[indices]
            unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
            bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
            print("\n==> %s:" % (rating))
            print("  * Most Correlated Unigrams are: %s" % (', '.join(unigrams[-N:])))
            print("  * Most Correlated Bigrams are: %s" % (', '.join(bigrams[-N:])))

        # Split test, train and perform tfidf vectorizer
        X = df['Requirement']
        y = df['Rating']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(X_train)
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
        # print(X_train_tfidf)

        # Test different classifiers
        models = [
            RandomForestClassifier(n_estimators=200, max_depth=30, random_state=0, class_weight='balanced'),
            svm.SVC(decision_function_shape="ovo", C=10, gamma=0.1, verbose=False, kernel="rbf"),
            LinearSVC(),
            MultinomialNB(),
            LogisticRegression(C=1.0, random_state=42, solver='lbfgs', multi_class='ovr', max_iter=1000),
        ]

        index = 3
        # Cross Validation
        cv_df = pd.DataFrame(index=range(CV * len(models)))
        entries = []
        # Find best model
        for model in models:
            model_name = model.__class__.__name__
            accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
            for fold_idx, accuracy in enumerate(accuracies):
                entries.append((model_name, fold_idx, accuracy))
        cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

        # print('\nModel Performance: \n', cv_df.groupby('model_name').accuracy.mean().round(3), '\n')

        mean_accuracy = cv_df.groupby('model_name').accuracy.mean()
        std_accuracy = cv_df.groupby('model_name').accuracy.std()
        acc = pd.concat([mean_accuracy, std_accuracy], axis=1,
                        ignore_index=True)
        acc.columns = ['Mean Accuracy', 'Standard deviation']
        print('ACCURACY: ', acc)

        # Display Box plot
        plt.figure(figsize=(8, 5))
        sns.boxplot(x='model_name', y='accuracy',
                    data=cv_df,
                    color='lightblue',
                    showmeans=True)
        plt.title("MEAN ACCURACY (cv = 10)\n", size=14)
        plt.show()

        models[index].fit(X_train, y_train)
        y_pred = models[index].predict(X_test)
        # acc = np.mean(y_test == y_pred)
        acc = accuracy_score(y_pred, y_test)
        prec_rec_f1 = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        print('\n--- TFIDF Results for *Best Model* ---')
        print('Accuracy: ', '{:.1%}'.format(accuracy_score(y_test.values.ravel(), y_pred).round(3)))
        # print("\nPrec_Recall_FScore: \n", precision_recall_fscore_support(y_test, y_pred, average='weighted'))
        print('\nPrecision: ', '{:.1%}'.format(prec_rec_f1[0].round(3)), '\nRecall: ',
              '{:.1%}'.format(prec_rec_f1[1].round(3)), '\nFscore: ', '{:.1%}'.format(prec_rec_f1[2].round(3)))


        # Model Evaluation
        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index,
                                                                                         test_size=0.20, random_state=0)
        # Fit best model
        model = MultinomialNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Classification report
        print('\t\t\t\tCLASSIFICATION METRICS\n')
        print(metrics.classification_report(y_test, y_pred,
                                            target_names=df['Rating'].unique()))

        # Feature Importance when using Random Forest
        if index == 0:
            feat_importance = models[0].feature_importances_
            # print(feat_importance)
            feat_names = tfidf.get_feature_names()
            # print(feat_names)
            df_feat_importance = pd.DataFrame()
            df_feat_importance['feat'] = feat_names
            df_feat_importance['importance'] = feat_importance
            print('\n--- Top TFIDF features ---\n',
                  df_feat_importance.sort_values(by='importance', ascending=False).head(), '\n')

        # Display
        conf_mat = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(conf_mat, annot=True, fmt='d')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        all_sample_title = 'Accuracy Score: {0}'.format(acc)
        plt.title(all_sample_title, size=15)
        plt.tight_layout()
        plt.savefig('./Generated_Files/' + str(score_target) + '/Conf_Matrix_TFIDF_Multi_' + str(score_target) + '.png',
                    dpi=300, format='png', bbox_inches='tight')  # use format='svg' or 'pdf' for vectorial pictures
        plt.show(block=False)
        plt.pause(5)
        plt.clf()
        plt.close()

        model.fit(features, labels)

        
        N = 4
        for Rating, category_id in sorted(category_to_id.items()):
            indices = np.argsort(model.coef_[category_id])
            feature_names = np.array(tfidf.get_feature_names())[indices]
            unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
            bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
            print("\n==> '{}':".format(Rating))
            print("  * Top unigrams: %s" % (', '.join(unigrams)))
            print("  * Top bigrams: %s" % (', '.join(bigrams)))
        #
        # for predicted in category_id_df.category_id:
        #     for actual in category_id_df.category_id:
        #         if predicted != actual and conf_mat[actual, predicted] >= 5:
        #             print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual],
        #                                                                  id_to_category[predicted],
        #                                                                  conf_mat[actual, predicted]))
        #
        #             print(df.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['Product','Consumer_complaint']])
        #             print('')

        '''

    def test_function(self, score_target):
        print(self.features)
        print(self.dfm)
        print(self.data)


if __name__ == '__main__':
    # file = u'/Users/selina/Code/Python/Thesis_Code/software_requirements_small.xlsx'
    # file = u'/Users/selina/Documents/UNI/Thesis/Code/software_requirements_full.xlsx'
    # file = u'/Users/selina/Code/Python/Thesis/src/software_requirements_small.xlsx'
    file = u'/Users/selina/Code/Python/Thesis/src/software_requirements_full.xlsx'
    score_target = 'score'
    extractor = Specification_Evaluation(file)
    # Load saved features_matrix once model has been run
    # extractor.load_features_matrix(score_target)
    # '''
    extractor.get_data(score_target)
    extractor.get_features(score_target, metric_create=True, export_dfm=True, hotenc=True)
    extractor.preprocessing(score_target)
    extractor.combine_features(score_target)
    extractor.save_features_matrix(score_target)
    # '''
    extractor.combine_features(score_target)
    # extractor.multi_class_classification(score_target, tfidf=True, smote=True, combine_features=True, nlp_features=True, display=True, save_model=False)
    extractor.binary_classification(score_target, combine_features=True, nlp_features=True, smote=False, tfidf=True, display=True)
    # extractor.tri_classification(score_target, combine_features=True, nlp_features=True, smote=False, tfidf=True, display=True)
    # extractor.tfidf_features(score_target)
    # extractor.classifier_comparison(score_target)
    # extractor.grid_search(score_target)
    # extractor.test_function(score_target)
