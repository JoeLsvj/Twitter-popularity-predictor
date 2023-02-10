print("predicting tweets' popularity...")

### Import the standard modules

#from modules import *
# Import the classes built for twitter:
from twitter_classes import *

### Authentication:

api = authentication().api()
client = authentication().client()

### Creation of the Dataset:

# Initialize the query class object:
query = query(dictionary=food_dict, number_of_tokens=2, context=context)
# Call the functions to make the queries
query_api = query.make_query_api()
query_client = query.make_query_client()

# Initialize the text_mining class object, passing the instantiation of the query class.
# Then create the dataset, and clean the tweets.
miner = text_mining(query=query)
dataset = miner.create_dataset_scraper()
# expand, clean, and save to csv
dataset.to_csv(path_or_buf="./raw_dataset_1.csv")
miner.expand()
dataset_clean = miner.clean()
dataset.to_csv(path_or_buf="./clean_dataset_1.csv")

# Import an existing clean dataset:
#dataset_clean = pd.read_csv('combined-files.csv')
#dataset_clean = dataset_clean.drop('Unnamed: 19', axis = 1)
#display(dataset_clean)

# Further cleaning, and eventually selecting for the mediatic impact of the author.
# eventually remove the tweets with <2 retweets.
for i in range(len(dataset_clean)):
    if (dataset_clean['retweets'][i] < 2):
        dataset_clean = dataset_clean.drop(i, axis = 0)
dataset_clean = dataset_clean.reset_index(drop = True)
# Also, interpreting the popularity of a tweet as the success (this last thing is more interesting from the point of view 
# of the practical usage for a corporate client), we can take only the tweets belonging to a certain mediatic impact (measured
# with the number of followers or other indexes). In this case we can take a range of followers, and measure the regression metric 
# in base of these tweets. In this case having the benchmark with the dummy regressor (which returns the mean) can be very useful.
mediatic_impact = 10000
for i in range(len(dataset_clean)):
    if (dataset_clean['followers'][i] > mediatic_impact + mediatic_impact*0.3 or dataset_clean['followers'][i] < mediatic_impact - mediatic_impact*0.3):
        dataset_clean = dataset_clean.drop(i, axis = 0)
dataset_clean = dataset_clean.reset_index(drop = True)
#display(dataset_clean)

### Features analysis:

# Create the scatter-plot matrix
dataset_plot = dataset_clean.drop(['date', 'author', 'text', 'favourites', 'status', 'quotes', 'char_count', 'word_count', 'stop_words'], axis=1)
scatter_matrix = pd.plotting.scatter_matrix(dataset_plot, figsize=(20,20))
plt.savefig("./scatter_matrix.png")
# Create the scatter-plot matrix with the actual used features, with the applied transformations
dataset_plot['target'] = np.log((dataset_plot['retweets'] + dataset_plot['replies']).to_numpy())
dataset_plot = dataset_plot.drop(['retweets', 'replies', 'URLs'], axis = 1)
dataset_plot['followers'] = np.log(dataset_plot['followers'].to_numpy())
scatter_matrix = pd.plotting.scatter_matrix(dataset_plot, figsize=(16,16))
plt.savefig("./scatter_matrix_log.png")

### Cluster analysis: Unsupervised learning techniques to analyze the quality of imported tweets:

#clusters = cluster(dataset=dataset_clean['text'])
#clusters.NMF()
#clusters.LDA()

### TRAINING AND ASSESSING SECTION:

# Create a function of the class to insert these stuff. Call it "make_features_target"
target_data = dataset_clean[['retweets', 'replies', 'quotes']]
features_data = dataset_clean.drop(["date", "author", "retweets", "replies", "quotes", "status", "favourites", "char_count", "word_count", "stop_words"], axis=1)

# Taking the log also of the followers. As we can see from the scatterplot matrix, also the distribution of the followers count is skewed to the left. 
features_data['followers'] = np.log(features_data['followers'].to_numpy())
print(target_data.shape, features_data.shape)

# instantiate the vectorizer as an object of the class `TfidfVectorizer`, and assign it to the varibale vectorizer.
vectorizer = TfidfVectorizer(max_df=0.95, min_df=1, stop_words=None, ngram_range=(2,5))

# One may have the crazy idea to use tfidf also on the emojis. This can be particularly useful with tweets with a lot of emojis.
#vectorizer_emojis = TfidfVectorizer(ngram_range=(1,5), stop_words = None, token_pattern=r'[^\s]')

# Now try to build some more sofisticated thing, using column transform. This is now a sort of obliged step, since we have several columns of different features. 
# The main problem is that the features related to the pure text, that are the tf-idf features, are created by the sklearn with an object of the type csx_sparse matrix of 
# scipy. This object is a unique block: cannot be divided in rows of a dataset, because looses the type of spoarse matrix (becomes a list of series objects) and the LT 
# of sklearn complains (wants pandas dataframe, sparse matrix or np arrays). So, the possible choices are 2: or trasform all the other columns of features into sparse objects and adding 
# to the tf-idf matrix, or using the column transformer and a pipeline. This second option allows also to normalize all the columns, that is a quite common procedure in text mining. 
normalize = Normalizer()
standard_scaler = StandardScaler()
hot_encoder = OneHotEncoder()
# Build the trasformation:
transformer = ColumnTransformer(
    [
        ('tfidf', vectorizer, 'text'),
        #('tfidfEmotes', vectorizer_emojis, 'emojis'),
        #('normalize', normalize, ['followers']),
        #('categorical_encoding', hot_encoder, ['media', '#tag', '@', 'numaric', 'upercase', 'URLs'])
    ], remainder = standard_scaler)

# Create the features:
features = transformer.fit_transform(features_data)

# This is only to see the dictionary of words, used for tfidf features.
vectorizer.fit_transform(features_data['text'])
print(vectorizer.get_feature_names_out())

# Create the target, and apply some transformations:
target = target_data['retweets'] + target_data['replies'] #+ target_data['quotes']
# Explore the distribution of the targets (non transformed)
target.plot.hist(grid=True, bins=60, rwidth=0.8, color='#607c8e')
# Since the distribution of the targets is very skewed to the left, take a log transformation, 
# in order to make the distribution more bell-shaped.
target = np.log(target)
target.plot.hist(grid=True, bins=60, rwidth=0.8, color='#607c8e')
# The situation is in fact impreved really well. So, in the further part of the program, 
# the targets are considered to be transformed with the logarithm.

## Actual training:
models_list = [
    DummyRegressor(),
    RandomForestRegressor(),   # better with the test bag
    svm.SVR(kernel = 'poly', coef0 = 1.2, C = 1.0, epsilon = 0.1),
    #linear_model.LinearRegression(),   #0.011
    #linear_model.PoissonRegressor(),   #0.38
    linear_model.Ridge(alpha=0.001), #0.045 , better with lower alpha
    #linear_model.Lasso(alpha=0.001),   #0.41
]

train_comparing_targets = pd.DataFrame(target)
errors_train = pd.DataFrame()

for model in models_list:

    model.fit(features, target)
    predicts = model.predict(features)

    model_str = '{}'.format(model)
    train_comparing_targets[model_str] = predicts

    #print(metrics.mean_absolute_percentage_error(target, predicts))
    errors_train[model_str] = [ metrics.mean_absolute_percentage_error(target, predicts), 
                                metrics.r2_score(target, predicts),
                                np.sqrt(metrics.mean_squared_error(target, predicts)),
                                metrics.mean_squared_error(target, predicts),
                                ]

train_comparing_targets.to_csv(path_or_buf="./train_target_comparing.csv")
errors_train.to_csv(path_or_buf="./train_errors.csv")

# splitting the whole dataset into train set and test set. In this case using a simple static division
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

test_comparing_targets = pd.DataFrame(y_test)
errors_test = pd.DataFrame()

for model in models_list:

    model.fit(X_train, y_train)
    predicts = model.predict(X_test)

    model_str = '{}'.format(model)
    test_comparing_targets[model_str] = predicts

    errors_test[model_str] =[   metrics.mean_absolute_percentage_error(y_test, predicts), 
                                metrics.r2_score(y_test, predicts),
                                np.sqrt(metrics.mean_squared_error(y_test, predicts)),
                                metrics.mean_squared_error(y_test, predicts)
                            ]

test_comparing_targets.to_csv(path_or_buf="./test_target_comparing.csv")
errors_test.to_csv(path_or_buf="./test_errors.csv")

# Now try to assess again the Learning techniques, but with the cross validation technique. 
# We can take for example 5-fold CV or 10-fold CV, depending also on the computational time, so on the efficiency.
cv_train = pd.DataFrame()
cv_test = pd.DataFrame()

metrics_list = ['neg_mean_absolute_percentage_error', 'r2', 'neg_root_mean_squared_error', 'neg_mean_squared_error']

for model in models_list:
    model_str = '{}'.format(model)
    # train the models and perform cross-validation
    cross_validation = cross_validate(model, features, target, scoring = metrics_list, cv = 5, return_train_score = True, n_jobs = -1)
    # create the datasets with the metrics/scores selected.
    cv_train[model_str] = [np.mean(cross_validation['train_'+ metric]) for metric in metrics_list]
    cv_test[model_str] = [np.mean(cross_validation['test_'+ metric]) for metric in metrics_list]

cv_train.to_csv(path_or_buf="./cv_train.csv")
cv_test.to_csv(path_or_buf="./cv_test.csv")

#Try to plot some interesting things. This can be useful to assess the learning techniques, and analyze the flexibility against the effectiveness. 
for model in models_list:
    _, train_scores, test_scores, fit_times, score_times = learning_curve(  estimator = model, 
                                                                            X = features, 
                                                                            y = target, 
                                                                            cv = 5,
                                                                            n_jobs = -1, 
                                                                            scoring = 'neg_mean_absolute_percentage_error', 
                                                                            return_times = True
                                                                        )
    # transform the error in accuracy. Remember that with this function the error calculated is -MAPE.
    train_scores = 1 + train_scores
    test_scores = 1 + test_scores

    # create the figues:
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    ax1.plot(fit_times.mean(axis=1), train_scores.mean(axis=1), "o-", label = 'train_score', color = 'green')
    ax1.fill_between(
        fit_times.mean(axis=1),
        train_scores.mean(axis=1) - train_scores.std(axis=1),
        train_scores.mean(axis=1) + train_scores.std(axis=1),
        alpha=0.3,
        color = 'green'
    )
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("Fit time (s)")
    ax1.set_title(f"train_score of {model.__class__.__name__} ")
    ax2.plot(fit_times.mean(axis=1), test_scores.mean(axis=1), "o-", label = 'cross_validation_score', color = 'purple')
    #ax2.fill_between(
    #    fit_times.mean(axis=1),
    #    test_scores.mean(axis=1) - test_scores.std(axis=1),
    #    test_scores.mean(axis=1) + test_scores.std(axis=1),
    #    alpha=0.3,
    #    color = 'purple'
    #)
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel("Fit time (s)")
    ax2.set_title(f"cross_validation_score of {model.__class__.__name__} ")
    # show the resulting plots:
    plt.show()
    save_str = '{}'.format{model}
    plt.savefig("./" + save_str + ".png")


# Performing parameters tuning for the Random Forest (and maybe the others LT). 
# The choice of the Random Forest (as first model) is because it is the best model, according to the previous analysis.

# First create the base model to tune:
RF_regressor = RandomForestRegressor(random_state = 10)
# Look at parameters used by the learning technique:
#pprint(RF_regressor.get_params())

# Setting the parameters values to create the grid:
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(50, 800, num = 8)]
# Minimum number of samples required to split a node
min_samples_split = [2, 10, 100, 1000]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 4, 10, 50]

# Random-Grid search CV
# Create the random grid
#random_grid = {'n_estimators': n_estimators,
#               'max_depth': max_depth,
#               'min_samples_split': min_samples_split,
#               'min_samples_leaf': min_samples_leaf,
#               }
#
## Random search of parameters, using kFold cross validation. Use all available cores in parallel.
#RF_random_search = RandomizedSearchCV(  estimator = RF_regressor, 
#                                        param_distributions = random_grid,
#                                        n_iter = 100,   # search across different combinations
#                                        scoring = 'neg_mean_absolute_percentage_error', 
#                                        cv = 4, 
#                                        verbose = 1, 
#                                        random_state = 10, 
#                                        n_jobs = -1,
#                                        return_train_score = True,
#                                        refit = True
#                                        )
## Fit the random search model
#RF_random_search.fit(features, target)

# We can use also the GridSearchCV function, instead of the RandomizedSearchCV.
# The code is quite similar as the case before. The paramter grid remains the same.

# Create the search grid
search_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               }

RF_random_search = GridSearchCV(estimator = RF_regressor, 
                                param_distributions = search_grid,
                                scoring = 'neg_mean_absolute_percentage_error', 
                                cv = 5, 
                                n_jobs = -1,
                                return_train_score = True,
                                refit = True
                                )
# Fit the grid search model
RF_random_search.fit(features, target)

# get the best parameters found:
print(RF_random_search.best_params_)
# get the cross validation score with the best parameters found
print(RF_random_search.best_score_)
# We can use the function score if the refit parameter is set to True.
RF_random_search.score(features, target)


# Validation curves for some flexibility parameters. For example n_trees for the random forest (trees in the bag); or the c parameter for SVM.

#flexibility_range = np.logspace(200, 1000, 5)
flexibility_range = [int(x) for x in np.linspace(start = 200, stop = 1200, num = 6)]
print(flexibility_range)

train_scores, test_scores = validation_curve(
    RandomForestRegressor(),
    features,
    target,
    param_name="n_estimators",
    param_range=flexibility_range,
    scoring="neg_mean_absolute_percentage_error",
    n_jobs=-1,
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with Random Forest")
plt.xlabel("$n_trees$")
plt.ylabel("Score")
#plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(
    flexibility_range, train_scores_mean, label="Training score", color="darkorange", lw=lw
)
plt.fill_between(
    flexibility_range,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.2,
    color="darkorange",
    lw=lw,
)
plt.semilogx(
    flexibility_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw
)
plt.fill_between(
    flexibility_range,
    test_scores_mean - test_scores_std,
    test_scores_mean + test_scores_std,
    alpha=0.2,
    color="navy",
    lw=lw,
)
plt.legend(loc="best")
#plt.show()
