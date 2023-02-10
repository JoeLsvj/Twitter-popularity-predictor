print("predicting tweets' popularity")

# Import the standard modules
#from modules import *

# Import the classes built for twitter:
from twitter_classes import *

# Authentication:
api = authentication().api()
client = authentication().client()

# Initialize the query class object:
query = query(dictionary=food_dict, number_of_tokens=2, context=context)
# Call the functions to make the queries
query_api = query.make_query_api()
query_client = query.make_query_client()

# Initialize the text_mining class object, passing the instantiation of the query class.
# Then create the dataset, and clean the tweets.
miner = text_mining(query=query)
dataset = miner.create_dataset_scraper()
# expand, clean and save to csv
dataset.to_csv(path_or_buf="./raw_dataset.csv")
miner.expand()
dataset_clean = miner.clean()
dataset.to_csv(path_or_buf="./clean_dataset.csv")

### Features analysis
# create a scatter matrix with some features.
dataset_plot = dataset_clean.drop(['date', 'author', 'text', 'favourites', 'quotes'], axis=1)
scatter_matrix = pd.plotting.scatter_matrix(dataset_plot, figsize=(20,20))
plt.savefig("./scatter_matrix.png")

### Cluster analysis.
#clusters = cluster(dataset=dataset_clean['text'])
#clusters.NMF()
#clusters.LDA()

### TRAINING PART
# Create a function of the class to insert these stuff. Call it "make_features_target"
target_data = dataset_clean[['retweets', 'replies', 'quotes']]
features_data = dataset_clean.drop(["date", "author", "retweets", "replies", "quotes", "status", "favourites", "char_count", "word_count", "stop_words"], axis=1)
print(target_data.shape, features_data.shape)


# Transform the features and the targets, using column transform. This is now a sort of obliged step, since we have several columns of different features. 
# The main problem is that the features related to the pure text, that are the tf-idf features, are created by the sklearn with an object of the type csx_sparse matrix of 
# scipy. This object is a unique block: cannot be divided in rows of a dataset, because looses the type of spoarse matrix (becomes a list of series objects) and the LT 
# of sklearn complains (wants pandas dataframe, sparse matrix or np arrays).

# instantiate the vectorizer as an object of the class `TfidfVectorizer`, and assign it to the varibale vectorizer.
vectorizer = TfidfVectorizer(max_df=0.95, min_df=1, stop_words='english', ngram_range=(2,5))

normalize = Normalizer()

transformer = ColumnTransformer(
    [
        ('tfidf', vectorizer, 'text'),
        ('normalize', normalize, ["followers"])
    ], remainder = 'passthrough')

# create the transformed features
features = transformer.fit_transform(features_data)

# Now create the targets and apply some transformations to them
standard_scaler = StandardScaler()
normalize = Normalizer()

target = target_data['retweets'] + target_data['replies'] #+ target_data['quotes']
#target = normalize.fit_transform(target.to_numpy().reshape(len(target),1))
target = np.log(target)

# Get the dictionary of the tf-idf features
#vectorizer.fit_transform(features_data['text'])
#print(vectorizer.get_feature_names_out())

from sklearn import linear_model

models_list = [
    RandomForestRegressor(),   # better with the test bag
    linear_model.LinearRegression(),   #0.011
    #linear_model.HuberRegressor(),    # for outliers...
    #linear_model.LogisticRegression(),  # discrete, classification
    #linear_model.PoissonRegressor(),   #0.38
    linear_model.Ridge(alpha=0.1), #0.045 , better with lower alpha
    linear_model.Lasso(alpha=0.5),   #0.41
    #linear_model.ARDRegression(),  #0.05 very good. But requires more time, and the data to be dense.
]

train_comparing_targets = pd.DataFrame(target)
errors_train = pd.DataFrame()

for model in models_list:

    model.fit(features, target)
    # coefficient of determination for the linear model
    #print(model.score(features, target))
    predicts = model.predict(features)

    model_str = '{}'.format(model)
    train_comparing_targets[model_str] = predicts

    max_target = []
    for i in range(len(target)):
        max_target.append(max(target[i], predicts[i]))
    #print(    np.mean(abs(target-predicts)/max_target)       )
    #print(metrics.mean_absolute_percentage_error(target, predicts))
    errors_train[model_str] = [metrics.mean_absolute_percentage_error(target, predicts), np.mean(abs(target-predicts)/max_target)]

train_comparing_targets.to_csv(path_or_buf="./train_target_comparing.csv")
errors_train.to_csv(path_or_buf="./train_errors.csv")

# splitting the whole dataset into train set and test set.
# In this case using a simple static division, without cross fold validation and other more complex things...
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

#print(X_train.shape, X_test.shape, "\n")
#print(type(X_train), "\n")

# pass from pandas series to numpay ndarray, in order (to compute the second error...)
y_test = y_test.to_numpy()
y_train = y_train.to_numpy()

test_comparing_targets = pd.DataFrame(y_test)
errors_test = pd.DataFrame()

for model in models_list:

    model.fit(X_train, y_train)
    # coefficient of determination for the linear model
    #print(model.score(features, y_test))
    predicts = model.predict(X_test)

    model_str = '{}'.format(model)
    test_comparing_targets[model_str] = predicts

    max_target = []
    for i in range(len(y_test)):
        max_target.append(max(y_test[i], predicts[i]))
    #print(    np.mean(abs(y_test-predicts)/max_target)       )
    #print(metrics.mean_absolute_percentage_error(y_test, predicts))
    errors_test[model_str] = [metrics.mean_absolute_percentage_error(y_test, predicts), np.mean(abs(y_test-predicts)/max_target)]

test_comparing_targets.to_csv(path_or_buf="./test_target_comparing.csv")
errors_test.to_csv(path_or_buf="./test_errors.csv")

print("end program")
