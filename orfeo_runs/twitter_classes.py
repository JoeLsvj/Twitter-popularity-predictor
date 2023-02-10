from modules import *

# importing the script with the credentials for twitter as a module
import twetter_credentials as credentials

import dictionaries
from dictionaries import context
from dictionaries import food_dict

# create a class for the authentication for both the API versions, 
# taking the credentials from the imported script. Initializing the arguments as default 
# positional arguments, we can initialize the class without passing any argument in the object instantiation.
# We can also initialize only one API version access (API or client, which requires only the bearer token).
class authentication():
    def __init__(self,  consumer_key=credentials.consumer_key, 
                        consumer_secret=credentials.consumer_secret, 
                        access_token= credentials.access_token, 
                        access_secret=credentials.access_secret, 
                        bearer_token= credentials.bearer_token        ):
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.access_token = access_token
        self.access_secret = access_secret
        self.bearer_token = bearer_token

    # API v2.0 authentication via client
    def client(self):
        client = tweepy.Client( bearer_token = self.bearer_token,
                                consumer_key = self.consumer_key,
                                consumer_secret = self.consumer_secret,
                                access_token = self.access_token,
                                access_token_secret = self.access_secret,
                                wait_on_rate_limit=True)
        return client
    
    # API v1.1 authentication via API
    def api(self):
        auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret)
        auth.set_access_token(self.access_token, self.access_secret)
        api = tweepy.API(auth, wait_on_rate_limit=True)
        return api


# Create a class with some functions to build the searching query. The syntax for the searching query is different 
# among the two API versions (the reason for this is quite obfuscated), and also the keywords/parameters for advance 
# reaserch of tweets are also different. For example in the API v 1.1 there is no possibility to search a tweet with the 
# keyword `context` (that allows to search tweets belonging to a very large set of listed contexts and subcontexts) but only
# giving the words related to a topic, with the proper logic operators. In the API v2.0 the query syntax offers the possibility 
# to search tweets directly specifing the `context` keyword (with the related number of the desired context to search). However,
# there is no more the possibility to filter out the tweets using the number of retweets or replies... This last choise is again
# very obfuscated...
class query():

    # Give a default value to the constructor's parameters. In this way, if we want only the query
    # in v1.1 version, the class can be initialized passing only the dictionary and the number of tokens.
    # For the query version v 2.0, we can pass only the context string.
    def __init__(self, dictionary=None, number_of_tokens=None, context=None):
        self.dictionary = dictionary
        self.number_of_tokens = number_of_tokens
        self.context = context


    # Define a function to build a single query for the Cursor method in tweepy, in the form of a string.
    # The query is built picking randomly a certain number_of_tokens from the dictionary, and writing them to 
    # the query. Passing this query to the Cursor(), the tweets with all the words selected are searched.
    # This function must be called inside a loop, in order to create a datset of tweets about the topic 
    # specified in the dictionary. 
    def make_query_api(self):
        L = len(self.dictionary)
        q = "("
        for i in range(self.number_of_tokens):
            j = random.randint(0, L-1)
            q = q + self.dictionary[j] + " "
        # Append here all the other filters. For example we can specify the temporal interval, the minimum
        # number of retweets or replies, and ask only for original tweets, excluding retweets and replies.
        q = q + ") " + "min_retweets:3 min_replies:1 until:2022-12-15 since:2022-01-01 -filter:replies -filter:retweets lang:en"
        return q


    # Define a function to make a query for the client Paginator() method in tweepy, in the form of a string.
    # Since the context does not change, this query can be created once, calling this function, without the need
    # of performin a `for` loop.
    def make_query_client(self):
        # With the API v2.0 we can search with the context keyword, but the possibility to filter with the 
        # minimum number of retweets or replies is unavailable. The possibility to specify the temporal interval
        # is guaranteed by a parameter of the Paginator function.
        q =  self.context + " -is:reply -is:retweet lang:en"
        return q

    # The context related number is listed in the twetter official web page. For example the food context number is 152. 
    # However, in the required format, it is mandatory to specify the sub-context/entity number. These (very large) numbers
    # are not specified, and have to be discovered directly by the user, printing the `context_annotations` parameter of a tweet,
    # returned by the `Paginator` method. This parameter, shows all the annotations about the entities and context of a certain tweet.
    # A tweet can belong to different contexts and sub-contexts (or a mixture of them) -> all this information is reported.
    # The number 825047692124442624, for example, indicates a generic sub-context about food. This function does exactly this.
    def find_context(tweet):
        print(tweet.context_annotations)
        return


    # Define a function to make a single query for the `StreamingClinet()` method in tweepy, in the form of a list.
    # This client (v 2.0) method accepts queries only in the form of a list, and the parameter is even no more called query,
    # in the tweepy' function definition. For now, this function remains un-used.
    def make_query_stream():
        L = len(self.dictionary)
        q = []
        for i in range(self.number_of_tokens):
            j = random.randint(0, L-1)
            q.append(self.dictionary[j])
        return q


# Now create a class for the initial text_mining phase (this implementation will be useful in the final code 
# for production that wont be a notebook format). The aim of the class is to provide functions to create the dataset
# for all the implementations selected, and the API version selected. The first functions are able to create the entire dataset
# (always for all the API versions and more :)) ) with the lables of interest: the date of creation of the tweets, the authors,
# of course the text, the authors' followers, and the fundamental metrics for predicting popularity: the number of retweets,
# the number of replies messages to the tweets, and the number of quotes (that are the number of time a certain tweet has been
# mentioned in other tweets' text). The number of likes metric has been excluded, because a tweet can be popular, but with a little
# number of likes, maybe because the content is controversial or storng but non condivisible. In the food filed we can think about 
# tweets regarding disgusting alimentary habits (like eating pizza with pineapple).

class text_mining():

    # Define the constructor operator, with default values set to None. In this way we are not 
    # obliged to parse arguments at the moment of initialization of the class object. In particular,
    # if the class is intended to also create the dataset (and not only for cleaning it), only the query 
    # parameter has to be passed. The query parameter is an instantiation of the query class created above, 
    # not the query string created with that class. In fact, every function to created the dataset, will create 
    # the proper query string using the query-class functions with the instantiation provided in this constructor.
    # If you want only to clean an already prepared dataset, pass only the `dataset` argument 
    # (better to be a Pandas DataFrame object type).
    def __init__(self, query=None, dataset=None, corpus=None):
        self.query = query
        self.dataset = dataset
        self.corpus = corpus
    
    # function to create a dataset for the tweets, using the API v2.0 via tweepy.client
    # Non deprecated since it uses the API v2.0
    def create_dataset_client(self):
        
        # Create the proper query string for API v2 and Paginator method, using the proper function of the query-class.
        query_client = self.query.make_query_client()

        # The Paginator() method is used for doing pagination, instead of tweepy.Cursor. This instruction creates pages of tweets. 
        # In particular `limit` pages, each page with `max_results` number of tweets. The method `flatten` simply transforms the 
        # page-structure in an array of tweets (with all the metadata specified) and returns a `limit` number of tweets.
        # Maybe this is useful to avoid the rate limit error with Cursor and API v1.1. Notice that the tweets which can be 
        # imported with the API v1.1 are unlimited. Instead, the ones imported with the version 2.0 are limited to 2M.
        # Moreover, with the `Paginator` method, we have to specify all the fields/attributes of tweets we want to import, 
        # with the `tweet_field` variable, unless they are not defined later (with Cursor is not required).
        tweets = tweepy.Paginator(  client.search_recent_tweets, query=query_client, 
                                    tweet_fields=['created_at', 'text', 'id','author_id', 'public_metrics', 'entities', ], 
                                    #user_fields=['id', 'username','public_metrics'], 
                                    #expansions=['author_id', 'attachments.media_keys'], 
                                    max_results=100, limit=50                                ).flatten(limit=5000)

        # Create a list with all the attributes, iterating on all the tweets returned by Paginator
        tweets_list = [[    tweet.created_at,
                            # with the API v2.0 seems cumbersome to get the users' followers number.
                            # However, collected the tweets, the two API can be mixed up. We can use the 
                            # function get_user() from tAPI v1.1 that allows to get followers count in a simple way. 
                            # The identical function get_user() for the API v2.0 does not allow this (really obfuscated)
                            api.get_user(user_id=tweet.author_id).name, 
                            tweet.text,
                            api.get_user(user_id=tweet.author_id).followers_count,
                            # the interesting attributes are contained in the public metrics varibale.
                            # There practically no indication of this in the documentation (that is actually obfuscated).
                            tweet.public_metrics['retweet_count'], 
                            tweet.public_metrics['reply_count'], 
                            tweet.public_metrics['quote_count']     ]   for tweet in tweets if (tweet.public_metrics['retweet_count']>=1)]


        column_lables = [ 'date', 'author', 'text', 'author followers', 'retweets', 'replies', 'quotes' ]
        dataset = pd.DataFrame(tweets_list, columns=column_lables)

        # Select and store in a varibale the column of the dataset with the actual text of the tweets
        tweets_text = dataset['text']
        # Then upload the class attribute `dataset` (initially set to None) with the text column of the tweets,
        # in order to use it in the next cleaning procedure.
        self.dataset = dataset
        self.corpus = tweets_text 
        return dataset

    # The following function is able to create a dataset of tweets, without using the twetter API. To do so,
    # the function uses the library `snscrape`, that limits the number of tweet to import to 1M. Moreover,
    # the searching query uses the syntax of the API v1.1, with also the possibility to get the number of replies
    # and the number of quotes, which is not possible with the tweepy Cursor.
    def create_dataset_scraper(self):
        
        tweets_list = []
        # define the number of iteration over the whole dictionary about food. Since it is not possible to search
        # directly with the `context`, to get the tweets about food, we can build a different query every loop iteration
        # choosing random words in the food's dictionary.
        total_iterations = 2000
        items_per_query = 2

        for i in range(total_iterations):
            # Define the query in the API v1.1 standard (the actual one used in twetter app to search )
            query_api = self.query.make_query_api()
            # instantiation of the snscraper class
            tweets = sntwitter.TwitterSearchScraper(query_api).get_items()
            # attributes of interest: .date, .user.username, .user.followersCount, .content, .replyCount, .retweetCount, .quoteCount, media (count or even type)
            for count,tweet in enumerate(tweets):
                # define a variable for the media count:
                if (tweet.media != None): media = len(tweet.media)
                else: media = 0
                tweets_list.append([    tweet.date, 
                                        tweet.user.username, 
                                        tweet.content, 
                                        tweet.user.followersCount,
                                        #tweet.user.friendsCount,
                                        tweet.user.statusesCount,
                                        tweet.user.favouritesCount,
                                        #tweet.user.listedCount,
                                        tweet.retweetCount,
                                        tweet.replyCount,
                                        tweet.quoteCount,
                                        media                           ])
                if (count == items_per_query): break
        # create the dataset with Pandas as before:
        column_lables = [   'date', 'author', 'text', 
                            'followers', 
                            #'friends', 
                            'status', 
                            'favourites',
                            #'tot_listed', 
                            'retweets', 'replies', 'quotes', 'media' ]
        dataset = pd.DataFrame(tweets_list, columns=column_lables)
        # Select and store in a varibale the column of the dataset with the actual text of the tweets
        # Then upload the class attribute `dataset` (initially set to None) with the text column of the tweets,
        # in order to use it in the next cleaning procedure.
        self.dataset = dataset
        self.corpus = dataset['text']
        return self.dataset


    # define a function to create the dataset for the tweets using the API v1.1 with tweepy.
    # Almost deprecated in favor of scraper.
    def create_dataset_api(self):

        tweets_list = []
        total_iterations = 50
        items_per_query = 4
        # same structure as before
        for i in range(total_iterations):
            query_api = self.query.make_query_api()
            tweets = tweepy.Cursor(api.search_tweets, q = query_api, lang = "en", tweet_mode='extended').items(items_per_query)

            tweets_list = tweets_list + [   [tweet.created_at, 
                                            tweet.user.name,
                                            tweet.user.followers_count,
                                            tweet.text,
                                            tweet.retweet_count,
                                            tweet.favorite_count]   for tweet in tweets ]
        # In this case the dataset is composed by the columns:
        # date of pubblication, author, text, user's follower, retweets, likes
        # With the API v1.1 for tweepy there is no possbility to get the number of replies and quotes
        column_lables = [ 'date', 'author', 'author followers', 'text', 'retweets', 'likes' ]
        dataset = pd.DataFrame(tweets_list, columns=column_lables)
        # Select and store in a varibale the column of the dataset with the actual text of the tweets
        tweets_text = dataset['text']
        # Then upload the class attribute `dataset` (initially set to None) with the text column of the tweets,
        # in order to use it in the next cleaning procedure.
        self.dataset = dataset
        self.corpus = tweets_text
        return dataset


    # The following function is quite similar (the structure is identical) as the one before.
    # The only difference is that, in order to avoid (or try to) the problems with the limit rate
    # of requests with the twetter API, the tweets imported are grouped in (two) pages instead of in single items.
    # So the `Cursor` function is used as paginator with `Cursor().pages(number of pages)`, instead of with `.items()`.
    # The number of tweets imported each cycle is not large, so 2 pages might be enough (or this function not so useful).
    # Almost deprecated in favor of scraper...
    def create_dataset_pages(self):

        tweets_list = []    # we can construct a bigger dataset commenting this line, 
                            # and using more calls of the code box, that appends newer tweets in the list.
        total_iterations = 50
        items_per_page = 5
        n_pages = 2

        for i in range(total_iterations):
            # To avoid problems with high number of reuqests, we can ask only for tweets that are actually retweets.
            # The problem then is to find out the number of followers of the original author.
            query_api = self.query.make_query_api()
            # Use the same syntax as before to import a dataset of tweets about food:
            tweets = tweepy.Cursor(api.search_tweets, q = query_api, lang = 'en', count=items_per_page, tweet_mode='extended').pages(n_pages)
            # create a list for the tweets, including, respectively: date of creation, authot' username, actual text, number of retweets, number of likes.
            for tweet in tweets:
                for i in range(len(tweet)):
                    tweets_list.append([tweet[i].created_at, 
                                        tweet[i].user.name,
                                        tweet[i].user.followers_count,
                                        tweet[i].text, 
                                        tweet[i].retweet_count, 
                                        tweet[i].favorite_count]    )

        column_lables = [ 'date', 'author', 'author followers', 'text', 'retweets', 'likes' ]
        dataset = pd.DataFrame(tweets_list, columns=column_lables)
        # Select and store in a varibale the column of the dataset with the actual text of the tweets
        tweets_text = dataset['text']
        # Then upload the class attribute `dataset` (initially set to None) with the text column of the tweets,
        # in order to use it in the next cleaning procedure.
        self.dataset = dataset
        self.corpus = tweets_text
        return dataset

    def expand(self):
        #self.dataset['word_count'] = self.dataset['text'].apply(lambda x : len([word for word in str(x).split() if len(word)>1])) 
        self.dataset['word_count'] = self.dataset['text'].apply(lambda x : len(str(x).split()))
        #self.dataset['char_count'] = self.dataset['text'].apply(lambda x : regex.findall(r'\X', x)).apply(lambda i : len(i))
        self.dataset['char_count'] = self.dataset['text'].apply(lambda x : len(x))
        self.dataset['stop_words'] = self.dataset['text'].apply(lambda x : len([t for t in x.split() if t in STOP_WORDS]))
        self.dataset['#tag'] = self.dataset['text'].apply(lambda x : len([t for t in x.split() if t.startswith('#')]))
        self.dataset['@'] = self.dataset['text'].apply(lambda x : len([t for t in x.split() if t.startswith('@')]))
        self.dataset['numaric'] = self.dataset['text'].apply(lambda x : len([t for t in x.split() if t.isdigit()]))
        self.dataset['upercase'] = self.dataset['text'].apply(lambda x : len([t for t in x.split() if t.isupper()]))
        # Extract the emails
        #self.dataset['emails'] = self.dataset['text'].apply(lambda x : re.findall(r'([A-Za-z0-9+_-]+@[A-Za-z0-9+_-]+\.[A-Za-z0-9+_-]+)', x))
        # Count the emails
        #self.dataset['emails_count'] = self.dataset['text'].apply(lambda x : re.findall(r'([A-Za-z0-9+_-]+@[A-Za-z0-9+_-]+\.[A-Za-z0-9+_-]+)', x)).apply(lambda x : len(x))
        # Count URL: 
        self.dataset['URLs'] = self.dataset['text'].apply(lambda i : re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', i)).apply(lambda i : len(i))
        # we can also count the number of emoji
        self.dataset['emojis_count'] = self.dataset['text'].apply(lambda x : regex.findall(r'\X', x)).apply(lambda x : len([char for char in x if(char in emoji.UNICODE_EMOJI)]))
        # Count all punctiations and special characters from data, excluding the emojis already counted:
        #self.dataset['special_char'] = self.dataset['text'].apply(lambda x : re.findall('[^A-Z a-z 0-9-]', x)).apply(lambda x : len(x))
        # Extracting emotes. This might be useful for a tf-idf features apart from the ones with the pure text.
        #self.dataset['emojis'] = self.dataset['text'].apply(lambda x : regex.findall(r'\X', x)).apply(lambda x : ''.join([char for char in x if(char in emoji.UNICODE_EMOJI)]))
        return #self.dataset

    # Define a function to clean the tweets. This function takes the single tweet and process it.
    # Notice that this is not a class member function (does not contain the self attribute).
    def tweet_cleaner(tweet):
        # force to lower case
        tweet = tweet.lower()
        # substitute some unused/incompatible/strange chars with their used alias 
        # (for example for successfully applying the following expansions of the constracted forms)
        for key in dictionaries.char_alias:
            value = dictionaries.char_alias[key]
            tweet = tweet.replace(key, value)
        # expand the contracted forms, very frequent in english
        for key in dictionaries.contractions:
            value = dictionaries.contractions[key]
            tweet = tweet.replace(key, value)
        # again the lower case, because yes. It is needed...
        tweet = tweet.lower()
        # remove all the e-mails
        tweet = re.sub(r'([A-Za-z0-9+_]+@[A-Za-z0-9+_]+\.[A-Za-z0-9+_]+)',' ', tweet)
        # remove the URLs
        tweet = re.sub(r'https?://[^\s<>"]+|www\.[^\s<>"]+',' ', tweet)
        # remove mentions
        tweet = re.sub('@[A-Za-z0-9_]+', '', tweet)
        # removes all numbers
        tweet = re.sub('[0-9]', '', tweet)
        # remove other characters
        #tweet = unicodedata.normalize('NFKD', tweet).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        tweet = re.sub('[^A-Z a-z 0-9-]+','', tweet)
        # remove extra spaces
        tweet = " ".join(tweet.split())
        # remove the stopwords
        #tweet = " ".join(t for t in tweet.split() if t not in STOP_WORDS)
        # stemming procedure
        nlp = spacy.load('en_core_web_sm')
        stem_sentence=[]
        doc = nlp(tweet)
        for token in doc:
            stem_sentence.append(token.lemma_)
            stem_sentence.append(" ")
        tweet = "".join(stem_sentence)
        # lemmatization:
        #nlp = spacy.load('en_core_web_sm')
        #doc = nlp(tweet)
        #lis = []
        #for token in doc:
        #    lemma = token.lemma_
        #    if lemma == '-PRON-' or lemma == 'be':
        #        lemma = token.text
        #    lis.append(lemma)
        #tweet = " ".join(lis)
        # We can correct spelling using textblob: 
        #tweet = str(TextBlob(tweet).correct())
        return tweet

    # The whole dataset function, to clean the whole dataset. This function is a class member function,
    # and takes as parameter a cleaner function with the algorithm to process the single tweet.
    def clean(self, cleaner_function=tweet_cleaner):
        # re-define the class attribute `dataset` applying the function to clean the single tweet, and return it 
        self.corpus = self.corpus.apply(cleaner_function)
        self.dataset['text'] = self.corpus
        # Remove tweets that are too short ...
        for i in range(len(self.dataset)):
            if self.dataset['word_count'][i] < 5:
                self.dataset = self.dataset.drop(i, axis=0)
        self.dataset = self.dataset.reset_index(drop=True)
        # Remove tweets that contains strange chars, such as chars from different alphabets.
        #undecoded_chars = self.dataset['text'].apply(lambda x : re.findall('[^\w\s]', x)).apply(lambda i : len(i)) - self.dataset['emojis_count']
        #for i in range(len(self.dataset)):
        #    if self.dataset['special_char'][i] != undecoded_chars[i]:
        #        self.dataset = self.dataset.drop(i, axis=0)
        #self.dataset = self.dataset.reset_index()
        #### A function that removes tweets that have been modified too much is very desired.... 
        #### The stopwords might be a problem in this case, but will'see.
        return self.dataset



class cluster():

    def __init__(       self, 
                        number_of_clusters = 8, 
                        number_top_words = 6, 
                        batch_size = 64, 
                        init = "nndsvda", 
                        dataset = None, 
                        *args, **kwargs     ):

        self.number_of_clusters = number_of_clusters
        self.number_top_words = number_top_words
        self.batch_size = batch_size
        self.init = init
        self.dataset = dataset
    

    def plot_top_words(model, feature_names, n_top_words, title):
        fig, axes = plt.subplots(2, 4, figsize=(10, 5), sharex=True)
        axes = axes.flatten()
        for topic_idx, topic in enumerate(model.components_):
            top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
            top_features = [feature_names[i] for i in top_features_ind]
            weights = topic[top_features_ind]

            ax = axes[topic_idx]
            ax.barh(top_features, weights, height=0.7)
            ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 8})
            ax.invert_yaxis()
            ax.tick_params(axis="both", which="major", labelsize=8)
            for i in "top right left".split():
                ax.spines[i].set_visible(False)
            fig.suptitle(title, fontsize=10)

        plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
        plt.show()

    def NMF(self, plotting_function=plot_top_words):
        # Use tf-idf features for NMF.
        print("Extracting tf-idf features for NMF...")
        tfidf_vectorizer = TfidfVectorizer( max_df=0.95, min_df=2, stop_words="english" )

        t0 = time()
        tfidf = tfidf_vectorizer.fit_transform(self.dataset)
        print("done in %0.3fs." % (time() - t0))

        # Fit the NMF model
        print( "Fitting the NMF model (Frobenius norm) with tf-idf features" )

        nmf = NMF(
            n_components=self.number_of_clusters,
            random_state=1,
            init=self.init,
            beta_loss="frobenius",
            alpha_W=0.00005,
            alpha_H=0.00005,
            l1_ratio=1,
        ).fit(tfidf)

        tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
        plotting_function(nmf, tfidf_feature_names, self.number_top_words, "Topics in NMF model (Frobenius norm)" )

        # Fit the NMF model
        print("Fitting the NMF model (generalized Kullback-Leibler divergence) with tf-idf features" )

        nmf = NMF(
            n_components=self.number_of_clusters,
            random_state=1,
            init=self.init,
            beta_loss="kullback-leibler",
            solver="mu",
            max_iter=1000,
            alpha_W=0.00005,
            alpha_H=0.00005,
            l1_ratio=0.5,
        ).fit(tfidf)

        tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
        plotting_function( nmf, tfidf_feature_names, self.number_top_words, "Topics in NMF model (generalized Kullback-Leibler divergence)")


    

    def LDA(self, plotting_function=plot_top_words):
        # Use tf (raw term count) features for LDA.
        print("Extracting tf features for LDA...")
        tf_vectorizer = CountVectorizer( max_df=0.95, min_df=2, stop_words="english" )

        t0 = time()
        tf = tf_vectorizer.fit_transform(self.dataset)
        print("done in %0.3fs." % (time() - t0))

        print( "Fitting LDA models with tf features" )
        lda = LatentDirichletAllocation(
            n_components=self.number_of_clusters,
            max_iter=5,
            learning_method="online",
            learning_offset=50.0,
            random_state=0,
        )

        t0 = time()
        lda.fit(tf)
        print("done in %0.3fs." % (time() - t0))

        tf_feature_names = tf_vectorizer.get_feature_names_out()
        plotting_function(lda, tf_feature_names, self.number_top_words, "Topics in LDA model")

