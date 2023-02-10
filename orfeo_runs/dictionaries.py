# Characters uniforming procedure for further procedures
char_alias = {
"’":"'",
"’":"'",
"’":"'"
}

# Contaction to Expansion > can't TO can not ,you'll TO you will
contractions = {
"ain't": "am not / are not / is not / has not / have not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is / how does",
"i'd": "I had / I would",
"i'd've": "I would have",
"i'll": "I shall / I will",
"i'll've": "I shall have / I will have",
"i'm": "I am",
"i've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}

# context for the query search
context = 'context:152.825047692124442624'

# list of words related to food for the query builder function
food_dict = ['almonds', 'anchovy', 'appetizer', 'apple', 'aroma', 'aromatic', 'artisanal', 'avocado', 'bacon', 'bagel', 'baked', 'baking', 'banana', 'barbecue', 'barbecued', 'barbeque', 'barley', 'basil', 'batter', 'battered', 'beans', 'beef', 'beer', 'biscuit', 'bitter', 'blanched', 'bland', 'blended', 'blueberry', 'boil', 'boiled', 'bouillon', 'braised', 'bread', 'bread crumbs', 'breaded', 'brined', 'broccoli', 'broiled', 'broiling', 'broth', 'bun', 'burger', 'burrito', 'butter', 'buttered', 'buttermilk', 'buttery', 'cabbage', 'cake', 'calzone', 'candy', 'canned', 'cantaloupe', 'capers', 'carbs', 'cardamom', 'caress', 'carrot', 'carrots', 'casserole', 'celery', 'cereal', 'charred', 'cheese', 'cheesecake', 'cheesy', 'chewy', 'chicken', 'chili', 'chilli', 'chimichanga', 'chips', 'chocolate', 'choose', 'chooser', 'chopped', 'chopping', 'chowder', 'chunky', 'chutney', 'cider', 'cinnamon', 'clams', 'cocoa', 'coconut', 'cod', 'coffee', 'cold', 'condiment', 'confectioners', 'cooked', 'cookie', 'cooking', 'coriander', 'corn', 'crab', 'creamy', 'crisp', 'crispy', 'croissant', 'crumble', 'crumbled', 'crumbly', 'crunchy', 'crust', 'crusty', 'cucumber', 'cupcake', 'cured', 'curry', 'custard', 'custardy', 'damp', 'deep-frying', 'deglazing', 'dehydrated', 'delicacy', 'delicious', 'dense', 'dessert', 'deviled', 'diced', 'dices', 'dietary', 'dip', 'dish', 'dough', 'doughnut', 'doughy', 'dressing', 'dried', 'dry', 'dumpling', 'earthy', 'eaten', 'edamame', 'egg roll', 'eggplant', 'eggs', 'enchilada', 'exquisite', 'fajitas', 'falafel', 'fat-free', 'fatty', 'fermented', 'fibrous', 'fiery', 'filling', 'fish', 'flaky', 'flash-frozen', 'flavored', 'flavorful', 'fluffy', 'fondue', 'food', 'fortified', 'fresh', 'fried', 'fried chicken', 'fries', 'fritter', 'fritters', 'frosted', 'frozen', 'fruits', 'fruity', 'fusion', 'garlic', 'garlicky', 'garnish', 'garnished', 'gelato', 'glazed', 'glazing', 'glistening', 'gluten-free', 'glutenous', 'gnocchi', 'grain', 'granola', 'grape', 'grapes', 'grated', 'gravy', 'greasy', 'grilled', 'grinded', 'gritty', 'guacamole', 'gumbo', 'gummy', 'gyro', 'halibut', 'ham', 'hamburger', 'healthy', 'herbaceous', 'herbal', 'herbed', 'herbs', 'homogenous', 'honey', 'hummus', 'ice cream', 'icecream', 'ingesting', 'ingredients', 'jam', 'juice', 'juices', 'juicy', 'juxtaposing', 'ketchup', 'lard', 'lasagna', 'latke', 'lemon', 'lemony', 'lettuce', 'lime', 'loaf', 'lobster', 'low-fat', 'mackerel', 'margarine', 'marinade', 'marinated', 'marmalade', 'marrow', 'marshmallow', 'mashed', 'mayonnaise', 'meal', 'meat', 'meatloaf', 'meaty', 'melon', 'melted', 'microwaveable', 'mild', 'milk', 'minced', 'mixing', 'moist', 'mousse', 'muffin', 'mushroom', 'mushrooms', 'mushy', 'mustard', 'nachos', 'non-fat', 'noodle', 'nourishing', 'nutrients', 'nutritious', 'nuts', 'nutty', 'oatmeal', 'oil', 'olive oil', 'olives', 'onion', 'orange', 'organic', 'oysters', 'paella', 'pan-frying', 'pancake', 'pancakes', 'pasta', 'pasteurized', 'patty', 'peanuts', 'peas', 'pecan', 'pepper', 'peppers', 'peppery', 'pesto', 'pickle', 'pickled', 'pickles', 'pie', 'pierogi', 'piquant', 'pistachios', 'pita', 'pizza', 'placed', 'platter', 'poached', 'popcorn', 'pork', 'porridge', 'potato', 'potatoes', 'premium', 'preparing', 'pretzel', 'processed', 'pudding', 'pungent', 'pureed', 'quesadilla', 'queso', 'quiche', 'quick-cooking', 'ramen', 'ravioli', 'raw', 'recipe', 'reddish', 'refined', 'refreshing', 'rice', 'rich', 'risotto', 'roast', 'roasted', 'rum', 'salad', 'salami', 'salmon', 'salsa', 'salted', 'salting', 'salty', 'sandwich', 'sardines', 'sarding', 'sauce', 'saucy', 'sausage', 'sauté', 'savory', 'scone', 'seafood', 'seared', 'searing', 'seasoned', 'seasoning', 'seaweed', 'seed', 'shawarma', 'shredded', 'shrimp', 'simmer', 'simmered', 'simmering', 'sizzled', 'slices', 'slushy', 'smirched', 'smoked', 'smoky', 'smooth', 'smoothie', 'smothered', 'smothering', 'soba', 'sodium', 'soggy', 'sorbet', 'souffle', 'soup', 'soupy', 'sour', 'soy sauce', 'spaghetti', 'spanakopita', 'spice', 'spiced', 'spices', 'spicy', 'spinach', 'spongy', 'spring roll', 'squash', 'stale', 'starchy', 'steamed', 'stew', 'stewed', 'stewing', 'stir fry', 'stir-fried', 'strawberry', 'strips', 'stuffed', 'subs', 'succulent', 'sugar', 'sugary', 'sushi', 'sustainable', 'sweet', 'tabbouleh', 'tableware', 'tabouli', 'taco', 'tamale', 'tangy', 'tart', 'tarts', 'taste', 'tasteless', 'tasting', 'tasty', 'tea', 'tempura', 'tender', 'tenderized', 'tenderizing', 'tenderloin', 'teriyaki', 'thick', 'thin', 'toast', 'toasted', 'toasty', 'tofu', 'tomato', 'tortilla', 'tortillas', 'tough', 'trout', 'truffle', 'tumble', 'turkey', 'udon', 'unprocessed', 'unrefined', 'vegan', 'vegetables', 'vegetal', 'veggies', 'velvety', 'vinegar', 'waffle', 'walnuts', 'watermelon', 'wheat', 'whole-grain', 'wholesome', 'wine', 'yam', 'yogurt', 'zesty']

tag_dict = []













