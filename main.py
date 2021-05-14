def openFile(path):
    # param path: path/to/file.ext (str)
    # Returns contents of file (str)
    with open(path) as file:
        data = file.read()
    return data


imdb_data = openFile('sentiment_labelled_sentences/sentiment_labelled_sentences/imdb_labelled.txt')
amzn_data = openFile('sentiment_labelled_sentences/sentiment_labelled_sentences/amazon_cells_labelled.txt')
yelp_data = openFile('sentiment_labelled_sentences/sentiment_labelled_sentences/yelp_labelled.txt')

# combine datasets
datasets = [imdb_data, amzn_data, yelp_data]

combined_dataset = []
# separate samples from each other
for dataset in datasets:
    combined_dataset.extend(dataset.split('\n'))

# separate each label from each sample
dataset = [sample.split('\t') for sample in combined_dataset]

import pandas as pd

df = pd.DataFrame(data=dataset, columns=['Reviews', 'Labels'])

# Remove any blank reviews
df = df[df["Labels"].notnull()]

# shuffle the dataset for later.
# Note this isn't necessary (the dataset is shuffled again before used),
# but is good practice.
df = df.sample(frac=1)


import string

df['Word Count'] = [len(review.split()) for review in df['Reviews']]

df['Uppercase Char Count'] = [sum(char.isupper() for char in review) \
                              for review in df['Reviews']]

df['Special Char Count'] = [sum(char in string.punctuation for char in review) \
                            for review in df['Reviews']]

positive_mask = df['Labels'] == '1'
positive_samples = df[positive_mask]
negative_samples = df[~positive_mask]

# positive_samples['Word Count'].describe()
# negative_samples['Word Count'].describe()
#
# positive_samples['Uppercase Char Count'].describe()
# negative_samples['Uppercase Char Count'].describe()
#
# positive_samples['Special Char Count'].describe()
# negative_samples['Special Char Count'].describe()


from collections import Counter

def getMostCommonWords(reviews, n_most_common, stopwords=None):
    # param reviews: column from pandas.DataFrame (e.g. df['Reviews'])
        #(pandas.Series)
    # param n_most_common: the top n most common words in reviews (int)
    # param stopwords: list of stopwords (str) to remove from reviews (list)
    # Returns list of n_most_common words organized in tuples as
        #('term', frequency) (list)

    # flatten review column into a list of words, and set each to lowercase
    flattened_reviews = [word for review in reviews for word in \
                         review.lower().split()]


    # remove punctuation from reviews
    flattened_reviews = [''.join(char for char in review if \
                                 char not in string.punctuation) for \
                         review in flattened_reviews]


    # remove stopwords, if applicable
    if stopwords:
        flattened_reviews = [word for word in flattened_reviews if \
                             word not in stopwords]


    # remove any empty strings that were created by this process
    flattened_reviews = [review for review in flattened_reviews if review]

    return Counter(flattened_reviews).most_common(n_most_common)




import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


# without removing stopwords
# print(getMostCommonWords(positive_samples['Reviews'], 10))

# with removing stopwords
print(getMostCommonWords(positive_samples['Reviews'], 10, stopwords.words('english')))

print(getMostCommonWords(negative_samples['Reviews'], 10, stopwords.words('english')))


#
# from tabulate import tabulate
# print(tabulate(positive_samples, headers='keys', tablefmt='psql'))
# print(tabulate(df['Word Count'].describe()));
#
