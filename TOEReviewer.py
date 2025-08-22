import matplotlib.pyplot as plt
import pandas as pd
import seaborn
from wordcloud import WordCloud

import cleaningFunctions

reviews = pd.read_csv("TOEreviews.csv")
data_dim = reviews.shape # Find the shape of Dataset
cols = reviews.columns
colsList = cols.tolist()

reviewsNa = cleaningFunctions.check_null(reviews) # Removes all the Na Rows

# Tokenize the data
reviewsNa = reviewsNa.copy() # Creates a copy of the db
reviewsNa['tokenized'] = reviewsNa["Review"].apply(cleaningFunctions.tokenize) # tokenizes the Reviews

# Remove StopWords & Punctuation
reviewsNa['tokenized'] = reviewsNa["tokenized"].apply(cleaningFunctions.remove_stopwords)
reviewsNa['tokenized'] = reviewsNa["tokenized"].apply(cleaningFunctions.remove_punctuation)

# STEMMING
reviewsNa['tokenized'] = reviewsNa["tokenized"].apply(cleaningFunctions.stemming)# Reduces word to base word (Removes suffixes and prefixes )


#reviewsNa['sentiment'] = reviewsNa["tokenized"].apply(cleaningFunctions.sentiment) # Providing the polarity and subjectivity
reviewsNa['polarity'] = reviewsNa["tokenized"].apply(cleaningFunctions.polarity) # isolating the polarity

# Checking if it is +VE or -VE

reviewsNa['comboPolarity'] = reviewsNa.apply(cleaningFunctions.combine,axis=1)
reviewsNa['positive'] = reviewsNa["comboPolarity"].apply(cleaningFunctions.positive)# Give a Bool for +VE and -VE

# Summation of +VE & -VE
true_count = (reviewsNa['positive'] == True).sum()
false_count = (reviewsNa['positive'] == False).sum()
total = true_count + false_count # Sum of all Cleaned rows in Data

# Visualistaion

# Histogram
seaborn.histplot(data=reviewsNa, x="comboPolarity", hue="Rating") # Creates a Histogram
plt.show()
seaborn.histplot(data=reviewsNa, x="polarity", hue="Rating") # Creates a Histogram
plt.show()

# Pie Chart
rating_counts = reviewsNa['Rating'].value_counts() # Counts the Number of rows in the colour
labels = rating_counts.index # Creates Labels to group the different Ratings
colours = seaborn.color_palette('pastel')[0:3] # Add 3 pastel colours for the Pie
plt.pie(rating_counts,labels=labels,colors=colours, autopct='%1.1f%%')
plt.title("Distribution of Ratings") # Title for the plot
plt.show()

# Pie Chart
reviewsNa['positive'] = reviewsNa['positive'].replace({True: "Positive", False: "Negative"})
positive_counts = reviewsNa['positive'].value_counts() # Counts the Number of rows in the colour
labels = positive_counts.index # Creates Labels to group the different Ratings
colours = seaborn.color_palette('pastel')[0:2] # Add 3 pastel colours for the Pie
plt.pie(positive_counts,labels=labels,colors=colours, autopct='%1.1f%%')
plt.title("Distribution of Positive Reviews") # Title for the plot
plt.show()

# Scatter Graph
seaborn.regplot(data=reviewsNa, x="polarity", y="Rating")
plt.title("Polarity vs Ratings")
plt.show()

seaborn.regplot(data=reviewsNa, x="comboPolarity", y="Rating")
plt.title("Combined Polarity vs Ratings")
plt.show()

# Box Plot
seaborn.boxplot(data=reviewsNa, x="Rating", y="polarity",whis=1.5)
seaborn.swarmplot(data=reviewsNa, x="Rating", y="polarity", color=".25")
plt.title("Polarity vs Ratings")
plt.show()

# Finding Popular Word in +VE reviews
positive_review  = reviewsNa[reviewsNa['positive'] == "Positive"].copy()
negative_review = reviewsNa[reviewsNa['positive'] == "Negative"].copy()


positive_review['tokenized'] = positive_review['tokenized'].apply(cleaningFunctions.spelling_correct)
negative_review['tokenized'] = negative_review['tokenized'].apply(cleaningFunctions.spelling_correct)
positive_review['tokenized'] = positive_review['tokenized'].apply(cleaningFunctions.remove_stopwords)
negative_review['tokenized'] = negative_review['tokenized'].apply(cleaningFunctions.remove_stopwords)

pos_words = [word for tokens in positive_review["tokenized"] for word in tokens]
pos_text = " ".join(pos_words) # Joins all the +VE text together
pos_wordCloud = WordCloud(width=800, height=400, max_words=10).generate(pos_text)
plt.imshow(pos_wordCloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Frequent Positive Words")
plt.show()

neg_words = [word for tokens in negative_review["tokenized"] for word in tokens]
neg_text = " ".join(neg_words) # Joins all the +VE text together
neg_wordCloud = WordCloud(width=800, height=400, max_words=10).generate(neg_text)
plt.imshow(neg_wordCloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Frequent Negative Words")
plt.show()

lstNgWrds = neg_wordCloud.words_ # list of 10 top Positive words
lstPsWrds = pos_wordCloud.words_ # list of 10 top Negative words
truePer = true_count / total  # % of positive review
falsePer = false_count / total # % of negative review


avg_rating = reviewsNa['Rating'].mean() # average rating over the year