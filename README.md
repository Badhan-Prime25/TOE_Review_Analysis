# Customer Review Sentiment Analysis

**Objective:** Analyse Customer satisfaction during the first year of new new management at Touch of East using Google Reviews.

### Steps
- Data collection & preprocessing (remove stopwords, stemming, toeknization)
- Sentiment Classification (TextBlob Polarity {0.3 threshold})
- Correlation Analysis between ratings and polarity (Normalising & Combining Polarity with Rating)
- Visulalisation: word cloud for positive/negative words, scatter plots, histograms

  ### Results
  - 96.6% reviews were positive; average rating 4.85/5
  - Stronger alignment between rating & polarity after normalising
  - Common positive themes: "friendly", "staff", "love"
  - Common Negative Themes in the Isolated Events: "never", "return", "rush"

📊Tools: Python, Pandas, NLTK, TextBlob, Matplotlib
📄[Read Full Report] (https://github.com/Badhan-Prime25/TOE_Review_Analysis/blob/main/Customer%20Review%20Analysis.pdf)
