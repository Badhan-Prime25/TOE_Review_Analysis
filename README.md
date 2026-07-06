# Customer Review Sentiment Analysis – Touch of East

## Overview
This project analyses Google Review data for Touch of East, a beauty and wellness business, during its first year under new management. The aim was to understand customer satisfaction, identify common themes in feedback, and compare written review sentiment with customer star ratings.

## Objectives
- Collect and clean customer review data
- Classify reviews as positive or negative using sentiment polarity
- Compare TextBlob sentiment polarity with Google star ratings
- Identify common positive and negative review themes
- Visualise customer satisfaction using charts and word clouds

## Tools Used
- Python
- Pandas
- NLTK
- TextBlob
- Matplotlib
- WordCloud

## Methodology
The reviews were cleaned by removing stopwords, punctuation, and applying tokenisation and stemming. Sentiment analysis was then performed using TextBlob polarity scores. Reviews were classified as positive or negative based on a polarity threshold.

The analysis also compared review polarity with Google star ratings. A combined normalised rating and polarity metric was created to better understand overall customer satisfaction.

## Key Findings
- 96.6% of analysed reviews were classified as positive
- The average Google rating was 4.85 out of 5
- Positive reviews commonly mentioned words such as "friendly", "staff", and "love"
- Negative reviews appeared to be isolated incidents rather than consistent service issues
- Combining normalised ratings with sentiment polarity showed a stronger alignment with overall customer satisfaction

## Visualisations
The project includes:
- Sentiment distribution pie chart
- Rating distribution chart
- Polarity vs rating scatter plot
- Combined polarity and rating analysis
- Positive and negative word clouds

## Conclusion
The analysis suggests that Touch of East achieved a high level of customer satisfaction during its first year under new management. Most customers gave highly positive feedback, particularly around staff professionalism, friendliness, and service quality.

## Future Improvements
- Use a larger review dataset over a longer time period
- Compare results before and after the management change
- Use more advanced NLP models such as VADER or transformer-based sentiment models
- Add topic modelling to identify specific service areas mentioned in reviews

📄[Read Full Report](https://github.com/Badhan-Prime25/TOE_Review_Analysis/blob/main/Customer%20Review%20Analysis.pdf)
