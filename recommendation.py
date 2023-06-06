

import numpy as np
import pandas as pd
from statistics import harmonic_mean
from langdetect import detect
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('coursea_data.csv')
df.drop(['Unnamed: 0', 'course_organization'], axis=1, inplace=True)

df = df[df.course_students_enrolled.str.endswith('k')]
df['course_students_enrolled'] = df['course_students_enrolled'].apply(lambda enrolled: eval(enrolled[:-1]) * 1000)

minmax_scaler = MinMaxScaler()
scaled_ratings = minmax_scaler.fit_transform(df[['course_rating', 'course_students_enrolled']])
df['course_rating'] = scaled_ratings[:, 0]
df['course_students_enrolled'] = scaled_ratings[:, 1]
df['overall_rating'] = df[['course_rating', 'course_students_enrolled']].apply(lambda row: harmonic_mean(row), axis=1)

df = df[df.course_title.apply(lambda title: detect(title) == 'en')]

vectorizer = TfidfVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(df.course_title)


def recommend_by_course_title(title, recomm_count=2):
    title_vector = vectorizer.transform([title])
    cosine_sim = cosine_similarity(vectors, title_vector)
    idx = np.argsort(np.array(cosine_sim[:, 0]))[-recomm_count:]
    sdf = df.iloc[idx].sort_values(by='overall_rating', ascending=False)
    return sdf
