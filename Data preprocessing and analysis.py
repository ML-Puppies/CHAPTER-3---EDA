import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import  MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re

#CHAPTER 3: DATA PREPROCESSING AND ANALYSIS
#3.1. Understanding the Data
#3.1.1. Introduction to the Dataset

movies = pd.read_csv("../DATASET/movies.csv", encoding='ISO-8859-1', dtype='unicode')
print("Sample data of movies dataset")
print(movies.head())
ratings = pd.read_csv("../DATASET/ratings.csv", encoding='ISO-8859-1', dtype='unicode')
print("Sample data of ratings dataset")
print(ratings.head())
print("Shape of movies dataset: ", movies.shape)
print("Shape of ratings dataset: ", ratings.shape)

#3.1.2. Description of Variables
# Checking basic information about the dataset
print(movies.info())
print(ratings.info())

# Display summary statistics for numerical columns
print(movies.describe())
print(ratings.describe())

# Display unique values count for categorical columns
categorical_columns_movies = movies.select_dtypes(include=['object']).columns
for col in categorical_columns_movies:
    print(f"{col}: {movies[col].nunique()} unique values")
categorical_columns_ratings = ratings.select_dtypes(include=['object']).columns
for col in categorical_columns_ratings:
    print(f"{col}: {ratings[col].nunique()} unique values")

# checking data types of each column
print("movies dtype: ", movies.dtypes)
print("ratings dtype:", ratings.dtypes)

#3.2. Raw Data Preprocessing
#3.2.1. Checking and Handling Missing Data
# Checking for missing values
print("movies missing values",movies.isnull().sum())
print("ratings missing values",ratings.isnull().sum())
#3.2.2. Converting Formats, Data Types, Column Names
#Converting Data Types
movies['movieId'] = pd.to_numeric(movies['movieId'], errors='coerce')  # Chuyển về dạng số
ratings['userId'] = pd.to_numeric(ratings['userId'], errors='coerce')
ratings['movieId'] = pd.to_numeric(ratings['movieId'], errors='coerce')
ratings['rating'] = pd.to_numeric(ratings['rating'], errors='coerce')
ratings['timestamp'] = pd.to_numeric(ratings['timestamp'], errors='coerce')  # Giữ timestamp dạng số

print(movies.dtypes)
print(ratings.dtypes)

#3.2.3. Data Normalization
# 1. Convert timestamp to datetime
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

# 2. Extract useful time-related features
ratings['year'] = ratings['timestamp'].dt.year
ratings['month'] = ratings['timestamp'].dt.month
ratings['day_of_week'] = ratings['timestamp'].dt.dayofweek
ratings['hour'] = ratings['timestamp'].dt.hour

# 3. Calculate elapsed time (years since the last rating)
latest_timestamp = ratings['year'].max()
ratings['elapsed_time'] = latest_timestamp - ratings['year']

# 4. Drop original timestamp column
ratings.drop(columns=['timestamp'], inplace=True)

# 5. Save preprocessed ratings
ratings.to_csv("../DATASET/processed_ratings.csv", index=False)
print("3.2.3 Data Normalization completed!")

# 3.3. Exploratory Data Analysis (EDA)
# 1. Distribution of ratings
plt.figure(figsize=(8, 5))
sns.histplot(ratings['rating'], bins=10, kde=True)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# 2. Top 10 most rated movies
top_movies = ratings['movieId'].value_counts().head(10)
top_movies = top_movies.to_frame().reset_index()
top_movies.columns = ['movieId', 'num_ratings']
top_movies = top_movies.merge(movies[['movieId', 'title']], on='movieId', how='left')

plt.figure(figsize=(10, 5))
sns.barplot(x='num_ratings', y='title', data=top_movies, hue='title', palette='viridis')
plt.title('Top 10 Most Rated Movies')
plt.xlabel('Number of Ratings')
plt.ylabel('Movie Title')
plt.show()

# 3. Top 10 most active users
top_users = ratings['userId'].value_counts().head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x=top_users.index.astype(str), y=top_users.values, palette='magma')
plt.title('Top 10 Users with Most Ratings')
plt.xlabel('User ID')
plt.ylabel('Number of Ratings')
plt.xticks(rotation=45)
plt.show()

# 4. Average rating per genre
movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))
mlb = MultiLabelBinarizer()
genres_encoded = pd.DataFrame(mlb.fit_transform(movies['genres']), columns=mlb.classes_)
# Ensure movieId is index
genres_encoded.index = movies['movieId']
# Merge genres_encoded with movies for analysis
movies = movies.join(genres_encoded)

# Calculate average rating per genre
avg_movie_rating = ratings.groupby('movieId')['rating'].mean().to_frame()
avg_movie_rating = avg_movie_rating.reindex(genres_encoded.index).fillna(0)  # Reindex để đảm bảo đúng shape

# Compute genre average rating
genre_avg_rating = genres_encoded.T.dot(avg_movie_rating['rating'])
genre_avg_rating = genre_avg_rating / genres_encoded.sum()

plt.figure(figsize=(12, 5))
sns.barplot(x=genre_avg_rating.index, y=genre_avg_rating.values, palette='coolwarm')
plt.title('Average Rating per Genre')
plt.xlabel('Genre')
plt.ylabel('Average Rating')
plt.xticks(rotation=90)
plt.show()

# 5. Rating trend over years
rating_trend = ratings.groupby('year')['rating'].mean()

plt.figure(figsize=(10, 5))
sns.lineplot(x=rating_trend.index, y=rating_trend.values, marker='o')
plt.title('Average Rating Over Years')
plt.xlabel('Year')
plt.ylabel('Average Rating')
plt.show()

print("3.3 Exploratory Data Analysis completed!")

# 3.4. Data Preprocessing for the Model

from scipy.sparse import csr_matrix

# Chuyển đổi userId và movieId về index liên tục để tránh giá trị lớn
ratings['userId'] = ratings['userId'].astype("category").cat.codes
ratings['movieId'] = ratings['movieId'].astype("category").cat.codes

# Create Sparse Matrix
user_item_sparse = csr_matrix((ratings['rating'], (ratings['userId'], ratings['movieId'])))

print("Sparse Matrix Shape:", user_item_sparse.shape)

# 3.4.1. Collaborative Filtering

# Lưu ma trận dưới dạng sparse (để tránh lỗi bộ nhớ)
import scipy.sparse
scipy.sparse.save_npz("../DATASET/user_item_sparse.npz", user_item_sparse)

# 3.4.2. Content-Based Filtering
# Convert genres column from list to a string
# Kiểm tra và xử lý NaN trước khi chuyển đổi
movies['genres'] = movies['genres'].fillna('')  # Thay NaN bằng chuỗi rỗng

# Chuyển đổi thành chuỗi văn bản
movies['genres_str'] = movies['genres'].astype(str)

# Kiểm tra nếu toàn bộ genres đều trống
if movies['genres_str'].str.strip().eq('').all():
    raise ValueError("Error: All genres are empty after processing!")

# Apply TF-IDF transformation
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres_str'])

# Convert TF-IDF matrix to DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=movies['movieId'], columns=tfidf.get_feature_names_out())

# Save processed datasets for model training
tfidf_df.to_csv("../DATASET/tfidf_genres.csv")

# 3.4.3. Tạo cột năm công chiếu phim trích từ "title" trong movies dataset
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
# Display result
print(movies)
# Save the new dataset to a CSV file
movies.to_csv("../DATASET/movies_with_year.csv", index=False)
print("3.4 Data Preprocessing for the Model completed!")
