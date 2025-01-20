# -Building-Recommendation-Systems-FAST-with-Turi-Create-by-Apple
Building a recommendation system with Turi Create is a simple and fast process, thanks to its high-level APIs designed for creating machine learning models efficiently. Turi Create is an open-source Python library from Apple that simplifies the process of building machine learning models, especially for tasks like recommendations, classification, regression, and image classification.
Objective:

We will create a movie recommendation system using Turi Create. This system will be based on a dataset of movie ratings (such as the MovieLens dataset) and will recommend movies based on user preferences.
Prerequisites:

    Install Turi Create: You will need to install the Turi Create library.

pip install turicreate

    Dataset: For this example, we’ll use the MovieLens 100K dataset. You can download this dataset from the MovieLens website.

Steps to Build the Recommendation System:

    Load Data: Load the dataset containing the ratings information (user ID, movie ID, rating).
    Create a Model: Use Turi Create to build a recommendation model.
    Evaluate the Model: Evaluate the model’s performance.
    Make Predictions: Use the model to predict movie recommendations.

Python Code Example

# Step 1: Import Libraries
import turicreate as tc

# Step 2: Load the dataset
# Replace this with the path to your MovieLens dataset (CSV file containing user-item interactions)
# Example: ratings.csv contains 'user_id', 'movie_id', and 'rating' columns.
data = tc.SFrame.read_csv('ratings.csv')  # Ensure you have the correct path

# Step 3: Data Overview
print(data.head())  # Display the first few rows to understand the structure of the dataset

# Step 4: Create a Recommender Model
# Turi Create provides an easy-to-use model for item recommendations based on user-item interactions
model = tc.recommender.create(data, user_id='user_id', item_id='movie_id', target='rating')

# Step 5: Evaluate the Model
# We can use the evaluate function to assess the performance of the recommendation model.
# It returns a dictionary of metrics, including precision, recall, and F1-score
evaluation = model.evaluate(data)
print("Model Evaluation:", evaluation)

# Step 6: Make Predictions (Recommendations)
# Now, let's use the model to make predictions for a specific user. We’ll recommend movies for user_id = 1
# Note: You can use any user_id from your dataset
user_id = 1
recommendations = model.recommend(users=[user_id], num_items=10)

print(f"Top 10 recommendations for user {user_id}:")
print(recommendations)

# Step 7: Save the Model (Optional)
# You can save the trained model for later use.
model.save('movie_recommender.model')

# Step 8: Load the Saved Model (Optional)
# You can load the model again for further use or predictions in the future.
# loaded_model = tc.load_model('movie_recommender.model')

Explanation of the Code:

    Loading the Data: We load the ratings.csv file, which contains the user-item interactions. Each row in the dataset represents a rating given by a user for a movie. The dataset typically has the columns user_id, movie_id, and rating.

    Creating the Recommender Model: Turi Create’s recommender.create() function is used to build a recommendation model. The parameters include:
        user_id: The column that represents the user.
        item_id: The column representing the items (in this case, movies).
        target: The column representing the rating given by users (the value we're predicting).

    Evaluating the Model: Turi Create’s evaluate() function gives us a performance summary, including precision, recall, and F1-score. This helps us understand the quality of the recommendation system.

    Making Predictions: After training the model, we use the recommend() function to get the top N recommendations for a user. In this case, we recommended the top 10 movies for user_id = 1.

    Saving and Loading the Model: Optionally, you can save your model using model.save() and load it back using tc.load_model() for later use.

Sample Output:

Top 10 recommendations for user 1:
+---------+-----------+-------+
| movie_id|   score   | user_id|
+---------+-----------+-------+
|  32     |  4.5      |   1   |
|  52     |  4.1      |   1   |
|  77     |  3.9      |   1   |
|  29     |  3.8      |   1   |
|  13     |  3.7      |   1   |
|  8      |  3.5      |   1   |
|  99     |  3.4      |   1   |
|  11     |  3.2      |   1   |
|  67     |  3.1      |   1   |
|  44     |  2.9      |   1   |
+---------+-----------+-------+

Here, the score column represents the predicted rating for each movie that the model recommends to the user. The movies are sorted based on the predicted score.
Key Points:

    Turi Create allows you to easily build recommendation systems with minimal code.
    Recommender Systems are based on collaborative filtering, where the model learns user preferences based on historical data.
    The evaluate function helps to validate the recommendation system by providing metrics like precision and recall.
    The recommend function generates the top N movie recommendations for a given user.

Advanced Usage:

    You can use additional information, such as movie metadata (e.g., genre, year) or user demographics (e.g., age, location), to build more advanced hybrid recommendation models.

Conclusion:

Using Turi Create, you can quickly build and deploy a recommendation system, reducing the complexity of traditional machine learning models. It's a perfect tool for rapid prototyping and solving recommendation problems with minimal effort.
