from scipy.sparse import hstack, csr_matrix
import pandas as pd
import joblib
import lightgbm as lgb

# Load the trained vectorizers and LabelBinarizer
cv_name = joblib.load('cv_name.pkl')
cv_category = joblib.load('cv_category.pkl')
tv_description = joblib.load('tv_description.pkl')
lb_brand = joblib.load('lb_brand.pkl')

# Load the trained LightGBM model
gbm = lgb.Booster(model_file='lgb_model.txt')




# Make prediction function
def make_prediction(processed_data):

    # Example: Create a DataFrame with a single row using the processed input data
    input_df = pd.DataFrame(processed_data, index=[0])

    # Convert 'item_condition_id', 'category_name', and 'brand_name' to category type
    input_df['item_condition_id'] = input_df['item_condition_id'].astype('category')
    input_df['category_name'] = input_df['category_name'].astype('category')
    input_df['brand_name'] = input_df['brand_name'].astype('category')

    # Transform 'name', 'category_name', and 'item_description' using the trained vectorizers
    X_name = cv_name.transform(input_df['name'])
    X_category = cv_category.transform(input_df['category_name'])
    X_description = tv_description.transform(input_df['item_description'])

    # Transform 'brand_name' using the trained LabelBinarizer
    X_brand = lb_brand.transform(input_df['brand_name'])

    # Create dummy variables for 'item_condition_id'
    X_dummies = csr_matrix(pd.get_dummies(input_df[['item_condition_id']], sparse=True).values)

    # Combine all the features
    input_features = hstack((X_dummies, X_description, X_brand, X_category, X_name)).tocsr()

    # Use the trained LightGBM model to make predictions
    predicted_price = gbm.predict(input_features, num_iteration=gbm.best_iteration)[0]

    return predicted_price