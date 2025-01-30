
# Mobile Price Classification using AWS SageMaker


## Overview
This project builds a machine learning model to classify mobile phone price ranges using AWS SageMaker. The dataset is preprocessed and split into training and testing sets, and a Random Forest model is trained using SageMaker’s managed ML services. The final trained model is deployed as an endpoint for inference.
## Key Highlights
- Data Processing: Prepares and splits dataset for training/testing.

- AWS SageMaker Integration: Uses SageMaker for data storage, training, and deployment.

- Random Forest Classifier: Trains a model using Scikit-Learn.

- Model Deployment: Deploys the trained model as an endpoint on AWS SageMaker.

- Evaluation Metrics: Reports accuracy and classification metrics.

##  Dependencies
Install the following Python libraries to run this project:

- Python 3.x

- AWS SDK (*boto3*)

- SageMaker SDK (*sagemaker*)

- Scikit-Learn (*sklearn*)

- Pandas (*pandas*)

- Joblib (*joblib*)

- NumPy (*numpy*)
## Methodology
- Data Loading & Processing:

  - Reads mobile price classification dataset (mpc_train.csv).

  - Checks for missing values and distribution of target variable.

  - Splits dataset into training (85%) and testing (15%) sets.

  - Saves preprocessed data as CSV files.

- Data Upload to S3:

  - Uploads training and testing datasets to an S3 bucket.

- Model Training:

  - Trains a Random Forest Classifier using SageMaker’s managed training environment.

  - Uses hyperparameters (n_estimators=100, random_state=0).

  - Saves the trained model as a .joblib file.

- Model Evaluation:

  - Computes accuracy and generates a classification report on the test set.

- Model Deployment:

  - Deploys the trained model as an AWS SageMaker endpoint for inference.

- Endpoint Management:

  - Deletes the endpoint after deployment to optimize resource usage.
## Results
- Training Data Shape: 85% of dataset

- Testing Data Shape: 15% of dataset

- Model Accuracy: Evaluated using accuracy score and classification report.

- Deployment: Model successfully deployed as an endpoint.
## Conclusion
This project demonstrates the integration of AWS SageMaker for training and deploying machine learning models. It leverages cloud-based training, reducing local computation needs and ensuring scalability.
## Future Improvements

- Optimize hyperparameters using SageMaker automatic tuning.

- Use more advanced models like XGBoost or deep learning.

- Implement real-time predictions and monitoring for deployed endpoints.

- Improve feature engineering to enhance model performance.
