import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb


# Preprocess the dataset:
# takes in combined data set comprised of
# marks which data came from which set with additional label column
# drops noise columns of glasses, hair and instance
# fixes strange nan values, replaces 0 with 'zero' in mixed column
# converts work experience and additional income column into numeric
# fills na values in numerical columns with median value of column
# fills na values in categorical columns with 'unspecified' value
# finally splits the dataset again and returns training x and y, as well as testing x separately
def preprocess(total):
    # drop worthless noise columns
    total = total.drop('Wears Glasses', axis=1)
    total = total.drop('Hair Color', axis=1)
    total = total.drop('Instance', axis=1)

    # replace actual 0 with string zero so pandas doesn't get confused
    total['Housing Situation'].replace(0, 'zero')

    # replace strange NaN values
    total = total.replace("nA", np.NaN)
    total['Work Experience in Current Job [years]'] = \
        total['Work Experience in Current Job [years]'].replace('#NUM!', np.nan)

    # convert work experience to numeric and replace nans with median value of column
    total['Work Experience in Current Job [years]'] = pd.to_numeric(total['Work Experience in Current Job [years]'])
    total['Work Experience in Current Job [years]'].fillna(total['Work Experience in Current Job [years]'].median(),
                                                           inplace=True)

    # convert additional income to numeric and replace nans with median value of column
    total['Yearly Income in addition to Salary (e.g. Rental Income)'] = total.apply(
            lambda row: float(row['Yearly Income in addition to Salary (e.g. Rental Income)'].split()[0]), axis=1
            )

    # fill nans of numericals with median values
    total['Year of Record'].fillna(total['Year of Record'].median(), inplace=True)
    total['Age'].fillna(total['Age'].median(), inplace=True)
    total['Body Height [cm]'].fillna(total['Body Height [cm]'].median(), inplace=True)

    # fill nans of categoricals with 'unspecified' value
    total['Gender'] = total['Gender'].fillna('unspecified')
    total['University Degree'] = total['University Degree'].fillna('unspecified')
    total['Profession'].fillna('unspecified', inplace=True)
    total['Country'].fillna('unspecified', inplace=True)
    total['Satisfation with employer'].fillna('unspecified', inplace=True)

    # split combined set back into constituents
    return_train_x = total[total['is_training'] == 1]
    return_test_x = total[total['is_training'] == 0]

    # remove index column used to tell them apart
    return_train_x = return_train_x.drop('is_training', axis=1)
    return_test_x = return_test_x.drop('is_training', axis=1)

    return return_test_x, return_train_x


# read in data from csv files
training_set = pd.read_csv('tcd-ml-1920-group-income-train.csv')
testing_set = pd.read_csv('tcd-ml-1920-group-income-test.csv')

# isolate x and y from training, get x of testing data set
train_x = pd.DataFrame(training_set.iloc[:, :-1])
train_y = pd.Series(training_set['Total Yearly Income [EUR]'])
test_x = pd.DataFrame(testing_set.iloc[:, :-1])

# establish index to keep track of which data came from where when combined
train_x['is_training'] = 1
test_x['is_training'] = 0
combined = pd.concat([train_x, test_x])

# preprocess both sets
X_test, training_X = preprocess(combined)

# define categorical columns
categorical_cols = ['Housing Situation', 'Satisfation with employer', 'Gender', 'Country', 'Profession',
                    'University Degree']

# target encode categorical columns
target_encoder = ce.TargetEncoder(cols=categorical_cols)
training_X = target_encoder.fit_transform(training_X, train_y)
X_test = target_encoder.transform(X_test)

# split data into test and train sets, only used for local MAE checking
x_train, x_test, y_train, y_test = train_test_split(training_X, train_y, test_size=0.2)

# define training and testing data for lgb model
training = lgb.Dataset(training_X, label=train_y)
# testing = lgb.Dataset(x_test, label=y_test)

# define lgb parameters
parameters = {"boosting": "gbdt", 'learning_rate': 0.002, "num_leaves": 210,
              "n_jobs": 4, "verbosity": 2, 'max_depth': 24}

# commence training of model
print("Starting Model Training...")
model = lgb.train(parameters, training, 115000, verbose_eval=500)
print("Model Trained, Predicting Reals...")
# predict values for output
predicted_values = model.predict(X_test)

# predict values to compare to real values for MAE calculation
print("Predicting Fakes...")
predicted_test_values = model.predict(x_test)

# calculate MAE from predicted and real data
print("Mean Absolute Error: ", mean_absolute_error(y_test, predicted_test_values))

# write the real predicted data to disk
print("Writing out...")
submission = pd.read_csv('tcd-ml-1920-group-income-submission.csv')
submission['Total Yearly Income [EUR]'] = predicted_values
submission.to_csv('tcd-ml-1920-group-income-submission.csv', index=False)

# All Done
print("Finished writing, exiting...")
