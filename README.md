import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

# 1. Define dataset columns
columns = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']

# 2. Load the dataset
df = pd.read_csv(r"C:\Users\dell\OneDrive\Desktop\PraveenML\magic+gamma+telescope\magic04.data", names=columns)

# 3. Preprocess target variable (re-assigning g=1 and h=0)
df['class'] = (df['class'] == 'g').astype(int)

# 4. Split dataset into train, validation, and test sets (60% / 20% / 20%)
train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

# 5. Define data scaling and oversampling function
def scale_data(dataframe, oversample=False):
    x = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    if oversample:
        ros = RandomOverSampler()
        x, y = ros.fit_resample(x, y)
        
    data = np.hstack((x, np.reshape(y, (-1, 1))))

    return data, x, y

# 6. Convert back to DataFrames for scaling processing
train = pd.DataFrame(train)
valid = pd.DataFrame(valid)
test = pd.DataFrame(test)

# 7. Scale and balance data splits
train, X_train, y_train = scale_data(train, oversample=True)
valid, X_valid, y_valid = scale_data(valid, oversample=False)
test, X_test, y_test = scale_data(test, oversample=False)

# 8. Train Gaussian Naive Bayes Model
nb = GaussianNB()
nb.fit(X_train, y_train)
