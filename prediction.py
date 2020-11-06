import pandas as pd

# Read data
df = pd.read_csv('incident.csv', encoding='ISO-8859-1')

# Dropping rows with NaN target value
df.dropna(subset=['Assignment group', 'Description'], inplace=True)

# NLP Modeling
X = df['Description']
y = df['Assignment group']

# Encoding target variable
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder().fit(y)
y = pd.Series(le.transform(y))

# Splitting Data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Vectorisation of description
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words='english', min_df=5).fit(X_train)
vector = cv.transform(X_train)

# Logistic regression
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression().fit(vector, y_train)

# Pickleing model
import pickle

file = open('cv.pkl', 'wb')
pickle.dump(cv, file)
file.close()

file2 = open('clf.pkl', 'wb')
pickle.dump(clf, file2)
file2.close()

file3 = open('le.pkl', 'wb')
pickle.dump(le, file3)
file3.close()


# Prediction
def predict(description):
    prediction = clf.predict(cv.transform([description]))
    return le.inverse_transform(prediction)[0]


message = '''Not able to open excel in projectwise'''
print(predict(message))
