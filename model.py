import pandas as pd
#import pandas_profiling as pp

df = pd.read_csv("Combined_powercycle.csv")

print(df.dtypes)
#profile = pp.profileReport(df)
#profile.to_file("edAreport.html")

df.drop_duplicates()
import sklearn.model_selection as Md
y = df["PE"]
X = df.drop("PE",axis=1)
X_train,X_test,y_train,y_test = Md.train_test_split (X,y,test_size = 0.20)
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
#print(y_pred)
#print(y_test)
from sklearn.metrics import mean_squared_error,mean_absolute_error
mse = mean_squared_error(y_pred,y_test)
mae = mean_absolute_error(y_pred,y_test)

print(mse,mae)
import pickle
#serialisation
pickle.dump(regressor,open('model.pkl','wb'))
#deserialisation
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[15,42,1000,75]]))




