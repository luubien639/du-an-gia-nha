import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('data.csv')
model = LinearRegression()
model.fit(df[['SquareMeters', 'Bedrooms']], df['Price'])
print("Mo hinh da huan luyen xong!")