import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

# Open the spreadsheet file and read it in
data = pd.read_csv("fruits.csv")
print(data)

# Data validation: flag watermelons too light to be real
is_watermelon = data["label"] == "watermelon"
too_light = data["weight"] < 1000
suspicious = data[is_watermelon & too_light]

print("Suspicious rows:")
print(suspicious)

# Split the table: the feature columns, and the answer column
fruits = data[["weight", "skin", "color"]].values
labels = data["label"].values

model = DecisionTreeClassifier()
model.fit(fruits, labels)

# An ordinary red apple. With the poisoned row in the data, the model calls it a watermelon.
print(model.predict([[160, 0, 2]]))

# Show the rules the model learned, including where the poison landed
print(export_text(model, feature_names=["weight", "skin", "color"]))
