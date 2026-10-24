from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text

# Each fruit = [weight in grams, skin, color]
# Skin: 0 = smooth, 1 = bumpy
# Color: 2 = red, 3 = orange, 4 = green
fruits = [
    [150, 0, 2],    # apple
    [170, 0, 4],    # apple
    [140, 1, 3],    # orange
    [130, 1, 3],    # orange
    [4500, 0, 4],   # watermelon (new!)
    [6000, 0, 4],   # watermelon (new!)
]

labels = ["apple", "apple", "orange", "orange", "watermelon", "watermelon"]

model = DecisionTreeClassifier()
model.fit(fruits, labels)

print(model.predict([[5000, 0, 4]]))

# Show the rules the model wrote for itself
print(export_text(model, feature_names=["weight", "skin", "color"]))
