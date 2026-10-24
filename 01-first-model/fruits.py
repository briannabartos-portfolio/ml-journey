from sklearn.tree import DecisionTreeClassifier

# Each fruit = [weight in grams, skin, color]
# Skin: 0 = smooth, 1 = bumpy
# Color: 2 = red, 3 = orange, 4 = green
fruits = [[150, 0, 2], [170, 0, 4], [140, 1, 3], [130, 1, 3]]
labels = ["apple", "apple", "orange", "orange"]

model = DecisionTreeClassifier()
model.fit(fruits, labels)

# A watermelon: 5000 grams, smooth skin, green
# The model has only seen apples and oranges, so it confidently guesses one of them
print(model.predict([[5000, 0, 4]]))
