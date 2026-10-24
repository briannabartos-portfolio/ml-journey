from sklearn.tree import DecisionTreeClassifier

# Each fruit = [weight in grams, skin, color]
# Skin: 0 = smooth, 1 = bumpy
# Color: 2 = red, 3 = orange, 4 = green
fruits = [[150, 0, 2], [170, 0, 4], [140, 1, 3], [130, 1, 3]]
labels = ["apple", "apple", "orange", "orange"]

model = DecisionTreeClassifier()
model.fit(fruits, labels)

# Describe the fruit we want to ask about
weight = 5000
skin = 0
color = 4

# Safeguard: our heaviest training fruit was 170 grams
if weight > 300:
    print("I'm not sure. This fruit is unlike anything I learned from.")
else:
    print(model.predict([[weight, skin, color]]))
