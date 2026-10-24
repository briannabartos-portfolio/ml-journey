# ML Journey

Learning machine learning and AI safety from zero, one small experiment at a time.

I started this repository knowing almost nothing about how machine learning works. Each folder is one lesson: a small, runnable experiment plus notes on what I learned. The goal is to understand not just how models learn, but how they fail and how to make them safer.

> These are learning notes.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

Run any file from inside its folder, for example:

```bash
cd 03-data-poisoning
python3 load_fruits.py
```

## Learning log

### 01: My first model

**Files:** `01-first-model/`

- Regular programming means a person writes the rules. Machine learning means you give the computer labeled examples and it works out the rules itself.
- Each example is made of **features** (facts turned into numbers, like weight, skin, and color) and a **label** (the answer, like "apple").
- I trained a `DecisionTreeClassifier` from scikit-learn to tell apples from oranges.
- When I asked it about a watermelon, it confidently said "apple." It had only ever seen apples and oranges, so it had to pick one. This is an **out-of-distribution** error.
- After adding watermelon examples, I used `export_text` to print the rules the model wrote for itself. It ignored color entirely and learned the shortcut "bumpy means orange," which would call a pumpkin an orange (**shortcut learning**).

### 02: Safeguards

**Files:** `02-safeguards/`

- A **safeguard** is a check that runs before the model answers, so it can say "I'm not sure" instead of guessing.
- I practiced `if` / `else` and learned that edge cases matter: `temp < 60` and `temp <= 60` give different answers at exactly 60.
- Hand-written safeguards only catch the cases someone thought of in advance, so real AI safety stacks several imperfect layers (the **Swiss cheese model**): better data, safeguards, testing, and human review.

### 03: Data poisoning and data validation

**Files:** `03-data-poisoning/`

- I moved the training data out of the code and into `fruits.csv`, then loaded it with pandas. Adding a pumpkin row taught the model a new fruit without changing any code.
- That means whoever controls the data controls the model. To test this, I added one deliberately mislabeled row (`155,0,2,watermelon`). That single fake row made the model call ordinary red apples "watermelons." This is **data poisoning**.
- Printing the model's rules showed exactly where the poison landed, an example of **interpretability**.
- The model can't notice a 155-gram "watermelon" is impossible. A person has to. So I wrote a **data validation** check that flags any watermelon under 1,000 grams before training.
- Choosing the 1,000-gram line was a trade-off: too strict causes false alarms on real small watermelons, too loose lets sabotage through.

*The poisoned row in `fruits.csv` is intentional, so the validation check has something to catch.*
