import json
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Import your updated DST
from emotional_dst import process_utterance

# ===== 1. Define your labelled test set =====
# Replace with your real evaluation data
test_data = [
    ("I am furious about the delay", "angry"),
    ("I feel so nervous about tomorrow", "anxious"),
    ("I'm just sitting here with nothing to do", "bored"),
    ("Life is beautiful today", "happy"),
    ("I miss my friends so much", "lonely"),
    ("I feel down and hopeless", "sad"),
    ("Too much work is stressing me out", "stressed"),
    # Add more examples per class for balance
]

# ===== 2. Run predictions =====
y_true = []
y_pred = []

for text, true_label in test_data:
    result = process_utterance(text)
    pred_label = result["mapped_emotion"]
    y_true.append(true_label)
    y_pred.append(pred_label)
    print(f"Text: {text}\nTrue: {true_label} | Pred: {pred_label}\n")

# ===== 3. Metrics =====
print("\nClassification Report:")
print(classification_report(y_true, y_pred, labels=list(set(y_true))))

# ===== 4. Confusion Matrix =====
labels = sorted(list(set(y_true)))  # unique labels in test set
cm = confusion_matrix(y_true, y_pred, labels=labels)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Emotion Prediction Confusion Matrix (Refined Model)")
plt.show()
