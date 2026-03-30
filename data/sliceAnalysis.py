# Confusion Matrix
print("\nGenerating Slice Analysis")
y_pred = np.argmax(y_probs, axis=1)
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_labels,
    yticklabels=class_labels,
    cmap="Blues",
)
plt.title("Confusion Matrix")
plt.show()
print(classification_report(y_true, y_pred, target_names=class_labels))
