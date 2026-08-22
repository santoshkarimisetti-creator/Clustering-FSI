print("Sample of Gujarati Test Predictions vs. True Labels:")
for i in range(10):
    print(f"Prediction: {gujarati_y_pred[i]}, True Label: {gujarati_y_test[i]}")



print("Sample of Devanagari Test Predictions vs. True Labels:")
for i in range(10):
    print(f"Prediction: {devanagari_y_pred[i]}, True Label: {devanagari_y_test[i]}")



print("\nSample of Devanagari Test Predictions vs. True Labels (Character Names):")
for i in range(10):
    predicted_char = devanagari_class_names[devanagari_y_pred[i]]
    true_char = devanagari_class_names[devanagari_y_test[i]]
    print(f"Prediction: {predicted_char:<25}, True Label: {true_char}")