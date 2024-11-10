def predict_dialect(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax(-1).item()
    label = model.config.id2label[predicted_class]
    return label

# Example usage
text = "أنا ذاهب إلى السوق"  # Example Arabic sentence
predicted_dialect = predict_dialect(text)
print(f"Predicted Dialect: {predicted_dialect}")
