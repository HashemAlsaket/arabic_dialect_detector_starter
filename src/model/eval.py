# Evaluate on the test set
results = trainer.evaluate(tokenized_datasets["test"])
print(results)
