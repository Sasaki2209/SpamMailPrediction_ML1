import pandas as pd
import random

# email content
sentences = [

]

# Generate labels for spam (1) and ham (0)
labels = [random.choice(['ham', 'spam']) for _ in range(len(sentences))]

# Create a DataFrame
df = pd.DataFrame({
    'Category': labels,
    'Message': sentences
})

# Save to CSV
df.to_csv('mail.csv', index=False, encoding='utf-8')

print("CSV file has been generated successfully.")
