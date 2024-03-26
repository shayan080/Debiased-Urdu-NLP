import json

# Path to the original JSON file
original_file_path = 'urdu_gender_specific_full.json'
# Path for the new (or updated) JSON file
decoded_file_path = 'urdu_gender_specific_full.json'

# Step 1: Read the original JSON file
with open(original_file_path, 'r', encoding='utf-8') as file:
    content = json.load(file)  # The content is now in plain text (Python strings)

# Step 2 is implicitly handled by json.load()

# Step 3: Write the plain Urdu text to a new JSON file
with open(decoded_file_path, 'w', encoding='utf-8') as file:
    json.dump(content, file, ensure_ascii=False, indent=4)  # ensure_ascii=False allows for Urdu characters to be written properly

print("Conversion completed. The decoded content is saved in:", decoded_file_path)
