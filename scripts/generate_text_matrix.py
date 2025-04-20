import json
import os

# Define the variables
models = ["pixtral", "llama", "doct"]
quant_levels = [4, 8, 16, 32]
prompt_types = ["basic_extraction", "detailed", "few_shot", "locational", "step_by_step"]
image_numbers = [1017, 1018, 1019, 1020, 1021, 1022, 1023, 1025, 1026, 1027, 
                1028, 1029, 1030, 1031, 1038, 1039, 1040, 1041, 1042, 1043]

# Create the test cases array
test_cases = []

# Iterate through all combinations
for model in models:
    for quant_level in quant_levels:
        for prompt_type in prompt_types:
            for image_number in image_numbers:
                test_case = {
                    "model": model,
                    "quantization_level": quant_level,
                    "prompt_type": prompt_type,
                    "image_number": image_number,
                    "field_type": "both",
                    "image_path": f"data/images/{image_number}.jpg"
                }
                test_cases.append(test_case)

# Create the final test matrix structure
test_matrix = {
    "test_cases": test_cases
}

# Calculate total entries
total_entries = len(test_cases)
print(f"Total test cases: {total_entries}")

# Get the current directory path
current_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(current_dir, 'config', 'test_matrix.json')

# Ensure the config directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Save to a JSON file
with open(output_file, 'w') as f:
    json.dump(test_matrix, f, indent=2)

print(f"Test matrix file '{output_file}' has been created with {total_entries} test cases.")

# Verify the structure by checking a sample entry
sample = test_cases[0]
print(f"Sample test case: {sample}")