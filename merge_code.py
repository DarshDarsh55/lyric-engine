import os

base_dir = '/home/marshthedarsh55/'  # This is your actual home directory
extensions = ['.py', '.java', '.js', '.txt']  # Add other file types if needed
output_path = os.path.join(base_dir, 'all_code_combined.txt')

with open(output_path, 'w') as outfile:
    for foldername, _, filenames in os.walk(base_dir):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in extensions):
                full_path = os.path.join(foldername, filename)
                outfile.write(f"\n\n# ===== {full_path} =====\n")
                with open(full_path, 'r') as infile:
                    outfile.write(infile.read())
