def write_data(data, file_path):
    with open(file_path, 'w') as f:
        for item in data:
            f.write(f"{item}\n")
