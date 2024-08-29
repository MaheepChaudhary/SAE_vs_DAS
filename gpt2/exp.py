def read_file_in_batches(filename, batch_size=11):
    latex_final_list = []

    # Read the entire file and split into lines
    with open(filename, "r") as file:
        lines = [line.strip() for line in file.readlines()]

    # Process lines in batches
    for i in range(0, len(lines), batch_size):
        batch = lines[i : i + batch_size]
        latex_final_list.append(batch)

    return latex_final_list


# Example usage
filename = "latex_table.txt"
latex_final_list = read_file_in_batches(filename)

# Output the batches
for idx, batch in enumerate(latex_final_list):
    print(f"Batch {idx + 1}: {batch}")
