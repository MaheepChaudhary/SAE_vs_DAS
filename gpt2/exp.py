from pprint import pprint


def create_latex_table(data, headers):
    # Begin the LaTeX table environment
    latex_code = "\\begin{table}[h!]\n"
    latex_code += "\\centering\n"
    latex_code += "\\begin{tabular}{|" + " | ".join(["c"] * len(headers)) + "|}\n"
    latex_code += "\\hline\n"

    # Add headers
    latex_code += " & ".join(headers) + " \\\\\n"
    latex_code += "\\hline\n"

    # Add table data
    for row in data:
        latex_code += " & ".join(map(str, row)) + " \\\\\n"
        latex_code += "\\hline\n"

    # End the LaTeX table environment
    latex_code += "\\end{tabular}\n"
    latex_code += "\\caption{Your caption here}\n"
    latex_code += "\\label{table:your_label}\n"
    latex_code += "\\end{table}"

    return latex_code


def read_file_in_batches(filename, batch_size=12):
    latex_final_list = [
        [
            "Layer 0",
            "Layer 1",
            "Layer 2",
            "Layer 3",
            "Layer 4",
            "Layer 5",
            "Layer 6",
            "Layer 7",
            "Layer 8",
            "Layer 9",
            "Layer 10",
            "Layer 11",
        ]
    ]
    # Read the entire file and split into lines
    with open(filename, "r") as file:
        lines = [line.strip() for line in file.readlines()]

    # Process lines in batches
    for i in range(0, len(lines), batch_size):
        batch = lines[i : i + batch_size]
        latex_final_list.append(batch)

    return latex_final_list


def transpose_list(input_list):
    return list(map(list, zip(*input_list)))


# Example usage
filename = "latex_table_acc.txt"
latex_final_list = read_file_in_batches(filename)
headers = [
    "Layers",
    "Bloom SAE Continent",
    "Bloom SAE Country",
    "OpenAI SAE Continent",
    "OpenAI SAE Country",
    "Apollo SAE Continent",
    "Apollo SAE Country",
]
t_latex_final_list = transpose_list(latex_final_list)
pprint(latex_final_list)
pprint(t_latex_final_list)
latex_code = create_latex_table(latex_final_list, headers)
pprint(latex_code)
# Output the batches
