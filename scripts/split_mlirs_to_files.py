# The script split a large file with dumped MLIRs after '--print-mlir-after-all'
# into multiple smaller files based on pass markers
# CMD: split_mlirs_to_files.py file.txt
# Output: Directory "mlir_after_passes" (be default) with several seperate MLIR files after
#         each applied pass + index.txt file with order of applied passes.



import os
import re
import argparse

def split_file(input_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current_block = []
    current_name = None
    counters = {}
    file_index = 0
    generated_files = []

    for line in lines:
        if line.startswith("// -----// IR Dump After"):
            # Save previous block if exists
            if current_block and current_name:
                file_index += 1
                counters[current_name] = counters.get(current_name, 0) + 1
                filename = f"{file_index:03d}_{current_name}_{counters[current_name]}.mlir"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, "w", encoding="utf-8") as out:
                    out.writelines(current_block)
                generated_files.append(filename)

            # Start new block
            match = re.search(r"IR Dump After (\S+)", line)
            if match:
                current_name = match.group(1)
            else:
                current_name = "UnknownPass"

            current_block = [line]
        else:
            current_block.append(line)

    # Save the last block
    if current_block and current_name:
        file_index += 1
        counters[current_name] = counters.get(current_name, 0) + 1
        filename = f"{file_index:03d}_{current_name}_{counters[current_name]}.mlir"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as out:
            out.writelines(current_block)
        generated_files.append(filename)

    # Write index file
    index_path = os.path.join(output_dir, "index.txt")
    with open(index_path, "w", encoding="utf-8") as idx:
        for f in generated_files:
            idx.write(f + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a large MLIR dump file into multiple smaller files "
                    "based on pass markers."
    )
    parser.add_argument("input_file", help="Path to the input file with many MLIRs")
    parser.add_argument(
        "-o", "--output_dir", default="mlir_after_passes",
        help="Directory to save split MLIR files (default: mlir_after_passes)"
    )

    args = parser.parse_args()
    split_file(args.input_file, args.output_dir)
    print(f"MLIR file split into parts in '{args.output_dir}'")
