import os

input_dir = "labels_old"
output_dir = "txt_labels"
W, H = 256, 256

os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.endswith(".csv"):
        out_lines = []
        in_path = os.path.join(input_dir, file)
        with open(in_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 4:
                    x1, y1, x2, y2 = map(float, parts)
                    x_c = ((x1 + x2) / 2) / W
                    y_c = ((y1 + y2) / 2) / H
                    w = (x2 - x1) / W
                    h = (y2 - y1) / H
                    if(x_c <= 1 and y_c <= 1 and h <= 1 and w <= 1): 
                        out_lines.append(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                    else:
                        print(f'File is too big: {file}')

        out_path = os.path.join(output_dir, os.path.splitext(file)[0] + ".txt")
        with open(out_path, "w") as f:
            f.writelines(out_lines)
