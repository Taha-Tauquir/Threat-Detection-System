import csv
import os

if __name__ == "__main__":
    input_files = [
        "0166_input",
        "0286_input",
        "0790_input",
        "0930_input",
        "1174_input",
        "1322_input",
        "1324_input",
        "1968_input",
        "1974_input",
        "2216_input",

        "2228_input",
        "2230_input",
        "2232_input",
        "2242_input",
        "2378_input",
        "2488_input",
        "2504_input",
        "2630_input",
        "2634_input",
        "2732_input",

        "2736_input",
        "2744_input",
        "2746_input",
        "2754_input",
        "2762_input",
        "2768_input",
        "2876_input",
        "2884_input",
        "2896_input",
        "3146_input",

        "3148_input",
        "3394_input",
        "3404_input",
        "3418_input",
        "3514_input",
        "3644_input",
        "3648_input",
        "3652_input",
        "3654_input",
        "3802_input",
    ]

    base_path = "/Users/mzeeshan/Documents/PythonProjects/NewRppgHundred/rPPG-Toolbox/data/MAHNOB/cached/MAHNOB_SizeW128_SizeH128_ClipLength900_DataTypeDiffNormalized_DataAugNone_LabelTypeDiffNormalized_Crop_faceFalse_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse"

    # Output CSV file name
    csv_filename = os.path.join("/Users/mzeeshan/Documents/PythonProjects/NewRppgHundred/rPPG-Toolbox/data/MAHNOB/filelists", "train_list.csv")

    # Write to CSV
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["input_files"])  # write header
        for item in input_files:
            full_path = os.path.join(base_path, f"{item}.npy")
            print(full_path)
            writer.writerow([full_path]) 

    print(f"{csv_filename} created with {len(input_files)} entries.")
