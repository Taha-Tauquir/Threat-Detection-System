from Generate30SVideo import process_all_videos
from GenerateFrames import iterate_through_videos
from DetectFaceAndCrop import perform_face_detection
from CheckMissingFramesPct import analyze_sessions
from GenerateInputsNPY import process_inputs_npys
from MissingFramesFixation import FixMissingFrames
from DatasetSplitter import split_npy_files
from MAHNOBLabelProcessor import MAHNOBLabelProcessor,create_train_test_csv_lists_for_physnet


def main():
    # === Configuration ===
    #BASE_PATH = "C:/Users/Hp/Downloads/MAHNOB-SAMPLE_DATASET"
    BASE_PATH = "C:/Users/Hp/Downloads/MAHNOB-SAMPLE_DATASET/tst"
    ORIGIN_FOLDER = BASE_PATH
    CACHED_PATH = os.path.join(BASE_PATH, "cached",
        "MAHNOB_SizeW128_SizeH128_ClipLength900_DataTypeDiffNormalized_DataAugNone_LabelTypeDiffNormalized_Crop_faceFalse_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse"
    )

    # === Step 1: Trim videos to 30 seconds ===
    print("\nStep 1: Trimming videos to 30 seconds...")
    process_all_videos(BASE_PATH)

    # === Step 2: Generate 900 frames at 30 FPS ===
    print("\nStep 2: Generating 900 frames from trimmed videos...")
    # iterate_through_videos(BASE_PATH, ORIGIN_FOLDER)

    # === Step 3: Perform face detection and cropping ===
    print("\nStep 3: Performing face detection and cropping...")
    # perform_face_detection(BASE_PATH)

    # === Step 4: Fix inconsistent or missing frames ===
    print("\nStep 4: Fixing missing frames...")
    # FixMissingFrames(BASE_PATH)

    # === Step 5: Analyze session completeness ===
    print("\nStep 5: Analyzing sessions for frame consistency...")
    # analyze_sessions(BASE_PATH)

    # === Step 6: Generate input .npy files ===
    print("\nStep 6: Generating input NPY files...")
    #process_inputs_npys(BASE_PATH, CACHED_PATH)

    # === Step 7: Generate labels from 30Hz CSV files ===
    print("\nStep 7: Generating label NPY files...")
    processor = MAHNOBLabelProcessor(BASE_PATH, CACHED_PATH)
    #processor.iterate_csv_and_generate_labels()

    # === Step 8: Split labels into train and test sets ===
    print("\nStep 8: Splitting label NPY files into train/test...")
    #processor.split_labels_and_inputs(CACHED_PATH)


    # === Step 10: Create train_list.csv and test_list.csv ===
    #print("\nStep 10: Creating train/test CSV filelists...")

    #create_train_test_csv_lists_for_physnet(CACHED_PATH)



    print("\nPipeline completed. Uncomment specific steps above to run them manually.")

if __name__ == "__main__":
    import os
    main()
