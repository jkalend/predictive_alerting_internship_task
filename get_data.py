import kagglehub

# Download latest version
path1 = kagglehub.dataset_download("vishala28/swat-dataset-secure-water-treatment-system", output_dir="data_swat")

print("Path to dataset files:", path1)

path2 = kagglehub.dataset_download("palbha/cmapss-jet-engine-simulated-data", output_dir="data_cmapss")

print("Path to dataset files:", path2)
