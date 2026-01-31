#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include "ts_forecast.hpp"
#include "csv_reader.hpp"

namespace fs = std::filesystem;

// Save data to CSV file
void save_csv(const std::string& filename,
              const std::vector<std::string>& headers,
              const std::vector<std::vector<double>>& data) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // Write headers
    for (size_t i = 0; i < headers.size(); ++i) {
        file << headers[i];
        if (i < headers.size() - 1) file << ",";
    }
    file << "\n";

    // Write data
    file << std::fixed << std::setprecision(6);
    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) file << ",";
        }
        file << "\n";
    }

    file.close();
}

void process_dataset(const std::string& input_path, const std::string& output_dir,
                     const std::string& dataset_name, bool skip_first_col = true) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Processing: " << dataset_name << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // Load dataset
    ts::CSVReader::Dataset dataset;
    try {
        dataset = ts::CSVReader::read(input_path, skip_first_col);
    } catch (const std::exception& e) {
        std::cerr << "Error loading " << dataset_name << ": " << e.what() << std::endl;
        return;
    }

    std::cout << "Loaded: " << dataset.num_rows() << " rows, "
              << dataset.num_cols() << " columns" << std::endl;
    std::cout << "Columns: ";
    for (size_t i = 0; i < std::min(dataset.headers.size(), size_t(5)); ++i) {
        std::cout << dataset.headers[i] << " ";
    }
    if (dataset.headers.size() > 5) std::cout << "... (" << dataset.headers.size() << " total)";
    std::cout << std::endl;

    if (dataset.num_rows() < 10) {
        std::cerr << "Dataset too small, skipping." << std::endl;
        return;
    }

    // Data is already in multivariate format (rows = time, cols = features)
    const std::vector<std::vector<double>>& all_data = dataset.data;

    // Split into train/val/test (70/15/15)
    auto split = ts::train_val_test_split(all_data, 0.7, 0.15);

    std::cout << "Split: train=" << split.train.size()
              << ", val=" << split.val.size()
              << ", test=" << split.test.size() << std::endl;

    // Create output directory if needed
    fs::create_directories(output_dir);

    // =====================================================
    // Save raw splits
    // =====================================================
    save_csv(output_dir + "/" + dataset_name + "_train_raw.csv", dataset.headers, split.train);
    save_csv(output_dir + "/" + dataset_name + "_val_raw.csv", dataset.headers, split.val);
    save_csv(output_dir + "/" + dataset_name + "_test_raw.csv", dataset.headers, split.test);
    std::cout << "Saved raw splits" << std::endl;

    // =====================================================
    // MinMax Normalization
    // =====================================================
    ts::MinMaxScaler minmax_scaler;
    minmax_scaler.fit(split.train);  // Fit ONLY on training data

    auto train_minmax = minmax_scaler.transform(split.train);
    auto val_minmax = minmax_scaler.transform(split.val);
    auto test_minmax = minmax_scaler.transform(split.test);

    save_csv(output_dir + "/" + dataset_name + "_train_minmax.csv", dataset.headers, train_minmax);
    save_csv(output_dir + "/" + dataset_name + "_val_minmax.csv", dataset.headers, val_minmax);
    save_csv(output_dir + "/" + dataset_name + "_test_minmax.csv", dataset.headers, test_minmax);
    std::cout << "Saved MinMax normalized" << std::endl;

    // =====================================================
    // Standard Normalization (Z-score)
    // =====================================================
    ts::StandardScaler standard_scaler;
    standard_scaler.fit(split.train);  // Fit ONLY on training data

    auto train_standard = standard_scaler.transform(split.train);
    auto val_standard = standard_scaler.transform(split.val);
    auto test_standard = standard_scaler.transform(split.test);

    save_csv(output_dir + "/" + dataset_name + "_train_standard.csv", dataset.headers, train_standard);
    save_csv(output_dir + "/" + dataset_name + "_val_standard.csv", dataset.headers, val_standard);
    save_csv(output_dir + "/" + dataset_name + "_test_standard.csv", dataset.headers, test_standard);
    std::cout << "Saved Standard normalized" << std::endl;

    // Print summary for last column (usually target)
    size_t last_col = dataset.headers.size() - 1;
    std::cout << "Last column (" << dataset.headers[last_col] << ") stats:" << std::endl;
    std::cout << "  MinMax range: [" << minmax_scaler.min_vals()[last_col]
              << ", " << minmax_scaler.max_vals()[last_col] << "]" << std::endl;
    std::cout << "  Standard: mean=" << standard_scaler.means()[last_col]
              << ", std=" << standard_scaler.stds()[last_col] << std::endl;
}

int main() {
    const std::string BASE_DIR = "/Users/zimenglyu/Downloads/all_six_datasets";

    std::cout << "Processing All Datasets" << std::endl;
    std::cout << "========================" << std::endl;
    std::cout << "Base directory: " << BASE_DIR << std::endl;
    std::cout << "Split ratio: 70% train, 15% val, 15% test" << std::endl;
    std::cout << "Normalization: MinMax [0,1] and Standard (z-score)" << std::endl;
    std::cout << "Scalers fit on training set only!" << std::endl;

    // Define all datasets
    struct DatasetInfo {
        std::string subdir;
        std::string filename;
        std::string name;
        bool skip_first_col;
    };

    std::vector<DatasetInfo> datasets = {
        {"ETT-small", "ETTh1.csv", "ETTh1", true},
        {"ETT-small", "ETTh2.csv", "ETTh2", true},
        {"ETT-small", "ETTm1.csv", "ETTm1", true},
        {"ETT-small", "ETTm2.csv", "ETTm2", true},
        {"electricity", "electricity.csv", "electricity", true},
        {"exchange_rate", "exchange_rate.csv", "exchange_rate", true},
        {"traffic", "traffic.csv", "traffic", true},
        {"weather", "weather.csv", "weather", true},
        {"illness", "national_illness.csv", "illness", true},
    };

    int processed = 0;
    int failed = 0;

    for (const auto& ds : datasets) {
        std::string input_path = BASE_DIR + "/" + ds.subdir + "/" + ds.filename;
        std::string output_dir = BASE_DIR + "/" + ds.subdir;

        try {
            process_dataset(input_path, output_dir, ds.name, ds.skip_first_col);
            processed++;
        } catch (const std::exception& e) {
            std::cerr << "Failed to process " << ds.name << ": " << e.what() << std::endl;
            failed++;
        }
    }

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Summary" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Processed: " << processed << " datasets" << std::endl;
    std::cout << "Failed: " << failed << " datasets" << std::endl;
    std::cout << "\nOutput files for each dataset:" << std::endl;
    std::cout << "  {name}_train_raw.csv, {name}_val_raw.csv, {name}_test_raw.csv" << std::endl;
    std::cout << "  {name}_train_minmax.csv, {name}_val_minmax.csv, {name}_test_minmax.csv" << std::endl;
    std::cout << "  {name}_train_standard.csv, {name}_val_standard.csv, {name}_test_standard.csv" << std::endl;

    return failed > 0 ? 1 : 0;
}
