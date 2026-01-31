#include <iostream>
#include <fstream>
#include <iomanip>
#include "ts_forecast.hpp"
#include "csv_reader.hpp"

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
    std::cout << "Saved: " << filename << " (" << data.size() << " rows)" << std::endl;
}

int main() {
    const std::string DATA_PATH = "/Users/zimenglyu/Downloads/ETTh1.csv";
    const std::string OUTPUT_DIR = "/Users/zimenglyu/Downloads/";

    std::cout << "ETTh1 Dataset Normalization" << std::endl;
    std::cout << "============================" << std::endl;

    // Load dataset
    std::cout << "\nLoading ETTh1 dataset..." << std::endl;
    ts::CSVReader::Dataset dataset;
    try {
        dataset = ts::CSVReader::read(DATA_PATH, true);  // Skip date column
    } catch (const std::exception& e) {
        std::cerr << "Error loading dataset: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Dataset loaded: " << dataset.num_rows() << " rows, "
              << dataset.num_cols() << " columns" << std::endl;
    std::cout << "Columns: ";
    for (const auto& h : dataset.headers) std::cout << h << " ";
    std::cout << std::endl;

    // Data is already in multivariate format (rows = time, cols = features)
    const std::vector<std::vector<double>>& all_data = dataset.data;

    // Split into train/val/test (70/15/15)
    std::cout << "\nSplitting data (70% train, 15% val, 15% test)..." << std::endl;
    auto split = ts::train_val_test_split(all_data, 0.7, 0.15);

    std::cout << "Train size: " << split.train.size() << std::endl;
    std::cout << "Val size:   " << split.val.size() << std::endl;
    std::cout << "Test size:  " << split.test.size() << std::endl;

    // =====================================================
    // MinMax Normalization
    // =====================================================
    std::cout << "\n--- MinMax Normalization ---" << std::endl;
    std::cout << "Fitting on training set only..." << std::endl;

    ts::MinMaxScaler minmax_scaler;
    minmax_scaler.fit(split.train);  // Fit ONLY on training data

    // Transform all sets using parameters learned from training
    auto train_minmax = minmax_scaler.transform(split.train);
    auto val_minmax = minmax_scaler.transform(split.val);
    auto test_minmax = minmax_scaler.transform(split.test);

    // Print scaler parameters
    std::cout << "Scaler parameters (from training set):" << std::endl;
    for (size_t j = 0; j < dataset.headers.size(); ++j) {
        std::cout << "  " << dataset.headers[j]
                  << ": min=" << minmax_scaler.min_vals()[j]
                  << ", max=" << minmax_scaler.max_vals()[j] << std::endl;
    }

    // Save MinMax normalized data
    save_csv(OUTPUT_DIR + "ETTh1_train_minmax.csv", dataset.headers, train_minmax);
    save_csv(OUTPUT_DIR + "ETTh1_val_minmax.csv", dataset.headers, val_minmax);
    save_csv(OUTPUT_DIR + "ETTh1_test_minmax.csv", dataset.headers, test_minmax);

    // =====================================================
    // Standard Normalization (Z-score)
    // =====================================================
    std::cout << "\n--- Standard Normalization (Z-score) ---" << std::endl;
    std::cout << "Fitting on training set only..." << std::endl;

    ts::StandardScaler standard_scaler;
    standard_scaler.fit(split.train);  // Fit ONLY on training data

    // Transform all sets using parameters learned from training
    auto train_standard = standard_scaler.transform(split.train);
    auto val_standard = standard_scaler.transform(split.val);
    auto test_standard = standard_scaler.transform(split.test);

    // Print scaler parameters
    std::cout << "Scaler parameters (from training set):" << std::endl;
    for (size_t j = 0; j < dataset.headers.size(); ++j) {
        std::cout << "  " << dataset.headers[j]
                  << ": mean=" << standard_scaler.means()[j]
                  << ", std=" << standard_scaler.stds()[j] << std::endl;
    }

    // Save Standard normalized data
    save_csv(OUTPUT_DIR + "ETTh1_train_standard.csv", dataset.headers, train_standard);
    save_csv(OUTPUT_DIR + "ETTh1_val_standard.csv", dataset.headers, val_standard);
    save_csv(OUTPUT_DIR + "ETTh1_test_standard.csv", dataset.headers, test_standard);

    // =====================================================
    // Also save raw (unnormalized) splits for reference
    // =====================================================
    std::cout << "\n--- Saving raw splits ---" << std::endl;
    save_csv(OUTPUT_DIR + "ETTh1_train_raw.csv", dataset.headers, split.train);
    save_csv(OUTPUT_DIR + "ETTh1_val_raw.csv", dataset.headers, split.val);
    save_csv(OUTPUT_DIR + "ETTh1_test_raw.csv", dataset.headers, split.test);

    // =====================================================
    // Verification: check statistics
    // =====================================================
    std::cout << "\n--- Verification ---" << std::endl;

    // Check MinMax: train should be in [0,1], val/test may exceed slightly
    std::cout << "MinMax - checking OT column ranges:" << std::endl;
    size_t ot_idx = dataset.headers.size() - 1;  // OT is last column

    double train_min = train_minmax[0][ot_idx], train_max = train_minmax[0][ot_idx];
    for (const auto& row : train_minmax) {
        if (row[ot_idx] < train_min) train_min = row[ot_idx];
        if (row[ot_idx] > train_max) train_max = row[ot_idx];
    }
    std::cout << "  Train OT range: [" << train_min << ", " << train_max << "]" << std::endl;

    double val_min = val_minmax[0][ot_idx], val_max = val_minmax[0][ot_idx];
    for (const auto& row : val_minmax) {
        if (row[ot_idx] < val_min) val_min = row[ot_idx];
        if (row[ot_idx] > val_max) val_max = row[ot_idx];
    }
    std::cout << "  Val OT range:   [" << val_min << ", " << val_max << "]" << std::endl;

    double test_min = test_minmax[0][ot_idx], test_max = test_minmax[0][ot_idx];
    for (const auto& row : test_minmax) {
        if (row[ot_idx] < test_min) test_min = row[ot_idx];
        if (row[ot_idx] > test_max) test_max = row[ot_idx];
    }
    std::cout << "  Test OT range:  [" << test_min << ", " << test_max << "]" << std::endl;

    // Check Standard: train should have mean~0, std~1
    std::cout << "\nStandard - checking OT column statistics:" << std::endl;

    double train_mean = 0, train_var = 0;
    for (const auto& row : train_standard) train_mean += row[ot_idx];
    train_mean /= train_standard.size();
    for (const auto& row : train_standard) {
        train_var += (row[ot_idx] - train_mean) * (row[ot_idx] - train_mean);
    }
    train_var /= (train_standard.size() - 1);
    std::cout << "  Train OT: mean=" << train_mean << ", std=" << std::sqrt(train_var) << std::endl;

    double val_mean = 0;
    for (const auto& row : val_standard) val_mean += row[ot_idx];
    val_mean /= val_standard.size();
    std::cout << "  Val OT:   mean=" << val_mean << std::endl;

    double test_mean = 0;
    for (const auto& row : test_standard) test_mean += row[ot_idx];
    test_mean /= test_standard.size();
    std::cout << "  Test OT:  mean=" << test_mean << std::endl;

    std::cout << "\nDone! Files saved to: " << OUTPUT_DIR << std::endl;
    std::cout << "\nGenerated files:" << std::endl;
    std::cout << "  - ETTh1_train_raw.csv, ETTh1_val_raw.csv, ETTh1_test_raw.csv" << std::endl;
    std::cout << "  - ETTh1_train_minmax.csv, ETTh1_val_minmax.csv, ETTh1_test_minmax.csv" << std::endl;
    std::cout << "  - ETTh1_train_standard.csv, ETTh1_val_standard.csv, ETTh1_test_standard.csv" << std::endl;

    return 0;
}
