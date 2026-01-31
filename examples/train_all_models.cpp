#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include "ts_forecast.hpp"
#include "csv_reader.hpp"

namespace fs = std::filesystem;

const std::string BASE_DIR = "/Users/zimenglyu/Documents/code/git/ts-forecast-cpp/benchmark_datasets";

struct DatasetConfig {
    std::string name;
    std::string subdir;
    std::string train_file;
    std::string val_file;
    std::string test_file;
    int n_features;
    int target_idx;  // Index of OT column
};

// Load univariate data (OT column only)
std::vector<double> load_univariate(const std::string& filepath) {
    auto dataset = ts::CSVReader::read(filepath, false);
    // OT is the last column
    size_t ot_idx = dataset.headers.size() - 1;
    std::vector<double> data;
    for (const auto& row : dataset.data) {
        data.push_back(row[ot_idx]);
    }
    return data;
}

// Load multivariate data (all columns)
std::pair<std::vector<std::vector<double>>, int> load_multivariate(const std::string& filepath) {
    auto dataset = ts::CSVReader::read(filepath, false);
    int target_idx = static_cast<int>(dataset.headers.size()) - 1;  // OT is last
    return {dataset.data, target_idx};
}

void print_header(const std::string& text) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << text << std::endl;
    std::cout << std::string(70, '=') << std::endl;
}

void train_univariate_models(const std::string& dataset_name,
                              const std::vector<double>& train_data,
                              const std::vector<double>& val_data,
                              const std::string& output_dir) {
    std::cout << "\n--- Training Univariate Models for " << dataset_name << " ---" << std::endl;
    std::cout << "Train size: " << train_data.size() << ", Val size: " << val_data.size() << std::endl;

    fs::create_directories(output_dir);

    // ARIMA(2,1,1)
    try {
        std::cout << "  Training ARIMA(2,1,1)..." << std::flush;
        auto start = std::chrono::high_resolution_clock::now();

        ts::ARIMA arima(2, 1, 1);
        arima.fit(train_data);

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        // Validate
        auto forecast = arima.forecast(std::min(24, static_cast<int>(val_data.size())));
        std::vector<double> val_subset(val_data.begin(), val_data.begin() + forecast.predictions.size());
        auto metrics = ts::evaluate(val_subset, forecast.predictions);

        std::cout << " RMSE=" << std::fixed << std::setprecision(4) << metrics.rmse
                  << " (" << ms << "ms)" << std::endl;

        // Save model
        std::string model_path = output_dir + "/" + dataset_name + "_arima.bin";
        arima.save(model_path);
        std::cout << "    Saved: " << model_path << std::endl;
    } catch (const std::exception& e) {
        std::cout << " Failed: " << e.what() << std::endl;
    }

    // Simple Exponential Smoothing
    try {
        std::cout << "  Training SES..." << std::flush;
        auto start = std::chrono::high_resolution_clock::now();

        ts::SimpleExponentialSmoothing ses;
        ses.fit(train_data);

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        auto forecast = ses.forecast(std::min(24, static_cast<int>(val_data.size())));
        std::vector<double> val_subset(val_data.begin(), val_data.begin() + forecast.predictions.size());
        auto metrics = ts::evaluate(val_subset, forecast.predictions);

        std::cout << " RMSE=" << std::fixed << std::setprecision(4) << metrics.rmse
                  << " (" << ms << "ms)" << std::endl;

        // Save model
        std::string model_path = output_dir + "/" + dataset_name + "_ses.bin";
        ses.save(model_path);
        std::cout << "    Saved: " << model_path << std::endl;
    } catch (const std::exception& e) {
        std::cout << " Failed: " << e.what() << std::endl;
    }

    // Holt-Winters (period=24 for hourly data)
    if (train_data.size() >= 48) {
        try {
            std::cout << "  Training HoltWinters(24)..." << std::flush;
            auto start = std::chrono::high_resolution_clock::now();

            ts::HoltWinters hw(24, ts::HoltWinters::SeasonalType::ADDITIVE);
            hw.fit(train_data);

            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();

            auto forecast = hw.forecast(std::min(24, static_cast<int>(val_data.size())));
            std::vector<double> val_subset(val_data.begin(), val_data.begin() + forecast.predictions.size());
            auto metrics = ts::evaluate(val_subset, forecast.predictions);

            std::cout << " RMSE=" << std::fixed << std::setprecision(4) << metrics.rmse
                      << " (" << ms << "ms)" << std::endl;

            // Save model
            std::string model_path = output_dir + "/" + dataset_name + "_holtwinters.bin";
            hw.save(model_path);
            std::cout << "    Saved: " << model_path << std::endl;
        } catch (const std::exception& e) {
            std::cout << " Failed: " << e.what() << std::endl;
        }
    }

    // Prophet
    try {
        std::cout << "  Training Prophet..." << std::flush;
        auto start = std::chrono::high_resolution_clock::now();

        ts::Prophet prophet;
        prophet.add_seasonality("daily", 24.0, 5);
        prophet.fit(train_data);

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        auto forecast = prophet.forecast(std::min(24, static_cast<int>(val_data.size())));
        std::vector<double> val_subset(val_data.begin(), val_data.begin() + forecast.predictions.size());
        auto metrics = ts::evaluate(val_subset, forecast.predictions);

        std::cout << " RMSE=" << std::fixed << std::setprecision(4) << metrics.rmse
                  << " (" << ms << "ms)" << std::endl;

        // Save model
        std::string model_path = output_dir + "/" + dataset_name + "_prophet.bin";
        prophet.save(model_path);
        std::cout << "    Saved: " << model_path << std::endl;
    } catch (const std::exception& e) {
        std::cout << " Failed: " << e.what() << std::endl;
    }

    // DLinear
    try {
        std::cout << "  Training DLinear(96,24)..." << std::flush;
        auto start = std::chrono::high_resolution_clock::now();

        int seq_len = std::min(96, static_cast<int>(train_data.size()) / 4);
        int pred_len = 24;

        ts::DLinear dlinear(seq_len, pred_len, 25);
        dlinear.fit(train_data, 50, 0.001, 32);

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        auto forecast = dlinear.forecast(std::min(pred_len, static_cast<int>(val_data.size())));
        std::vector<double> val_subset(val_data.begin(), val_data.begin() + forecast.predictions.size());
        auto metrics = ts::evaluate(val_subset, forecast.predictions);

        std::cout << " RMSE=" << std::fixed << std::setprecision(4) << metrics.rmse
                  << " (" << ms << "ms)" << std::endl;

        // Save model
        std::string model_path = output_dir + "/" + dataset_name + "_dlinear.bin";
        dlinear.save(model_path);
        std::cout << "    Saved: " << model_path << std::endl;
    } catch (const std::exception& e) {
        std::cout << " Failed: " << e.what() << std::endl;
    }

    // NLinear
    try {
        std::cout << "  Training NLinear(96,24)..." << std::flush;
        auto start = std::chrono::high_resolution_clock::now();

        int seq_len = std::min(96, static_cast<int>(train_data.size()) / 4);
        int pred_len = 24;

        ts::NLinear nlinear(seq_len, pred_len);
        nlinear.fit(train_data, 50, 0.001, 32);

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        auto forecast = nlinear.forecast(std::min(pred_len, static_cast<int>(val_data.size())));
        std::vector<double> val_subset(val_data.begin(), val_data.begin() + forecast.predictions.size());
        auto metrics = ts::evaluate(val_subset, forecast.predictions);

        std::cout << " RMSE=" << std::fixed << std::setprecision(4) << metrics.rmse
                  << " (" << ms << "ms)" << std::endl;

        // Save model
        std::string model_path = output_dir + "/" + dataset_name + "_nlinear.bin";
        nlinear.save(model_path);
        std::cout << "    Saved: " << model_path << std::endl;
    } catch (const std::exception& e) {
        std::cout << " Failed: " << e.what() << std::endl;
    }

    // Linear
    try {
        std::cout << "  Training Linear(96,24)..." << std::flush;
        auto start = std::chrono::high_resolution_clock::now();

        int seq_len = std::min(96, static_cast<int>(train_data.size()) / 4);
        int pred_len = 24;

        ts::Linear linear(seq_len, pred_len);
        linear.fit(train_data, 50, 0.001, 32);

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        auto forecast = linear.forecast(std::min(pred_len, static_cast<int>(val_data.size())));
        std::vector<double> val_subset(val_data.begin(), val_data.begin() + forecast.predictions.size());
        auto metrics = ts::evaluate(val_subset, forecast.predictions);

        std::cout << " RMSE=" << std::fixed << std::setprecision(4) << metrics.rmse
                  << " (" << ms << "ms)" << std::endl;

        // Save model
        std::string model_path = output_dir + "/" + dataset_name + "_linear.bin";
        linear.save(model_path);
        std::cout << "    Saved: " << model_path << std::endl;
    } catch (const std::exception& e) {
        std::cout << " Failed: " << e.what() << std::endl;
    }
}

void train_multivariate_dlinear(const std::string& dataset_name,
                                 const std::vector<std::vector<double>>& train_data,
                                 const std::vector<std::vector<double>>& val_data,
                                 int target_idx,
                                 const std::string& output_dir) {
    std::cout << "\n--- Training Multivariate DLinear for " << dataset_name << " ---" << std::endl;
    std::cout << "Features: " << train_data[0].size() << ", Target idx: " << target_idx << std::endl;

    try {
        std::cout << "  Training DLinear Multivariate..." << std::flush;
        auto start = std::chrono::high_resolution_clock::now();

        int seq_len = std::min(48, static_cast<int>(train_data.size()) / 4);
        int pred_len = 24;

        ts::DLinear dlinear(seq_len, pred_len, 13, false);
        dlinear.fit(train_data, target_idx, 50, 0.001, 32);

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();

        auto forecast = dlinear.forecast(std::min(pred_len, static_cast<int>(val_data.size())));

        // Get actual OT values from validation
        std::vector<double> val_ot;
        for (size_t i = 0; i < forecast.predictions.size() && i < val_data.size(); ++i) {
            val_ot.push_back(val_data[i][target_idx]);
        }

        auto metrics = ts::evaluate(val_ot, forecast.predictions);

        std::cout << " RMSE=" << std::fixed << std::setprecision(4) << metrics.rmse
                  << " (" << ms << "ms)" << std::endl;

        // Save model
        std::string model_path = output_dir + "/" + dataset_name + "_dlinear_mv.bin";
        dlinear.save(model_path);
        std::cout << "    Saved: " << model_path << std::endl;
    } catch (const std::exception& e) {
        std::cout << " Failed: " << e.what() << std::endl;
    }
}

void process_dataset(const std::string& name, const std::string& subdir) {
    print_header("Processing: " + name);

    std::string train_path = BASE_DIR + "/" + subdir + "/" + name + "_train_standard.csv";
    std::string val_path = BASE_DIR + "/" + subdir + "/" + name + "_val_standard.csv";
    std::string output_dir = BASE_DIR + "/" + subdir + "/models";

    // Check if files exist
    if (!fs::exists(train_path)) {
        std::cout << "Train file not found: " << train_path << std::endl;
        return;
    }

    // Load univariate (OT only)
    auto train_uni = load_univariate(train_path);
    auto val_uni = load_univariate(val_path);

    // Train univariate models
    train_univariate_models(name, train_uni, val_uni, output_dir);

    // Load multivariate (all features)
    auto [train_mv, target_idx] = load_multivariate(train_path);
    auto [val_mv, _] = load_multivariate(val_path);

    // Train multivariate DLinear (skip if too many features)
    if (train_mv[0].size() <= 25) {  // Only for reasonably sized datasets
        train_multivariate_dlinear(name, train_mv, val_mv, target_idx, output_dir);
    } else {
        std::cout << "  Skipping multivariate (too many features: " << train_mv[0].size() << ")" << std::endl;
    }

    // Save scaler for this dataset
    ts::StandardScaler scaler;
    scaler.fit(train_mv);
    std::string scaler_path = output_dir + "/" + name + "_scaler.bin";
    scaler.save(scaler_path);
    std::cout << "  Saved scaler: " << scaler_path << std::endl;
}

int main() {
    std::cout << "Training All Models on Benchmark Datasets" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "Models: ARIMA, SES, HoltWinters, Prophet, DLinear, NLinear, Linear" << std::endl;
    std::cout << "Skipping: traffic (too large)" << std::endl;

    // ETT datasets
    process_dataset("ETTh1", "ETT-small");
    process_dataset("ETTh2", "ETT-small");
    process_dataset("ETTm1", "ETT-small");
    process_dataset("ETTm2", "ETT-small");

    // Other datasets
    process_dataset("exchange_rate", "exchange_rate");
    process_dataset("illness", "illness");
    process_dataset("weather", "weather");
    process_dataset("electricity", "electricity");

    print_header("Training Complete");

    // List saved models
    std::cout << "\nSaved models:" << std::endl;
    for (const auto& entry : fs::recursive_directory_iterator(BASE_DIR)) {
        if (entry.path().extension() == ".bin") {
            std::cout << "  " << entry.path().string() << std::endl;
        }
    }

    return 0;
}
