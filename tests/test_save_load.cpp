#include <iostream>
#include <cmath>
#include <cassert>
#include "ts_forecast.hpp"
#include "csv_reader.hpp"

int tests_passed = 0;
int tests_failed = 0;

void test_assert(bool condition, const std::string& test_name) {
    if (condition) {
        std::cout << "[PASS] " << test_name << std::endl;
        tests_passed++;
    } else {
        std::cout << "[FAIL] " << test_name << std::endl;
        tests_failed++;
    }
}

void test_minmax_scaler_save_load() {
    std::cout << "\n=== Testing MinMaxScaler Save/Load ===" << std::endl;

    // Create and fit scaler
    std::vector<std::vector<double>> data = {
        {10, 100, 1000},
        {20, 200, 2000},
        {30, 300, 3000},
        {40, 400, 4000},
        {50, 500, 5000}
    };

    ts::MinMaxScaler scaler;
    scaler.fit(data);

    // Save original parameters
    auto orig_min = scaler.min_vals();
    auto orig_max = scaler.max_vals();

    // Transform some data
    std::vector<std::vector<double>> test_data = {{25, 250, 2500}};
    auto orig_transformed = scaler.transform(test_data);

    // Save scaler
    scaler.save("/tmp/minmax_test.bin");
    test_assert(true, "MinMaxScaler save");

    // Load into new scaler
    ts::MinMaxScaler loaded_scaler;
    loaded_scaler.load("/tmp/minmax_test.bin");
    test_assert(loaded_scaler.is_fitted(), "MinMaxScaler load - is_fitted");

    // Compare parameters
    auto loaded_min = loaded_scaler.min_vals();
    auto loaded_max = loaded_scaler.max_vals();

    test_assert(loaded_min.size() == orig_min.size(), "MinMaxScaler load - size match");

    bool params_match = true;
    for (size_t i = 0; i < orig_min.size(); ++i) {
        if (std::abs(loaded_min[i] - orig_min[i]) > 1e-10 ||
            std::abs(loaded_max[i] - orig_max[i]) > 1e-10) {
            params_match = false;
            break;
        }
    }
    test_assert(params_match, "MinMaxScaler load - parameters match");

    // Transform same data with loaded scaler
    auto loaded_transformed = loaded_scaler.transform(test_data);

    bool transform_match = true;
    for (size_t i = 0; i < orig_transformed[0].size(); ++i) {
        if (std::abs(loaded_transformed[0][i] - orig_transformed[0][i]) > 1e-10) {
            transform_match = false;
            break;
        }
    }
    test_assert(transform_match, "MinMaxScaler load - transform matches");

    // Test inverse transform
    auto orig_inverse = scaler.inverse_transform(orig_transformed);
    auto loaded_inverse = loaded_scaler.inverse_transform(loaded_transformed);

    bool inverse_match = true;
    for (size_t i = 0; i < orig_inverse[0].size(); ++i) {
        if (std::abs(loaded_inverse[0][i] - orig_inverse[0][i]) > 1e-10) {
            inverse_match = false;
            break;
        }
    }
    test_assert(inverse_match, "MinMaxScaler load - inverse_transform matches");
}

void test_standard_scaler_save_load() {
    std::cout << "\n=== Testing StandardScaler Save/Load ===" << std::endl;

    std::vector<std::vector<double>> data = {
        {10, 100},
        {20, 200},
        {30, 300},
        {40, 400},
        {50, 500}
    };

    ts::StandardScaler scaler;
    scaler.fit(data);

    auto orig_means = scaler.means();
    auto orig_stds = scaler.stds();

    std::vector<std::vector<double>> test_data = {{25, 250}};
    auto orig_transformed = scaler.transform(test_data);

    // Save and load
    scaler.save("/tmp/standard_test.bin");
    test_assert(true, "StandardScaler save");

    ts::StandardScaler loaded_scaler;
    loaded_scaler.load("/tmp/standard_test.bin");
    test_assert(loaded_scaler.is_fitted(), "StandardScaler load - is_fitted");

    auto loaded_means = loaded_scaler.means();
    auto loaded_stds = loaded_scaler.stds();

    bool params_match = true;
    for (size_t i = 0; i < orig_means.size(); ++i) {
        if (std::abs(loaded_means[i] - orig_means[i]) > 1e-10 ||
            std::abs(loaded_stds[i] - orig_stds[i]) > 1e-10) {
            params_match = false;
            break;
        }
    }
    test_assert(params_match, "StandardScaler load - parameters match");

    auto loaded_transformed = loaded_scaler.transform(test_data);
    bool transform_match = true;
    for (size_t i = 0; i < orig_transformed[0].size(); ++i) {
        if (std::abs(loaded_transformed[0][i] - orig_transformed[0][i]) > 1e-10) {
            transform_match = false;
            break;
        }
    }
    test_assert(transform_match, "StandardScaler load - transform matches");
}

void test_dlinear_save_load() {
    std::cout << "\n=== Testing DLinear Save/Load ===" << std::endl;

    // Generate simple data
    std::vector<double> data(200);
    for (int i = 0; i < 200; ++i) {
        data[i] = 10 + 0.1 * i + 5 * std::sin(2 * M_PI * i / 24);
    }

    // Train model
    ts::DLinear model(48, 12, 13);
    model.fit(data, 20, 0.001, 16);

    test_assert(model.is_fitted(), "DLinear fit");

    // Get predictions before save
    std::vector<double> input(data.end() - 48, data.end());
    auto orig_pred = model.predict(input);

    // Save model
    model.save("/tmp/dlinear_test.bin");
    test_assert(true, "DLinear save");

    // Load into new model
    ts::DLinear loaded_model;
    loaded_model.load("/tmp/dlinear_test.bin");
    test_assert(loaded_model.is_fitted(), "DLinear load - is_fitted");
    test_assert(loaded_model.seq_len() == 48, "DLinear load - seq_len");
    test_assert(loaded_model.pred_len() == 12, "DLinear load - pred_len");

    // Get predictions from loaded model
    auto loaded_pred = loaded_model.predict(input);

    test_assert(loaded_pred.size() == orig_pred.size(), "DLinear load - prediction size");

    bool pred_match = true;
    double max_diff = 0;
    for (size_t i = 0; i < orig_pred.size(); ++i) {
        double diff = std::abs(loaded_pred[i] - orig_pred[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-10) {
            pred_match = false;
        }
    }
    test_assert(pred_match, "DLinear load - predictions match exactly");

    std::cout << "  Max prediction difference: " << max_diff << std::endl;

    // Test forecast
    auto orig_forecast = model.forecast(12);
    auto loaded_forecast = loaded_model.forecast(12);

    bool forecast_match = true;
    for (size_t i = 0; i < orig_forecast.predictions.size(); ++i) {
        if (std::abs(loaded_forecast.predictions[i] - orig_forecast.predictions[i]) > 1e-10) {
            forecast_match = false;
            break;
        }
    }
    test_assert(forecast_match, "DLinear load - forecast matches");
}

void test_nlinear_save_load() {
    std::cout << "\n=== Testing NLinear Save/Load ===" << std::endl;

    std::vector<double> data(150);
    for (int i = 0; i < 150; ++i) {
        data[i] = 50 + 0.2 * i + (rand() % 10 - 5);
    }

    ts::NLinear model(48, 12);
    model.fit(data, 20, 0.001, 16);

    test_assert(model.is_fitted(), "NLinear fit");

    std::vector<double> input(data.end() - 48, data.end());
    auto orig_pred = model.predict(input);

    model.save("/tmp/nlinear_test.bin");
    test_assert(true, "NLinear save");

    ts::NLinear loaded_model;
    loaded_model.load("/tmp/nlinear_test.bin");
    test_assert(loaded_model.is_fitted(), "NLinear load - is_fitted");

    auto loaded_pred = loaded_model.predict(input);

    bool pred_match = true;
    for (size_t i = 0; i < orig_pred.size(); ++i) {
        if (std::abs(loaded_pred[i] - orig_pred[i]) > 1e-10) {
            pred_match = false;
            break;
        }
    }
    test_assert(pred_match, "NLinear load - predictions match");
}

void test_linear_save_load() {
    std::cout << "\n=== Testing Linear Save/Load ===" << std::endl;

    std::vector<double> data(150);
    for (int i = 0; i < 150; ++i) {
        data[i] = 20 + 0.15 * i + (rand() % 6 - 3);
    }

    ts::Linear model(48, 12);
    model.fit(data, 20, 0.001, 16);

    test_assert(model.is_fitted(), "Linear fit");

    std::vector<double> input(data.end() - 48, data.end());
    auto orig_pred = model.predict(input);

    model.save("/tmp/linear_test.bin");
    test_assert(true, "Linear save");

    ts::Linear loaded_model;
    loaded_model.load("/tmp/linear_test.bin");
    test_assert(loaded_model.is_fitted(), "Linear load - is_fitted");

    auto loaded_pred = loaded_model.predict(input);

    bool pred_match = true;
    for (size_t i = 0; i < orig_pred.size(); ++i) {
        if (std::abs(loaded_pred[i] - orig_pred[i]) > 1e-10) {
            pred_match = false;
            break;
        }
    }
    test_assert(pred_match, "Linear load - predictions match");
}

void test_real_data_save_load() {
    std::cout << "\n=== Testing Save/Load with ETTh1 Data ===" << std::endl;

    // Load real data
    ts::CSVReader::Dataset dataset;
    try {
        dataset = ts::CSVReader::read("/Users/zimenglyu/Downloads/all_six_datasets/ETT-small/ETTh1.csv", true);
    } catch (...) {
        std::cout << "[SKIP] ETTh1 dataset not available" << std::endl;
        return;
    }

    auto OT = dataset.get_column("OT");
    std::vector<double> train(OT.begin(), OT.begin() + 1000);

    // Train and save scaler
    ts::StandardScaler scaler;
    scaler.fit({train});
    scaler.save("/tmp/etth1_scaler.bin");

    // Normalize data
    auto train_normalized = scaler.transform(train);

    // Train and save model
    ts::DLinear model(96, 24, 25);
    model.fit(train_normalized, 30, 0.001, 16);
    model.save("/tmp/etth1_dlinear.bin");

    // Get original predictions
    std::vector<double> input(train_normalized.end() - 96, train_normalized.end());
    auto orig_pred_norm = model.predict(input);
    auto orig_pred = scaler.inverse_transform(orig_pred_norm);

    // Load scaler and model
    ts::StandardScaler loaded_scaler;
    loaded_scaler.load("/tmp/etth1_scaler.bin");

    ts::DLinear loaded_model;
    loaded_model.load("/tmp/etth1_dlinear.bin");

    // Get predictions from loaded
    auto loaded_pred_norm = loaded_model.predict(input);
    auto loaded_pred = loaded_scaler.inverse_transform(loaded_pred_norm);

    test_assert(loaded_pred.size() == orig_pred.size(), "Real data - prediction size");

    bool pred_match = true;
    double max_diff = 0;
    for (size_t i = 0; i < orig_pred.size(); ++i) {
        double diff = std::abs(loaded_pred[i] - orig_pred[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-6) {  // Allow small numerical differences
            pred_match = false;
        }
    }
    test_assert(pred_match, "Real data - predictions match");
    std::cout << "  Max difference: " << max_diff << std::endl;

    std::cout << "\n  First 5 original predictions:  ";
    for (int i = 0; i < 5; ++i) std::cout << orig_pred[i] << " ";
    std::cout << "\n  First 5 loaded predictions:    ";
    for (int i = 0; i < 5; ++i) std::cout << loaded_pred[i] << " ";
    std::cout << std::endl;
}

int main() {
    std::cout << "Model Save/Load Test Suite" << std::endl;
    std::cout << "==========================" << std::endl;

    try {
        test_minmax_scaler_save_load();
        test_standard_scaler_save_load();
        test_dlinear_save_load();
        test_nlinear_save_load();
        test_linear_save_load();
        test_real_data_save_load();
    } catch (const std::exception& e) {
        std::cout << "\n[ERROR] Exception: " << e.what() << std::endl;
        tests_failed++;
    }

    std::cout << "\n==========================" << std::endl;
    std::cout << "Tests passed: " << tests_passed << std::endl;
    std::cout << "Tests failed: " << tests_failed << std::endl;
    std::cout << "==========================" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
