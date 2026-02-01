#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <sstream>
#include "ts_forecast.hpp"
#include "csv_reader.hpp"

namespace fs = std::filesystem;

const std::string BASE_DIR = "../benchmark_datasets";

struct EvalResult {
    std::string dataset;
    std::string model_name;
    double mse;
    double mae;
    size_t param_count;
    int test_data_points;          // Total rows in test data
    double total_inference_s;      // Total inference time in seconds
    double latency_per_point_s;    // Seconds per data point
    double throughput;             // Data points per second
    bool is_multivariate;
};

// Load univariate data (last column - OT)
std::vector<double> load_univariate(const std::string& filepath) {
    auto dataset = ts::CSVReader::read(filepath, false);
    size_t ot_idx = dataset.headers.size() - 1;
    std::vector<double> data;
    for (const auto& row : dataset.data) {
        data.push_back(row[ot_idx]);
    }
    return data;
}

void print_header() {
    std::cout << std::left
              << std::setw(12) << "Dataset"
              << std::setw(14) << "Model"
              << std::right
              << std::setw(12) << "MSE"
              << std::setw(11) << "MAE"
              << std::setw(8) << "Params"
              << std::setw(10) << "TestRows"
              << std::setw(12) << "Total(s)"
              << std::setw(14) << "Latency(s)"
              << std::setw(14) << "Throughput"
              << std::endl;
    std::cout << std::string(107, '-') << std::endl;
}

void print_result(const EvalResult& r) {
    std::cout << std::left
              << std::setw(12) << r.dataset
              << std::setw(14) << r.model_name
              << std::right << std::fixed
              << std::setprecision(6) << std::setw(12) << r.mse
              << std::setprecision(6) << std::setw(11) << r.mae
              << std::setw(8) << r.param_count
              << std::setw(10) << r.test_data_points
              << std::setprecision(4) << std::setw(12) << r.total_inference_s
              << std::scientific << std::setprecision(2) << std::setw(14) << r.latency_per_point_s
              << std::fixed << std::setprecision(2) << std::setw(14) << r.throughput
              << std::endl;
}

// Evaluate ARIMA - single step forecast
EvalResult evaluate_arima(const std::string& model_path,
                          const std::vector<double>& test_data,
                          const std::string& dataset_name) {
    EvalResult result;
    result.dataset = dataset_name;
    result.model_name = "ARIMA";
    result.is_multivariate = false;
    result.test_data_points = static_cast<int>(test_data.size());

    ts::ARIMA model;
    model.load(model_path);
    result.param_count = model.parameter_count();

    int n_points = static_cast<int>(test_data.size());

    // Warm-up
    model.forecast(1);

    // Predict 1 point at a time (ARIMA forecasts from trained state)
    double total_mse = 0.0, total_mae = 0.0;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_points; ++i) {
        auto forecast = model.forecast(1);
        // Note: ARIMA always predicts from end of training, so we compare to first test point
        // This is a limitation of classical models for streaming
    }
    auto end = std::chrono::high_resolution_clock::now();

    // For accuracy, use single forecast comparison
    auto forecast = model.forecast(n_points);
    for (int i = 0; i < n_points && i < static_cast<int>(forecast.predictions.size()); ++i) {
        double error = forecast.predictions[i] - test_data[i];
        total_mse += error * error;
        total_mae += std::abs(error);
    }

    result.mse = total_mse / n_points;
    result.mae = total_mae / n_points;
    result.total_inference_s = std::chrono::duration<double>(end - start).count();
    result.latency_per_point_s = result.total_inference_s / n_points;
    result.throughput = n_points / result.total_inference_s;

    return result;
}

// Evaluate SES - single step forecast
EvalResult evaluate_ses(const std::string& model_path,
                        const std::vector<double>& test_data,
                        const std::string& dataset_name) {
    EvalResult result;
    result.dataset = dataset_name;
    result.model_name = "SES";
    result.is_multivariate = false;
    result.test_data_points = static_cast<int>(test_data.size());

    ts::SimpleExponentialSmoothing model;
    model.load(model_path);
    result.param_count = model.parameter_count();

    int n_points = static_cast<int>(test_data.size());

    // Warm-up
    model.forecast(1);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_points; ++i) {
        auto forecast = model.forecast(1);
    }
    auto end = std::chrono::high_resolution_clock::now();

    // For accuracy
    auto forecast = model.forecast(n_points);
    double total_mse = 0.0, total_mae = 0.0;
    for (int i = 0; i < n_points && i < static_cast<int>(forecast.predictions.size()); ++i) {
        double error = forecast.predictions[i] - test_data[i];
        total_mse += error * error;
        total_mae += std::abs(error);
    }

    result.mse = total_mse / n_points;
    result.mae = total_mae / n_points;
    result.total_inference_s = std::chrono::duration<double>(end - start).count();
    result.latency_per_point_s = result.total_inference_s / n_points;
    result.throughput = n_points / result.total_inference_s;

    return result;
}

// Evaluate HoltWinters - single step forecast
EvalResult evaluate_holtwinters(const std::string& model_path,
                                const std::vector<double>& test_data,
                                const std::string& dataset_name) {
    EvalResult result;
    result.dataset = dataset_name;
    result.model_name = "HoltWinters";
    result.is_multivariate = false;
    result.test_data_points = static_cast<int>(test_data.size());

    ts::HoltWinters model(24);
    model.load(model_path);
    result.param_count = model.parameter_count();

    int n_points = static_cast<int>(test_data.size());

    // Warm-up
    model.forecast(1);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_points; ++i) {
        auto forecast = model.forecast(1);
    }
    auto end = std::chrono::high_resolution_clock::now();

    // For accuracy
    auto forecast = model.forecast(n_points);
    double total_mse = 0.0, total_mae = 0.0;
    for (int i = 0; i < n_points && i < static_cast<int>(forecast.predictions.size()); ++i) {
        double error = forecast.predictions[i] - test_data[i];
        total_mse += error * error;
        total_mae += std::abs(error);
    }

    result.mse = total_mse / n_points;
    result.mae = total_mae / n_points;
    result.total_inference_s = std::chrono::duration<double>(end - start).count();
    result.latency_per_point_s = result.total_inference_s / n_points;
    result.throughput = n_points / result.total_inference_s;

    return result;
}

// Evaluate Prophet - single step forecast
EvalResult evaluate_prophet(const std::string& model_path,
                            const std::vector<double>& test_data,
                            const std::string& dataset_name) {
    EvalResult result;
    result.dataset = dataset_name;
    result.model_name = "Prophet";
    result.is_multivariate = false;
    result.test_data_points = static_cast<int>(test_data.size());

    ts::Prophet model;
    model.load(model_path);
    result.param_count = model.parameter_count();

    int n_points = static_cast<int>(test_data.size());

    // Warm-up
    model.forecast(1);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_points; ++i) {
        auto forecast = model.forecast(1);
    }
    auto end = std::chrono::high_resolution_clock::now();

    // For accuracy
    auto forecast = model.forecast(n_points);
    double total_mse = 0.0, total_mae = 0.0;
    for (int i = 0; i < n_points && i < static_cast<int>(forecast.predictions.size()); ++i) {
        double error = forecast.predictions[i] - test_data[i];
        total_mse += error * error;
        total_mae += std::abs(error);
    }

    result.mse = total_mse / n_points;
    result.mae = total_mae / n_points;
    result.total_inference_s = std::chrono::duration<double>(end - start).count();
    result.latency_per_point_s = result.total_inference_s / n_points;
    result.throughput = n_points / result.total_inference_s;

    return result;
}

// Evaluate DLinear - streaming: predict 1 point at a time with sliding window
EvalResult evaluate_dlinear(const std::string& model_path,
                            const std::vector<double>& test_data,
                            const std::string& dataset_name,
                            const std::string& model_name,
                            bool is_multivariate = false) {
    EvalResult result;
    result.dataset = dataset_name;
    result.model_name = model_name;
    result.is_multivariate = is_multivariate;
    result.test_data_points = static_cast<int>(test_data.size());

    ts::DLinear model;
    model.load(model_path);
    result.param_count = model.parameter_count();

    int seq_len = model.seq_len();

    // Number of points we can predict (need seq_len history + 1 target)
    int n_predictions = static_cast<int>(test_data.size()) - seq_len;
    if (n_predictions <= 0) {
        result.mse = result.mae = 0;
        result.total_inference_s = 0;
        result.latency_per_point_s = 0;
        result.throughput = 0;
        return result;
    }

    // Warm-up
    std::vector<double> warmup_input(test_data.begin(), test_data.begin() + seq_len);
    model.predict(warmup_input);

    // Prepare inputs outside timing
    std::vector<std::vector<double>> inputs(n_predictions);
    for (int i = 0; i < n_predictions; ++i) {
        inputs[i] = std::vector<double>(test_data.begin() + i, test_data.begin() + i + seq_len);
    }

    // Timed inference - predict 1 point at a time
    std::vector<double> predictions(n_predictions);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_predictions; ++i) {
        auto pred = model.predict(inputs[i]);
        predictions[i] = pred[0];  // Take only first predicted point (streaming)
    }
    auto end = std::chrono::high_resolution_clock::now();

    // Compute metrics
    double total_mse = 0.0, total_mae = 0.0;
    for (int i = 0; i < n_predictions; ++i) {
        double actual = test_data[seq_len + i];
        double error = predictions[i] - actual;
        total_mse += error * error;
        total_mae += std::abs(error);
    }

    result.mse = total_mse / n_predictions;
    result.mae = total_mae / n_predictions;
    result.total_inference_s = std::chrono::duration<double>(end - start).count();
    result.latency_per_point_s = result.total_inference_s / n_predictions;
    result.throughput = n_predictions / result.total_inference_s;

    return result;
}

// Evaluate NLinear - streaming
EvalResult evaluate_nlinear(const std::string& model_path,
                            const std::vector<double>& test_data,
                            const std::string& dataset_name) {
    EvalResult result;
    result.dataset = dataset_name;
    result.model_name = "NLinear";
    result.is_multivariate = false;
    result.test_data_points = static_cast<int>(test_data.size());

    ts::NLinear model;
    model.load(model_path);
    result.param_count = model.parameter_count();

    int seq_len = model.seq_len();
    int n_predictions = static_cast<int>(test_data.size()) - seq_len;

    if (n_predictions <= 0) {
        result.mse = result.mae = 0;
        result.total_inference_s = 0;
        result.latency_per_point_s = 0;
        result.throughput = 0;
        return result;
    }

    // Warm-up
    std::vector<double> warmup_input(test_data.begin(), test_data.begin() + seq_len);
    model.predict(warmup_input);

    // Prepare inputs
    std::vector<std::vector<double>> inputs(n_predictions);
    for (int i = 0; i < n_predictions; ++i) {
        inputs[i] = std::vector<double>(test_data.begin() + i, test_data.begin() + i + seq_len);
    }

    // Timed inference
    std::vector<double> predictions(n_predictions);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_predictions; ++i) {
        auto pred = model.predict(inputs[i]);
        predictions[i] = pred[0];
    }
    auto end = std::chrono::high_resolution_clock::now();

    // Metrics
    double total_mse = 0.0, total_mae = 0.0;
    for (int i = 0; i < n_predictions; ++i) {
        double actual = test_data[seq_len + i];
        double error = predictions[i] - actual;
        total_mse += error * error;
        total_mae += std::abs(error);
    }

    result.mse = total_mse / n_predictions;
    result.mae = total_mae / n_predictions;
    result.total_inference_s = std::chrono::duration<double>(end - start).count();
    result.latency_per_point_s = result.total_inference_s / n_predictions;
    result.throughput = n_predictions / result.total_inference_s;

    return result;
}

// Evaluate Linear - streaming
EvalResult evaluate_linear(const std::string& model_path,
                           const std::vector<double>& test_data,
                           const std::string& dataset_name) {
    EvalResult result;
    result.dataset = dataset_name;
    result.model_name = "Linear";
    result.is_multivariate = false;
    result.test_data_points = static_cast<int>(test_data.size());

    ts::Linear model;
    model.load(model_path);
    result.param_count = model.parameter_count();

    int seq_len = model.seq_len();
    int n_predictions = static_cast<int>(test_data.size()) - seq_len;

    if (n_predictions <= 0) {
        result.mse = result.mae = 0;
        result.total_inference_s = 0;
        result.latency_per_point_s = 0;
        result.throughput = 0;
        return result;
    }

    // Warm-up
    std::vector<double> warmup_input(test_data.begin(), test_data.begin() + seq_len);
    model.predict(warmup_input);

    // Prepare inputs
    std::vector<std::vector<double>> inputs(n_predictions);
    for (int i = 0; i < n_predictions; ++i) {
        inputs[i] = std::vector<double>(test_data.begin() + i, test_data.begin() + i + seq_len);
    }

    // Timed inference
    std::vector<double> predictions(n_predictions);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_predictions; ++i) {
        auto pred = model.predict(inputs[i]);
        predictions[i] = pred[0];
    }
    auto end = std::chrono::high_resolution_clock::now();

    // Metrics
    double total_mse = 0.0, total_mae = 0.0;
    for (int i = 0; i < n_predictions; ++i) {
        double actual = test_data[seq_len + i];
        double error = predictions[i] - actual;
        total_mse += error * error;
        total_mae += std::abs(error);
    }

    result.mse = total_mse / n_predictions;
    result.mae = total_mae / n_predictions;
    result.total_inference_s = std::chrono::duration<double>(end - start).count();
    result.latency_per_point_s = result.total_inference_s / n_predictions;
    result.throughput = n_predictions / result.total_inference_s;

    return result;
}

void evaluate_dataset(const std::string& name, const std::string& subdir,
                      std::vector<EvalResult>& univariate_results,
                      std::vector<EvalResult>& multivariate_results) {
    std::string test_path = BASE_DIR + "/" + subdir + "/" + name + "_test_standard.csv";
    std::string models_dir = BASE_DIR + "/" + subdir + "/models";

    if (!fs::exists(test_path)) {
        std::cout << "Test file not found: " << test_path << std::endl;
        return;
    }

    auto test_data = load_univariate(test_path);

    std::cout << "\n--- " << name << " (test rows: " << test_data.size() << ") ---" << std::endl;

    // ARIMA
    std::string arima_path = models_dir + "/" + name + "_arima.bin";
    if (fs::exists(arima_path)) {
        auto result = evaluate_arima(arima_path, test_data, name);
        univariate_results.push_back(result);
        print_result(result);
    }

    // SES
    std::string ses_path = models_dir + "/" + name + "_ses.bin";
    if (fs::exists(ses_path)) {
        auto result = evaluate_ses(ses_path, test_data, name);
        univariate_results.push_back(result);
        print_result(result);
    }

    // HoltWinters
    std::string hw_path = models_dir + "/" + name + "_holtwinters.bin";
    if (fs::exists(hw_path)) {
        auto result = evaluate_holtwinters(hw_path, test_data, name);
        univariate_results.push_back(result);
        print_result(result);
    }

    // Prophet
    std::string prophet_path = models_dir + "/" + name + "_prophet.bin";
    if (fs::exists(prophet_path)) {
        auto result = evaluate_prophet(prophet_path, test_data, name);
        univariate_results.push_back(result);
        print_result(result);
    }

    // DLinear (univariate)
    std::string dlinear_path = models_dir + "/" + name + "_dlinear.bin";
    if (fs::exists(dlinear_path)) {
        auto result = evaluate_dlinear(dlinear_path, test_data, name, "DLinear");
        univariate_results.push_back(result);
        print_result(result);
    }

    // NLinear
    std::string nlinear_path = models_dir + "/" + name + "_nlinear.bin";
    if (fs::exists(nlinear_path)) {
        auto result = evaluate_nlinear(nlinear_path, test_data, name);
        univariate_results.push_back(result);
        print_result(result);
    }

    // Linear
    std::string linear_path = models_dir + "/" + name + "_linear.bin";
    if (fs::exists(linear_path)) {
        auto result = evaluate_linear(linear_path, test_data, name);
        univariate_results.push_back(result);
        print_result(result);
    }

    // DLinear MV (multivariate)
    std::string dlinear_mv_path = models_dir + "/" + name + "_dlinear_mv.bin";
    if (fs::exists(dlinear_mv_path)) {
        auto result = evaluate_dlinear(dlinear_mv_path, test_data, name, "DLinear-MV", true);
        multivariate_results.push_back(result);
        print_result(result);
    }
}

void save_results_csv(const std::vector<EvalResult>& results, const std::string& filepath) {
    std::ofstream file(filepath);
    file << "Dataset,Model,MSE,MAE,Parameters,TestDataPoints,"
         << "TotalInference_s,Latency_s,Throughput\n";

    for (const auto& r : results) {
        file << r.dataset << ","
             << r.model_name << ","
             << std::fixed << std::setprecision(8) << r.mse << ","
             << r.mae << ","
             << r.param_count << ","
             << r.test_data_points << ","
             << std::setprecision(6) << r.total_inference_s << ","
             << std::scientific << std::setprecision(6) << r.latency_per_point_s << ","
             << std::fixed << std::setprecision(2) << r.throughput << "\n";
    }

    file.close();
    std::cout << "Results saved to: " << filepath << std::endl;
}

void print_summary(const std::vector<EvalResult>& results, const std::string& title) {
    if (results.empty()) return;

    std::cout << "\n=== " << title << " ===" << std::endl;

    std::map<std::string, std::vector<EvalResult>> by_model;
    for (const auto& r : results) {
        by_model[r.model_name].push_back(r);
    }

    std::cout << std::left << std::setw(14) << "Model"
              << std::right << std::setw(12) << "Avg MSE"
              << std::setw(11) << "Avg MAE"
              << std::setw(9) << "Params"
              << std::setw(14) << "Latency(s)"
              << std::setw(14) << "Throughput" << std::endl;
    std::cout << std::string(74, '-') << std::endl;

    for (const auto& [model_name, model_results] : by_model) {
        double avg_mse = 0, avg_mae = 0, avg_latency = 0, avg_throughput = 0;
        size_t avg_params = 0;

        for (const auto& r : model_results) {
            avg_mse += r.mse;
            avg_mae += r.mae;
            avg_params += r.param_count;
            avg_latency += r.latency_per_point_s;
            avg_throughput += r.throughput;
        }

        size_t n = model_results.size();
        std::cout << std::left << std::setw(14) << model_name
                  << std::right << std::fixed << std::setprecision(6)
                  << std::setw(12) << avg_mse / n
                  << std::setw(11) << avg_mae / n
                  << std::setw(9) << avg_params / n
                  << std::scientific << std::setprecision(2)
                  << std::setw(14) << avg_latency / n
                  << std::fixed << std::setprecision(2)
                  << std::setw(14) << avg_throughput / n << std::endl;
    }
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << "Model Evaluation - Streaming Mode (1 point at a time)\n";
    std::cout << "========================================================================\n";
    std::cout << "\nMetrics:\n";
    std::cout << "  - MSE/MAE: Mean Squared/Absolute Error\n";
    std::cout << "  - Params: Model parameter count\n";
    std::cout << "  - TestRows: Total rows in test dataset\n";
    std::cout << "  - Total(s): Total inference time in seconds\n";
    std::cout << "  - Latency(s): Seconds per data point prediction\n";
    std::cout << "  - Throughput: Data points predicted per second\n";
    std::cout << std::endl;

    std::vector<EvalResult> univariate_results;
    std::vector<EvalResult> multivariate_results;

    print_header();

    // ETT datasets
    evaluate_dataset("ETTh1", "ETT-small", univariate_results, multivariate_results);
    evaluate_dataset("ETTh2", "ETT-small", univariate_results, multivariate_results);
    evaluate_dataset("ETTm1", "ETT-small", univariate_results, multivariate_results);
    evaluate_dataset("ETTm2", "ETT-small", univariate_results, multivariate_results);

    // Other datasets
    evaluate_dataset("exchange_rate", "exchange_rate", univariate_results, multivariate_results);
    evaluate_dataset("illness", "illness", univariate_results, multivariate_results);
    evaluate_dataset("weather", "weather", univariate_results, multivariate_results);
    evaluate_dataset("electricity", "electricity", univariate_results, multivariate_results);

    std::cout << std::string(107, '-') << std::endl;

    // Print summaries
    print_summary(univariate_results, "Univariate Models Summary");
    print_summary(multivariate_results, "Multivariate Models Summary");

    // Save to CSV files
    std::cout << std::endl;
    save_results_csv(univariate_results, BASE_DIR + "/evaluation_univariate.csv");
    save_results_csv(multivariate_results, BASE_DIR + "/evaluation_multivariate.csv");

    // Combined results
    std::vector<EvalResult> all_results;
    all_results.insert(all_results.end(), univariate_results.begin(), univariate_results.end());
    all_results.insert(all_results.end(), multivariate_results.begin(), multivariate_results.end());
    save_results_csv(all_results, BASE_DIR + "/evaluation_all.csv");

    std::cout << "\nEvaluation complete.\n";

    return 0;
}
