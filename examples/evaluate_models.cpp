#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <sstream>
#include "ts_forecast.hpp"
#include "csv_reader.hpp"

namespace fs = std::filesystem;

const std::string BASE_DIR = "/Users/zimenglyu/Documents/code/git/ts-forecast-cpp/benchmark_datasets";

struct EvalResult {
    std::string model_name;
    std::string dataset;
    double mse;
    double mae;
    size_t param_count;
    double latency_ms;      // Average inference latency per sample (ms)
    double throughput;      // Samples per second
    int test_samples;
    bool is_multivariate;
};

// Load univariate data (OT column only)
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
              << std::setw(15) << "Dataset"
              << std::setw(18) << "Model"
              << std::right
              << std::setw(12) << "MSE"
              << std::setw(12) << "MAE"
              << std::setw(12) << "Params"
              << std::setw(14) << "Latency(ms)"
              << std::setw(14) << "Throughput"
              << std::setw(10) << "Samples"
              << std::endl;
    std::cout << std::string(107, '-') << std::endl;
}

void print_result(const EvalResult& r) {
    std::cout << std::left
              << std::setw(15) << r.dataset
              << std::setw(18) << r.model_name
              << std::right << std::fixed << std::setprecision(6)
              << std::setw(12) << r.mse
              << std::setw(12) << r.mae
              << std::setw(12) << r.param_count
              << std::setprecision(4)
              << std::setw(14) << r.latency_ms
              << std::setprecision(2)
              << std::setw(14) << r.throughput
              << std::setw(10) << r.test_samples
              << std::endl;
}

// Generic evaluation function template
template<typename ModelType, typename PredictFunc>
EvalResult evaluate_model(const std::string& model_path,
                          const std::vector<double>& test_data,
                          const std::string& dataset_name,
                          const std::string& model_name,
                          int seq_len, int pred_len,
                          PredictFunc predict_func) {
    EvalResult result;
    result.dataset = dataset_name;
    result.model_name = model_name;
    result.is_multivariate = false;

    ModelType model;
    model.load(model_path);

    result.param_count = model.parameter_count();

    int n_samples = static_cast<int>(test_data.size()) - seq_len - pred_len + 1;
    if (n_samples <= 0) {
        result.mse = result.mae = 0;
        result.latency_ms = result.throughput = 0;
        result.test_samples = 0;
        return result;
    }

    result.test_samples = n_samples;

    double total_mse = 0.0, total_mae = 0.0;

    // Warm-up
    std::vector<double> warmup_input(test_data.begin(), test_data.begin() + seq_len);
    predict_func(model, warmup_input);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n_samples; ++i) {
        std::vector<double> input(test_data.begin() + i, test_data.begin() + i + seq_len);
        std::vector<double> actual(test_data.begin() + i + seq_len,
                                   test_data.begin() + i + seq_len + pred_len);

        auto pred = predict_func(model, input);

        for (int j = 0; j < pred_len && j < static_cast<int>(pred.size()); ++j) {
            double error = pred[j] - actual[j];
            total_mse += error * error;
            total_mae += std::abs(error);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    int total_predictions = n_samples * pred_len;
    result.mse = total_mse / total_predictions;
    result.mae = total_mae / total_predictions;
    result.latency_ms = total_time_ms / n_samples;
    result.throughput = n_samples / (total_time_ms / 1000.0);

    return result;
}

// Evaluate ARIMA model
EvalResult evaluate_arima(const std::string& model_path,
                          const std::vector<double>& test_data,
                          const std::string& dataset_name,
                          int pred_len) {
    EvalResult result;
    result.dataset = dataset_name;
    result.model_name = "ARIMA";
    result.is_multivariate = false;

    ts::ARIMA model;
    model.load(model_path);
    result.param_count = model.parameter_count();

    int n_samples = static_cast<int>(test_data.size()) - pred_len + 1;
    if (n_samples <= 0) {
        result.mse = result.mae = 0;
        result.latency_ms = result.throughput = 0;
        result.test_samples = 0;
        return result;
    }

    result.test_samples = n_samples;
    double total_mse = 0.0, total_mae = 0.0;

    // ARIMA uses stored data for forecasting
    auto start = std::chrono::high_resolution_clock::now();

    auto forecast = model.forecast(pred_len);
    for (int j = 0; j < pred_len; ++j) {
        double error = forecast.predictions[j] - test_data[j];
        total_mse += error * error;
        total_mae += std::abs(error);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    result.mse = total_mse / pred_len;
    result.mae = total_mae / pred_len;
    result.latency_ms = total_time_ms;
    result.throughput = 1.0 / (total_time_ms / 1000.0);
    result.test_samples = 1;

    return result;
}

// Evaluate SES model
EvalResult evaluate_ses(const std::string& model_path,
                        const std::vector<double>& test_data,
                        const std::string& dataset_name,
                        int pred_len) {
    EvalResult result;
    result.dataset = dataset_name;
    result.model_name = "SES";
    result.is_multivariate = false;

    ts::SimpleExponentialSmoothing model;
    model.load(model_path);
    result.param_count = model.parameter_count();

    auto start = std::chrono::high_resolution_clock::now();

    auto forecast = model.forecast(pred_len);
    double total_mse = 0.0, total_mae = 0.0;
    for (int j = 0; j < pred_len; ++j) {
        double error = forecast.predictions[j] - test_data[j];
        total_mse += error * error;
        total_mae += std::abs(error);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    result.mse = total_mse / pred_len;
    result.mae = total_mae / pred_len;
    result.latency_ms = total_time_ms;
    result.throughput = 1.0 / (total_time_ms / 1000.0);
    result.test_samples = 1;

    return result;
}

// Evaluate HoltWinters model
EvalResult evaluate_holtwinters(const std::string& model_path,
                                const std::vector<double>& test_data,
                                const std::string& dataset_name,
                                int pred_len) {
    EvalResult result;
    result.dataset = dataset_name;
    result.model_name = "HoltWinters";
    result.is_multivariate = false;

    ts::HoltWinters model(24);
    model.load(model_path);
    result.param_count = model.parameter_count();

    auto start = std::chrono::high_resolution_clock::now();

    auto forecast = model.forecast(pred_len);
    double total_mse = 0.0, total_mae = 0.0;
    for (int j = 0; j < pred_len; ++j) {
        double error = forecast.predictions[j] - test_data[j];
        total_mse += error * error;
        total_mae += std::abs(error);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    result.mse = total_mse / pred_len;
    result.mae = total_mae / pred_len;
    result.latency_ms = total_time_ms;
    result.throughput = 1.0 / (total_time_ms / 1000.0);
    result.test_samples = 1;

    return result;
}

// Evaluate Prophet model
EvalResult evaluate_prophet(const std::string& model_path,
                            const std::vector<double>& test_data,
                            const std::string& dataset_name,
                            int pred_len) {
    EvalResult result;
    result.dataset = dataset_name;
    result.model_name = "Prophet";
    result.is_multivariate = false;

    ts::Prophet model;
    model.load(model_path);
    result.param_count = model.parameter_count();

    auto start = std::chrono::high_resolution_clock::now();

    auto forecast = model.forecast(pred_len);
    double total_mse = 0.0, total_mae = 0.0;
    for (int j = 0; j < pred_len; ++j) {
        double error = forecast.predictions[j] - test_data[j];
        total_mse += error * error;
        total_mae += std::abs(error);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    result.mse = total_mse / pred_len;
    result.mae = total_mae / pred_len;
    result.latency_ms = total_time_ms;
    result.throughput = 1.0 / (total_time_ms / 1000.0);
    result.test_samples = 1;

    return result;
}

// Evaluate DLinear model
EvalResult evaluate_dlinear(const std::string& model_path,
                            const std::vector<double>& test_data,
                            const std::string& dataset_name,
                            const std::string& model_name,
                            bool is_multivariate = false) {
    EvalResult result;
    result.dataset = dataset_name;
    result.model_name = model_name;
    result.is_multivariate = is_multivariate;

    ts::DLinear model;
    model.load(model_path);

    result.param_count = model.parameter_count();

    int seq_len = model.seq_len();
    int pred_len = model.pred_len();

    int n_samples = static_cast<int>(test_data.size()) - seq_len - pred_len + 1;
    if (n_samples <= 0) {
        result.mse = result.mae = 0;
        result.latency_ms = result.throughput = 0;
        result.test_samples = 0;
        return result;
    }

    result.test_samples = n_samples;

    double total_mse = 0.0, total_mae = 0.0;

    // Warm-up
    std::vector<double> warmup_input(test_data.begin(), test_data.begin() + seq_len);
    model.predict(warmup_input);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n_samples; ++i) {
        std::vector<double> input(test_data.begin() + i, test_data.begin() + i + seq_len);
        std::vector<double> actual(test_data.begin() + i + seq_len,
                                   test_data.begin() + i + seq_len + pred_len);

        auto pred = model.predict(input);

        for (int j = 0; j < pred_len; ++j) {
            double error = pred[j] - actual[j];
            total_mse += error * error;
            total_mae += std::abs(error);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    int total_predictions = n_samples * pred_len;
    result.mse = total_mse / total_predictions;
    result.mae = total_mae / total_predictions;
    result.latency_ms = total_time_ms / n_samples;
    result.throughput = n_samples / (total_time_ms / 1000.0);

    return result;
}

// Evaluate NLinear model
EvalResult evaluate_nlinear(const std::string& model_path,
                            const std::vector<double>& test_data,
                            const std::string& dataset_name) {
    EvalResult result;
    result.dataset = dataset_name;
    result.model_name = "NLinear";
    result.is_multivariate = false;

    ts::NLinear model;
    model.load(model_path);

    result.param_count = model.parameter_count();

    int seq_len = model.seq_len();
    int pred_len = model.pred_len();

    int n_samples = static_cast<int>(test_data.size()) - seq_len - pred_len + 1;
    if (n_samples <= 0) {
        result.mse = result.mae = 0;
        result.latency_ms = result.throughput = 0;
        result.test_samples = 0;
        return result;
    }

    result.test_samples = n_samples;

    double total_mse = 0.0, total_mae = 0.0;

    std::vector<double> warmup_input(test_data.begin(), test_data.begin() + seq_len);
    model.predict(warmup_input);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n_samples; ++i) {
        std::vector<double> input(test_data.begin() + i, test_data.begin() + i + seq_len);
        std::vector<double> actual(test_data.begin() + i + seq_len,
                                   test_data.begin() + i + seq_len + pred_len);

        auto pred = model.predict(input);

        for (int j = 0; j < pred_len; ++j) {
            double error = pred[j] - actual[j];
            total_mse += error * error;
            total_mae += std::abs(error);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    int total_predictions = n_samples * pred_len;
    result.mse = total_mse / total_predictions;
    result.mae = total_mae / total_predictions;
    result.latency_ms = total_time_ms / n_samples;
    result.throughput = n_samples / (total_time_ms / 1000.0);

    return result;
}

// Evaluate Linear model
EvalResult evaluate_linear(const std::string& model_path,
                           const std::vector<double>& test_data,
                           const std::string& dataset_name) {
    EvalResult result;
    result.dataset = dataset_name;
    result.model_name = "Linear";
    result.is_multivariate = false;

    ts::Linear model;
    model.load(model_path);

    result.param_count = model.parameter_count();

    int seq_len = model.seq_len();
    int pred_len = model.pred_len();

    int n_samples = static_cast<int>(test_data.size()) - seq_len - pred_len + 1;
    if (n_samples <= 0) {
        result.mse = result.mae = 0;
        result.latency_ms = result.throughput = 0;
        result.test_samples = 0;
        return result;
    }

    result.test_samples = n_samples;

    double total_mse = 0.0, total_mae = 0.0;

    std::vector<double> warmup_input(test_data.begin(), test_data.begin() + seq_len);
    model.predict(warmup_input);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n_samples; ++i) {
        std::vector<double> input(test_data.begin() + i, test_data.begin() + i + seq_len);
        std::vector<double> actual(test_data.begin() + i + seq_len,
                                   test_data.begin() + i + seq_len + pred_len);

        auto pred = model.predict(input);

        for (int j = 0; j < pred_len; ++j) {
            double error = pred[j] - actual[j];
            total_mse += error * error;
            total_mae += std::abs(error);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    int total_predictions = n_samples * pred_len;
    result.mse = total_mse / total_predictions;
    result.mae = total_mae / total_predictions;
    result.latency_ms = total_time_ms / n_samples;
    result.throughput = n_samples / (total_time_ms / 1000.0);

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
    int pred_len = 24;

    std::cout << "\n--- " << name << " ---" << std::endl;

    // ARIMA
    std::string arima_path = models_dir + "/" + name + "_arima.bin";
    if (fs::exists(arima_path)) {
        auto result = evaluate_arima(arima_path, test_data, name, pred_len);
        univariate_results.push_back(result);
        print_result(result);
    }

    // SES
    std::string ses_path = models_dir + "/" + name + "_ses.bin";
    if (fs::exists(ses_path)) {
        auto result = evaluate_ses(ses_path, test_data, name, pred_len);
        univariate_results.push_back(result);
        print_result(result);
    }

    // HoltWinters
    std::string hw_path = models_dir + "/" + name + "_holtwinters.bin";
    if (fs::exists(hw_path)) {
        auto result = evaluate_holtwinters(hw_path, test_data, name, pred_len);
        univariate_results.push_back(result);
        print_result(result);
    }

    // Prophet
    std::string prophet_path = models_dir + "/" + name + "_prophet.bin";
    if (fs::exists(prophet_path)) {
        auto result = evaluate_prophet(prophet_path, test_data, name, pred_len);
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
    file << "Dataset,Model,MSE,MAE,Parameters,Latency_ms,Throughput,TestSamples\n";

    for (const auto& r : results) {
        file << r.dataset << ","
             << r.model_name << ","
             << std::fixed << std::setprecision(8) << r.mse << ","
             << r.mae << ","
             << r.param_count << ","
             << std::setprecision(4) << r.latency_ms << ","
             << r.throughput << ","
             << r.test_samples << "\n";
    }

    file.close();
    std::cout << "Results saved to: " << filepath << std::endl;
}

void print_summary(const std::vector<EvalResult>& results, const std::string& title) {
    std::cout << "\n=== " << title << " ===" << std::endl;

    std::map<std::string, std::vector<EvalResult>> by_model;
    for (const auto& r : results) {
        by_model[r.model_name].push_back(r);
    }

    std::cout << std::left << std::setw(15) << "Model"
              << std::right << std::setw(12) << "Avg MSE"
              << std::setw(12) << "Avg MAE"
              << std::setw(12) << "Avg Params"
              << std::setw(14) << "Avg Latency" << std::endl;
    std::cout << std::string(65, '-') << std::endl;

    for (const auto& [model_name, model_results] : by_model) {
        double avg_mse = 0, avg_mae = 0, avg_latency = 0;
        size_t avg_params = 0;

        for (const auto& r : model_results) {
            avg_mse += r.mse;
            avg_mae += r.mae;
            avg_params += r.param_count;
            avg_latency += r.latency_ms;
        }

        size_t n = model_results.size();
        std::cout << std::left << std::setw(15) << model_name
                  << std::right << std::fixed << std::setprecision(6)
                  << std::setw(12) << avg_mse / n
                  << std::setw(12) << avg_mae / n
                  << std::setw(12) << avg_params / n
                  << std::setprecision(4)
                  << std::setw(14) << avg_latency / n << std::endl;
    }
}

int main() {
    std::cout << "Model Evaluation: MSE, MAE, Parameters, Latency, Throughput" << std::endl;
    std::cout << "============================================================" << std::endl;

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

    // Save to separate CSV files
    std::cout << std::endl;
    save_results_csv(univariate_results, BASE_DIR + "/evaluation_univariate.csv");
    save_results_csv(multivariate_results, BASE_DIR + "/evaluation_multivariate.csv");

    // Also save combined results
    std::vector<EvalResult> all_results;
    all_results.insert(all_results.end(), univariate_results.begin(), univariate_results.end());
    all_results.insert(all_results.end(), multivariate_results.begin(), multivariate_results.end());
    save_results_csv(all_results, BASE_DIR + "/evaluation_all.csv");

    return 0;
}
