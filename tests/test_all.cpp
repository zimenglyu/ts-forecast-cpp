#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <string>
#include "ts_forecast.hpp"

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

void test_stats() {
    std::cout << "\n=== Testing Statistics Functions ===" << std::endl;

    std::vector<double> data = {1, 2, 3, 4, 5};

    test_assert(std::abs(ts::stats::mean(data) - 3.0) < 1e-10, "Mean calculation");
    test_assert(std::abs(ts::stats::variance(data) - 2.5) < 1e-10, "Variance calculation");
    test_assert(std::abs(ts::stats::std_dev(data) - std::sqrt(2.5)) < 1e-10, "Standard deviation");

    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {2, 4, 6, 8, 10};
    test_assert(std::abs(ts::stats::correlation(x, y) - 1.0) < 1e-10, "Perfect correlation");

    std::vector<double> diff = ts::stats::difference(data, 1);
    test_assert(diff.size() == 4, "Differencing size");
    test_assert(std::abs(diff[0] - 1.0) < 1e-10, "Differencing values");
}

void test_matrix() {
    std::cout << "\n=== Testing Matrix Operations ===" << std::endl;

    auto I = ts::matrix::identity(3);
    test_assert(I[0][0] == 1.0 && I[1][1] == 1.0 && I[2][2] == 1.0, "Identity matrix diagonal");
    test_assert(I[0][1] == 0.0 && I[1][0] == 0.0, "Identity matrix off-diagonal");

    ts::matrix::Matrix A = {{1, 2}, {3, 4}};
    auto At = ts::matrix::transpose(A);
    test_assert(At[0][1] == 3.0 && At[1][0] == 2.0, "Transpose");

    ts::matrix::Matrix B = {{1, 0}, {0, 1}};
    auto C = ts::matrix::multiply(A, B);
    test_assert(C[0][0] == A[0][0] && C[1][1] == A[1][1], "Matrix multiply with identity");

    ts::matrix::Vector v = {1, 2};
    auto result = ts::matrix::multiply(A, v);
    test_assert(std::abs(result[0] - 5.0) < 1e-10, "Matrix-vector multiply");

    ts::matrix::Matrix D = {{4, 7}, {2, 6}};
    auto D_inv = ts::matrix::inverse(D);
    auto product = ts::matrix::multiply(D, D_inv);
    test_assert(std::abs(product[0][0] - 1.0) < 1e-6 && std::abs(product[1][1] - 1.0) < 1e-6,
                "Matrix inverse");
}

void test_arima() {
    std::cout << "\n=== Testing ARIMA ===" << std::endl;

    // Generate AR(1) process
    std::vector<double> data(100);
    data[0] = 0;
    for (int i = 1; i < 100; ++i) {
        data[i] = 0.7 * data[i-1] + (std::rand() % 100 - 50) / 50.0;
    }

    ts::ARIMA arima(1, 0, 0);
    arima.fit(data);
    test_assert(arima.is_fitted(), "ARIMA fitting");

    auto forecast = arima.forecast(5);
    test_assert(forecast.predictions.size() == 5, "ARIMA forecast size");
    test_assert(forecast.lower_bound.size() == 5, "ARIMA confidence intervals");

    // Test with differencing
    std::vector<double> trend_data(100);
    for (int i = 0; i < 100; ++i) {
        trend_data[i] = 50 + 0.5 * i + (std::rand() % 10 - 5);
    }

    ts::ARIMA arima_d(1, 1, 0);
    arima_d.fit(trend_data);
    test_assert(arima_d.is_fitted(), "ARIMA with differencing");
}

void test_exponential_smoothing() {
    std::cout << "\n=== Testing Exponential Smoothing ===" << std::endl;

    std::vector<double> data(50);
    for (int i = 0; i < 50; ++i) {
        data[i] = 100 + (std::rand() % 20 - 10);
    }

    // Simple Exponential Smoothing
    ts::SimpleExponentialSmoothing ses;
    ses.fit(data);
    test_assert(ses.is_fitted(), "SES fitting");
    test_assert(ses.alpha() > 0 && ses.alpha() < 1, "SES alpha in valid range");

    auto ses_forecast = ses.forecast(5);
    test_assert(ses_forecast.predictions.size() == 5, "SES forecast size");

    // Holt's Linear
    std::vector<double> trend_data(50);
    for (int i = 0; i < 50; ++i) {
        trend_data[i] = 100 + 2 * i + (std::rand() % 10 - 5);
    }

    ts::HoltLinear holt;
    holt.fit(trend_data);
    test_assert(holt.is_fitted(), "Holt fitting");

    auto holt_forecast = holt.forecast(5);
    // Trend should continue upward
    test_assert(holt_forecast.predictions[4] > holt_forecast.predictions[0], "Holt captures trend");

    // Holt-Winters
    std::vector<double> seasonal_data(48);  // 4 years of monthly data
    for (int i = 0; i < 48; ++i) {
        seasonal_data[i] = 100 + 10 * std::sin(2 * M_PI * i / 12) + (std::rand() % 5 - 2);
    }

    ts::HoltWinters hw(12, ts::HoltWinters::SeasonalType::ADDITIVE);
    hw.fit(seasonal_data);
    test_assert(hw.is_fitted(), "Holt-Winters fitting");
    test_assert(hw.period() == 12, "Holt-Winters period");
}

void test_prophet() {
    std::cout << "\n=== Testing Prophet ===" << std::endl;

    std::vector<double> data(100);
    for (int i = 0; i < 100; ++i) {
        data[i] = 50 + 0.5 * i + 10 * std::sin(2 * M_PI * i / 7) + (std::rand() % 5 - 2);
    }

    ts::Prophet prophet;
    prophet.fit(data);
    test_assert(prophet.is_fitted(), "Prophet fitting");

    auto forecast = prophet.forecast(14);
    test_assert(forecast.predictions.size() == 14, "Prophet forecast size");

    // Test with timestamps
    std::vector<double> timestamps(100);
    for (int i = 0; i < 100; ++i) {
        timestamps[i] = static_cast<double>(i);
    }

    ts::Prophet prophet2;
    prophet2.fit(timestamps, data);
    test_assert(prophet2.is_fitted(), "Prophet with timestamps");

    auto trend = prophet2.get_trend(timestamps);
    test_assert(trend.size() == 100, "Prophet trend size");
}

void test_gradient_boosting() {
    std::cout << "\n=== Testing Gradient Boosting ===" << std::endl;

    // Create simple regression data
    std::vector<std::vector<double>> X;
    std::vector<double> y;

    for (int i = 0; i < 100; ++i) {
        double x1 = i / 10.0;
        double x2 = std::sin(i / 5.0);
        X.push_back({x1, x2});
        y.push_back(2 * x1 + 3 * x2 + (std::rand() % 10 - 5) / 10.0);
    }

    ts::GradientBoosting gb(20, 3, 0.1);
    gb.fit(X, y);

    auto predictions = gb.predict(X);
    test_assert(predictions.size() == 100, "GB prediction size");

    // Check that predictions are reasonable
    double mse = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        mse += (y[i] - predictions[i]) * (y[i] - predictions[i]);
    }
    mse /= y.size();
    test_assert(mse < 10.0, "GB reasonable MSE");

    // Time Series Gradient Boosting
    std::vector<double> ts_data(100);
    for (int i = 0; i < 100; ++i) {
        ts_data[i] = 50 + 0.3 * i + (std::rand() % 10 - 5);
    }

    ts::TimeSeriesGradientBoosting tsgb(5, 20, 3, 0.1);
    tsgb.fit(ts_data);
    test_assert(tsgb.is_fitted(), "TS Gradient Boosting fitting");

    auto ts_forecast = tsgb.forecast(10);
    test_assert(ts_forecast.predictions.size() == 10, "TS GB forecast size");
}

void test_evaluation() {
    std::cout << "\n=== Testing Evaluation Metrics ===" << std::endl;

    std::vector<double> actual = {1, 2, 3, 4, 5};
    std::vector<double> predicted = {1.1, 2.2, 2.9, 4.1, 4.8};

    auto metrics = ts::evaluate(actual, predicted);

    test_assert(metrics.mae > 0, "MAE positive");
    test_assert(metrics.mse > 0, "MSE positive");
    test_assert(metrics.rmse > 0, "RMSE positive");
    test_assert(metrics.rmse == std::sqrt(metrics.mse), "RMSE = sqrt(MSE)");
    test_assert(metrics.r2 > 0.9, "High R2 for good predictions");

    // Perfect predictions
    auto perfect_metrics = ts::evaluate(actual, actual);
    test_assert(std::abs(perfect_metrics.mae) < 1e-10, "Perfect MAE");
    test_assert(std::abs(perfect_metrics.r2 - 1.0) < 1e-10, "Perfect R2");
}

void test_train_test_split() {
    std::cout << "\n=== Testing Train/Test Split ===" << std::endl;

    std::vector<double> data(100);
    for (int i = 0; i < 100; ++i) {
        data[i] = i;
    }

    auto [train, test] = ts::train_test_split(data, 0.2);

    test_assert(train.size() == 80, "Train size");
    test_assert(test.size() == 20, "Test size");
    test_assert(train[0] == 0, "Train starts at beginning");
    test_assert(test[0] == 80, "Test starts after train");
}

void test_train_val_test_split() {
    std::cout << "\n=== Testing Train/Val/Test Split (70/15/15) ===" << std::endl;

    // Test univariate split
    std::vector<double> data(100);
    for (int i = 0; i < 100; ++i) {
        data[i] = static_cast<double>(i);
    }

    auto split = ts::train_val_test_split(data, 0.7, 0.15);

    test_assert(split.train.size() == 70, "Train size (70%)");
    test_assert(split.val.size() == 15, "Validation size (15%)");
    test_assert(split.test.size() == 15, "Test size (15%)");
    test_assert(split.train[0] == 0, "Train starts at 0");
    test_assert(split.val[0] == 70, "Val starts at 70");
    test_assert(split.test[0] == 85, "Test starts at 85");
    test_assert(split.train.back() == 69, "Train ends at 69");
    test_assert(split.test.back() == 99, "Test ends at 99");

    // Test multivariate split
    std::vector<std::vector<double>> mv_data(100);
    for (int i = 0; i < 100; ++i) {
        mv_data[i] = {static_cast<double>(i), static_cast<double>(i * 2)};
    }

    auto mv_split = ts::train_val_test_split(mv_data, 0.7, 0.15);

    test_assert(mv_split.train.size() == 70, "Multivariate train size");
    test_assert(mv_split.val.size() == 15, "Multivariate val size");
    test_assert(mv_split.test.size() == 15, "Multivariate test size");
    test_assert(mv_split.train[0][0] == 0, "Multivariate train starts correctly");
    test_assert(mv_split.test[14][0] == 99, "Multivariate test ends correctly");

    // Test different ratios (60/20/20)
    auto split2 = ts::train_val_test_split(data, 0.6, 0.2);
    test_assert(split2.train.size() == 60, "60/20/20 train size");
    test_assert(split2.val.size() == 20, "60/20/20 val size");
    test_assert(split2.test.size() == 20, "60/20/20 test size");
}

void test_minmax_scaler() {
    std::cout << "\n=== Testing MinMax Scaler ===" << std::endl;

    // Test univariate scaling
    std::vector<double> data = {0, 25, 50, 75, 100};

    ts::MinMaxScaler scaler;
    scaler.fit(data);

    test_assert(scaler.is_fitted(), "MinMax scaler fitted");
    test_assert(scaler.min_val() == 0, "MinMax min value");
    test_assert(scaler.max_val() == 100, "MinMax max value");

    auto scaled = scaler.transform(data);
    test_assert(std::abs(scaled[0] - 0.0) < 1e-10, "MinMax scaled min is 0");
    test_assert(std::abs(scaled[4] - 1.0) < 1e-10, "MinMax scaled max is 1");
    test_assert(std::abs(scaled[2] - 0.5) < 1e-10, "MinMax scaled middle is 0.5");

    // Test inverse transform
    auto restored = scaler.inverse_transform(scaled);
    test_assert(std::abs(restored[0] - data[0]) < 1e-10, "MinMax inverse at 0");
    test_assert(std::abs(restored[4] - data[4]) < 1e-10, "MinMax inverse at max");

    // Test fit_transform
    ts::MinMaxScaler scaler2;
    auto scaled2 = scaler2.fit_transform(data);
    test_assert(std::abs(scaled2[2] - 0.5) < 1e-10, "MinMax fit_transform");

    // Test multivariate scaling
    std::vector<std::vector<double>> mv_data = {
        {0, 0},
        {50, 100},
        {100, 200}
    };

    ts::MinMaxScaler mv_scaler;
    auto mv_scaled = mv_scaler.fit_transform(mv_data);

    test_assert(std::abs(mv_scaled[0][0] - 0.0) < 1e-10, "MV MinMax col1 min");
    test_assert(std::abs(mv_scaled[2][0] - 1.0) < 1e-10, "MV MinMax col1 max");
    test_assert(std::abs(mv_scaled[0][1] - 0.0) < 1e-10, "MV MinMax col2 min");
    test_assert(std::abs(mv_scaled[2][1] - 1.0) < 1e-10, "MV MinMax col2 max");

    // Test inverse transform multivariate
    auto mv_restored = mv_scaler.inverse_transform(mv_scaled);
    test_assert(std::abs(mv_restored[1][0] - 50) < 1e-10, "MV MinMax inverse");
    test_assert(std::abs(mv_restored[1][1] - 100) < 1e-10, "MV MinMax inverse col2");
}

void test_standard_scaler() {
    std::cout << "\n=== Testing Standard Scaler ===" << std::endl;

    // Test univariate standardization
    std::vector<double> data = {10, 20, 30, 40, 50};

    ts::StandardScaler scaler;
    scaler.fit(data);

    test_assert(scaler.is_fitted(), "Standard scaler fitted");
    test_assert(std::abs(scaler.mean() - 30.0) < 1e-10, "Standard scaler mean");

    auto scaled = scaler.transform(data);

    // Check mean is ~0 and std is ~1
    double scaled_mean = 0;
    for (double x : scaled) scaled_mean += x;
    scaled_mean /= scaled.size();
    test_assert(std::abs(scaled_mean) < 1e-10, "Standardized mean is 0");

    double scaled_var = 0;
    for (double x : scaled) scaled_var += (x - scaled_mean) * (x - scaled_mean);
    scaled_var /= (scaled.size() - 1);
    test_assert(std::abs(scaled_var - 1.0) < 1e-10, "Standardized variance is 1");

    // Test inverse transform
    auto restored = scaler.inverse_transform(scaled);
    for (size_t i = 0; i < data.size(); ++i) {
        test_assert(std::abs(restored[i] - data[i]) < 1e-10,
                    "Standard inverse transform at " + std::to_string(i));
    }

    // Test multivariate standardization
    std::vector<std::vector<double>> mv_data = {
        {10, 100},
        {20, 200},
        {30, 300},
        {40, 400},
        {50, 500}
    };

    ts::StandardScaler mv_scaler;
    auto mv_scaled = mv_scaler.fit_transform(mv_data);

    // Check each column has mean ~0
    double col1_mean = 0, col2_mean = 0;
    for (const auto& row : mv_scaled) {
        col1_mean += row[0];
        col2_mean += row[1];
    }
    col1_mean /= mv_scaled.size();
    col2_mean /= mv_scaled.size();

    test_assert(std::abs(col1_mean) < 1e-10, "MV Standard col1 mean is 0");
    test_assert(std::abs(col2_mean) < 1e-10, "MV Standard col2 mean is 0");

    // Test inverse transform
    auto mv_restored = mv_scaler.inverse_transform(mv_scaled);
    test_assert(std::abs(mv_restored[2][0] - 30) < 1e-10, "MV Standard inverse col1");
    test_assert(std::abs(mv_restored[2][1] - 300) < 1e-10, "MV Standard inverse col2");
}

void test_acf_pacf() {
    std::cout << "\n=== Testing ACF/PACF ===" << std::endl;

    std::vector<double> data(100);
    for (int i = 0; i < 100; ++i) {
        data[i] = std::sin(2 * M_PI * i / 10);
    }

    auto acf = ts::stats::acf(data, 20);
    test_assert(acf.size() == 21, "ACF size");
    test_assert(std::abs(acf[0] - 1.0) < 1e-10, "ACF at lag 0 is 1");
    test_assert(acf[10] > 0.9, "ACF periodic peak");

    auto pacf = ts::stats::pacf(data, 10);
    test_assert(pacf.size() == 11, "PACF size");
    test_assert(std::abs(pacf[0] - 1.0) < 1e-10, "PACF at lag 0 is 1");
}

int main() {
    std::cout << "Time Series Forecasting Library - Test Suite" << std::endl;
    std::cout << "=============================================" << std::endl;

    try {
        test_stats();
        test_matrix();
        test_arima();
        test_exponential_smoothing();
        test_prophet();
        test_gradient_boosting();
        test_evaluation();
        test_train_test_split();
        test_train_val_test_split();
        test_minmax_scaler();
        test_standard_scaler();
        test_acf_pacf();
    } catch (const std::exception& e) {
        std::cout << "\n[ERROR] Exception: " << e.what() << std::endl;
        tests_failed++;
    }

    std::cout << "\n=============================================" << std::endl;
    std::cout << "Tests passed: " << tests_passed << std::endl;
    std::cout << "Tests failed: " << tests_failed << std::endl;
    std::cout << "=============================================" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
