#include <iostream>
#include <vector>
#include <cmath>
#include "ts_forecast.hpp"

int main() {
    std::cout << "=== Prophet-like Model Example ===" << std::endl;

    // Generate daily data with trend, weekly and yearly seasonality
    std::vector<double> timestamps;
    std::vector<double> values;

    for (int day = 0; day < 730; ++day) {  // 2 years of daily data
        double t = static_cast<double>(day);

        // Trend
        double trend = 100.0 + 0.05 * day;

        // Weekly seasonality (period = 7)
        double weekly = 10.0 * std::sin(2 * M_PI * day / 7.0);

        // Yearly seasonality (period = 365.25)
        double yearly = 20.0 * std::sin(2 * M_PI * day / 365.25);

        // Noise
        double noise = 5.0 * (std::rand() % 100 - 50) / 50.0;

        timestamps.push_back(t);
        values.push_back(trend + weekly + yearly + noise);
    }

    std::cout << "Generated 2 years of daily data (730 points)" << std::endl;
    std::cout << "Data includes trend, weekly, and yearly seasonality" << std::endl;

    // Fit Prophet model
    std::cout << "\n--- Fitting Prophet Model ---" << std::endl;
    ts::Prophet prophet(ts::Prophet::GrowthType::LINEAR, true, true, false);
    prophet.set_changepoints(10, 0.8);
    prophet.fit(timestamps, values);

    std::cout << "Model fitted successfully!" << std::endl;

    // Get components
    std::cout << "\n--- Decomposition (sample) ---" << std::endl;
    std::vector<double> sample_times = {0, 100, 200, 365, 500, 700};
    std::vector<double> trend = prophet.get_trend(sample_times);
    std::vector<double> seasonal = prophet.get_seasonality(sample_times);

    std::cout << "Day\tTrend\tSeasonality" << std::endl;
    for (size_t i = 0; i < sample_times.size(); ++i) {
        std::cout << sample_times[i] << "\t"
                  << trend[i] << "\t"
                  << seasonal[i] << std::endl;
    }

    // Forecast
    std::cout << "\n--- Forecast 30 days ahead ---" << std::endl;
    ts::ForecastResult forecast = prophet.forecast(30);

    std::cout << "Day\tPrediction\tLower\tUpper" << std::endl;
    for (int i = 0; i < 30; i += 5) {
        std::cout << (731 + i) << "\t"
                  << forecast.predictions[i] << "\t"
                  << forecast.lower_bound[i] << "\t"
                  << forecast.upper_bound[i] << std::endl;
    }

    // Get changepoints
    std::cout << "\n--- Detected Changepoints ---" << std::endl;
    auto changepoints = prophet.get_changepoints();
    std::cout << "Number of changepoints: " << changepoints.size() << std::endl;
    for (size_t i = 0; i < std::min(changepoints.size(), size_t(5)); ++i) {
        std::cout << "  Day " << changepoints[i].timestamp
                  << ": rate change = " << changepoints[i].rate_change << std::endl;
    }

    // Add custom seasonality
    std::cout << "\n--- Prophet with Custom Seasonality ---" << std::endl;
    ts::Prophet prophet2;
    prophet2.add_seasonality("monthly", 30.5, 5);
    prophet2.fit(values);  // Using simple interface

    ts::ForecastResult forecast2 = prophet2.forecast(14);
    std::cout << "2-week forecast with monthly seasonality:" << std::endl;
    for (int i = 0; i < 14; ++i) {
        std::cout << "  Day " << (i + 1) << ": " << forecast2.predictions[i] << std::endl;
    }

    // Prophet with holidays
    std::cout << "\n--- Prophet with Holidays ---" << std::endl;
    ts::Prophet prophet3;

    // Add some holidays (e.g., New Year, Christmas)
    ts::Prophet::Holiday new_year("NewYear", {0, 365}, -1, 1);  // Day 0 and 365
    ts::Prophet::Holiday christmas("Christmas", {358, 723}, -2, 0);  // Dec 25

    prophet3.add_holiday(new_year);
    prophet3.add_holiday(christmas);

    prophet3.fit(timestamps, values);
    std::cout << "Model with holidays fitted!" << std::endl;

    // Evaluate model
    std::cout << "\n--- Model Evaluation ---" << std::endl;
    auto [train_ts, test_ts] = [&timestamps]() {
        size_t split = static_cast<size_t>(timestamps.size() * 0.8);
        return std::make_pair(
            std::vector<double>(timestamps.begin(), timestamps.begin() + split),
            std::vector<double>(timestamps.begin() + split, timestamps.end())
        );
    }();

    auto [train_vals, test_vals] = [&values]() {
        size_t split = static_cast<size_t>(values.size() * 0.8);
        return std::make_pair(
            std::vector<double>(values.begin(), values.begin() + split),
            std::vector<double>(values.begin() + split, values.end())
        );
    }();

    ts::Prophet prophet_eval;
    prophet_eval.fit(train_ts, train_vals);

    ts::ForecastResult eval_result = prophet_eval.predict(test_ts);
    ts::Metrics metrics = ts::evaluate(test_vals, eval_result.predictions);

    std::cout << "Out-of-sample metrics:" << std::endl;
    std::cout << metrics.to_string() << std::endl;

    // Neural Prophet-like model
    std::cout << "\n=== NeuralProphet-like Model ===" << std::endl;
    ts::NeuralProphet neural_prophet(14, 1, true, true);
    neural_prophet.fit(values, 200, 0.001);

    ts::ForecastResult np_forecast = neural_prophet.forecast(7);
    std::cout << "7-day forecast:" << std::endl;
    for (int i = 0; i < 7; ++i) {
        std::cout << "  Day " << (i + 1) << ": " << np_forecast.predictions[i] << std::endl;
    }

    return 0;
}
