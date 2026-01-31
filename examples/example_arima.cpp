#include <iostream>
#include <vector>
#include <cmath>
#include "ts_forecast.hpp"

int main() {
    std::cout << "=== ARIMA Example ===" << std::endl;

    // Generate sample time series data with trend and noise
    std::vector<double> data;
    for (int i = 0; i < 100; ++i) {
        double trend = 50.0 + 0.5 * i;
        double noise = 5.0 * std::sin(i * 0.2) + (std::rand() % 10 - 5);
        data.push_back(trend + noise);
    }

    std::cout << "Generated 100 data points with trend and noise" << std::endl;
    std::cout << "First 10 values: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;

    // Fit ARIMA model
    std::cout << "\nFitting ARIMA(2,1,1) model..." << std::endl;
    ts::ARIMA model(2, 1, 1);
    model.fit(data);

    std::cout << "Model fitted successfully!" << std::endl;
    std::cout << "AR coefficients: ";
    for (double phi : model.ar_coefficients()) {
        std::cout << phi << " ";
    }
    std::cout << std::endl;

    std::cout << "MA coefficients: ";
    for (double theta : model.ma_coefficients()) {
        std::cout << theta << " ";
    }
    std::cout << std::endl;

    std::cout << "Sigma: " << model.sigma() << std::endl;
    std::cout << "AIC: " << model.aic() << std::endl;
    std::cout << "BIC: " << model.bic() << std::endl;

    // Forecast
    std::cout << "\nForecasting 10 steps ahead..." << std::endl;
    ts::ForecastResult forecast = model.forecast(10);

    std::cout << "Forecasts with 95% confidence intervals:" << std::endl;
    for (size_t i = 0; i < forecast.predictions.size(); ++i) {
        std::cout << "  Step " << (i + 1) << ": "
                  << forecast.predictions[i]
                  << " [" << forecast.lower_bound[i]
                  << ", " << forecast.upper_bound[i] << "]"
                  << std::endl;
    }

    // Evaluate model on fitted values
    std::vector<double> fitted = model.fitted_values();
    if (!fitted.empty()) {
        // Calculate in-sample metrics
        std::vector<double> actual(data.begin() + model.d(), data.end());
        std::vector<double> predicted(fitted.begin() + model.d(), fitted.end());

        if (actual.size() == predicted.size() && !actual.empty()) {
            ts::Metrics metrics = ts::evaluate(actual, predicted);
            std::cout << "\nIn-sample evaluation metrics:" << std::endl;
            std::cout << metrics.to_string() << std::endl;
        }
    }

    // Auto ARIMA
    std::cout << "\n=== Auto ARIMA ===" << std::endl;
    ts::AutoARIMA auto_model(3, 2, 3);
    auto_model.fit(data);

    std::cout << "Selected model: ARIMA("
              << auto_model.selected_p() << ","
              << auto_model.selected_d() << ","
              << auto_model.selected_q() << ")" << std::endl;
    std::cout << "Best AIC: " << auto_model.best_aic() << std::endl;

    ts::ForecastResult auto_forecast = auto_model.forecast(5);
    std::cout << "Auto ARIMA forecasts:" << std::endl;
    for (size_t i = 0; i < auto_forecast.predictions.size(); ++i) {
        std::cout << "  Step " << (i + 1) << ": " << auto_forecast.predictions[i] << std::endl;
    }

    return 0;
}
