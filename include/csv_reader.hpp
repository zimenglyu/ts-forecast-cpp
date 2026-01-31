#ifndef TS_CSV_READER_HPP
#define TS_CSV_READER_HPP

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <stdexcept>

namespace ts {

/**
 * Simple CSV Reader for time series data
 */
class CSVReader {
public:
    struct Dataset {
        std::vector<std::string> headers;
        std::vector<std::vector<double>> data;  // rows x cols
        std::map<std::string, size_t> column_index;

        std::vector<double> get_column(const std::string& name) const {
            auto it = column_index.find(name);
            if (it == column_index.end()) {
                throw std::runtime_error("Column not found: " + name);
            }
            size_t idx = it->second;
            std::vector<double> col;
            col.reserve(data.size());
            for (const auto& row : data) {
                col.push_back(row[idx]);
            }
            return col;
        }

        std::vector<std::vector<double>> get_columns(const std::vector<std::string>& names) const {
            std::vector<std::vector<double>> result;
            result.reserve(data.size());

            std::vector<size_t> indices;
            for (const auto& name : names) {
                auto it = column_index.find(name);
                if (it == column_index.end()) {
                    throw std::runtime_error("Column not found: " + name);
                }
                indices.push_back(it->second);
            }

            for (const auto& row : data) {
                std::vector<double> new_row;
                new_row.reserve(indices.size());
                for (size_t idx : indices) {
                    new_row.push_back(row[idx]);
                }
                result.push_back(new_row);
            }
            return result;
        }

        size_t num_rows() const { return data.size(); }
        size_t num_cols() const { return headers.size(); }
    };

    static Dataset read(const std::string& filename, bool skip_first_col = false) {
        Dataset dataset;
        std::ifstream file(filename);

        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        std::string line;

        // Read header
        if (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            size_t col_idx = 0;
            bool first = true;

            while (std::getline(ss, cell, ',')) {
                if (first && skip_first_col) {
                    first = false;
                    continue;
                }
                first = false;
                // Remove quotes and whitespace
                cell.erase(0, cell.find_first_not_of(" \t\r\n\""));
                cell.erase(cell.find_last_not_of(" \t\r\n\"") + 1);
                dataset.headers.push_back(cell);
                dataset.column_index[cell] = col_idx++;
            }
        }

        // Read data
        while (std::getline(file, line)) {
            if (line.empty()) continue;

            std::stringstream ss(line);
            std::string cell;
            std::vector<double> row;
            bool first = true;

            while (std::getline(ss, cell, ',')) {
                if (first && skip_first_col) {
                    first = false;
                    continue;
                }
                first = false;

                cell.erase(0, cell.find_first_not_of(" \t\r\n\""));
                cell.erase(cell.find_last_not_of(" \t\r\n\"") + 1);

                try {
                    row.push_back(std::stod(cell));
                } catch (...) {
                    row.push_back(0.0);  // Handle non-numeric values
                }
            }

            if (row.size() == dataset.headers.size()) {
                dataset.data.push_back(row);
            }
        }

        return dataset;
    }
};

} // namespace ts

#endif // TS_CSV_READER_HPP
