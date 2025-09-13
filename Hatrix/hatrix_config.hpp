//======================================================================
// hatrix_config.hpp
//----------------------------------------------------------------------
// High-quality configuration system for the Hatrix library using
// Boost.Program_options and modern C++ configuration management.
//
// LICENSE: MIT
// AUTHOR : Nicholas Terek
// VERSION: 1.0.0
//======================================================================
#ifndef HATRIX_CONFIG_HPP
#define HATRIX_CONFIG_HPP

#include <boost/program_options.hpp>
#include <boost/any.hpp>
#include <boost/variant.hpp>
#include <boost/optional.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/filesystem.hpp>
#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>

namespace hatrix {
namespace config {

//--------------------------------------------------------------------------
// CONFIGURATION VALUE TYPES
//--------------------------------------------------------------------------
using ConfigValue = boost::variant<
    bool,
    int,
    double,
    std::string,
    std::vector<std::string>
>;

//--------------------------------------------------------------------------
// PERFORMANCE CONFIGURATION
//--------------------------------------------------------------------------
struct PerformanceConfig {
    // SIMD settings
    bool enable_simd = true;
    bool enable_avx2 = true;
    bool enable_avx512 = true;
    bool enable_fma = true;
    
    // Threading settings
    int num_threads = -1; // -1 means auto-detect
    bool enable_parallel = true;
    int thread_affinity = 0; // 0=no affinity, 1=compact, 2=scatter
    
    // Memory settings
    std::size_t cache_line_size = 64;
    std::size_t l1_cache_size = 32768;      // 32KB
    std::size_t l2_cache_size = 262144;     // 256KB
    std::size_t l3_cache_size = 8388608;    // 8MB
    bool use_aligned_alloc = true;
    std::size_t alignment = 64;
    
    // Optimization settings
    bool enable_prefetch = true;
    bool enable_vectorization = true;
    bool enable_loop_unrolling = true;
    int unroll_factor = 4;
    
    // Serialization support
    template<class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar & BOOST_SERIALIZATION_NVP(enable_simd);
        ar & BOOST_SERIALIZATION_NVP(enable_avx2);
        ar & BOOST_SERIALIZATION_NVP(enable_avx512);
        ar & BOOST_SERIALIZATION_NVP(enable_fma);
        ar & BOOST_SERIALIZATION_NVP(num_threads);
        ar & BOOST_SERIALIZATION_NVP(enable_parallel);
        ar & BOOST_SERIALIZATION_NVP(thread_affinity);
        ar & BOOST_SERIALIZATION_NVP(cache_line_size);
        ar & BOOST_SERIALIZATION_NVP(l1_cache_size);
        ar & BOOST_SERIALIZATION_NVP(l2_cache_size);
        ar & BOOST_SERIALIZATION_NVP(l3_cache_size);
        ar & BOOST_SERIALIZATION_NVP(use_aligned_alloc);
        ar & BOOST_SERIALIZATION_NVP(alignment);
        ar & BOOST_SERIALIZATION_NVP(enable_prefetch);
        ar & BOOST_SERIALIZATION_NVP(enable_vectorization);
        ar & BOOST_SERIALIZATION_NVP(enable_loop_unrolling);
        ar & BOOST_SERIALIZATION_NVP(unroll_factor);
    }
};

//--------------------------------------------------------------------------
// LOGGING CONFIGURATION
//--------------------------------------------------------------------------
struct LoggingConfig {
    std::string level = "info";
    bool console_output = true;
    bool file_output = true;
    std::string log_directory = "logs";
    std::string log_filename = "hatrix.log";
    std::size_t max_file_size = 10485760; // 10MB
    std::size_t max_files = 5;
    std::string pattern = "[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%t] [%s:%#] %v";
    bool async_logging = true;
    std::size_t queue_size = 8192;
    
    template<class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar & BOOST_SERIALIZATION_NVP(level);
        ar & BOOST_SERIALIZATION_NVP(console_output);
        ar & BOOST_SERIALIZATION_NVP(file_output);
        ar & BOOST_SERIALIZATION_NVP(log_directory);
        ar & BOOST_SERIALIZATION_NVP(log_filename);
        ar & BOOST_SERIALIZATION_NVP(max_file_size);
        ar & BOOST_SERIALIZATION_NVP(max_files);
        ar & BOOST_SERIALIZATION_NVP(pattern);
        ar & BOOST_SERIALIZATION_NVP(async_logging);
        ar & BOOST_SERIALIZATION_NVP(queue_size);
    }
};

//--------------------------------------------------------------------------
// ALGORITHM CONFIGURATION
//--------------------------------------------------------------------------
struct AlgorithmConfig {
    // Matrix generation settings
    std::string default_generation_method = "recursive";
    bool enable_iterative_generation = true;
    bool enable_recursive_generation = true;
    bool enable_simd_generation = true;
    
    // FWHT settings
    bool enable_inplace_transform = true;
    bool enable_batch_processing = true;
    std::size_t batch_size_threshold = 1024;
    
    // GEMM settings
    std::size_t gemm_tile_size = 256;
    std::size_t gemm_micro_kernel_size = 8;
    bool enable_gemm_optimization = true;
    
    // Validation settings
    bool enable_matrix_validation = true;
    bool enable_orthogonality_check = true;
    double tolerance = 1e-12;
    
    template<class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar & BOOST_SERIALIZATION_NVP(default_generation_method);
        ar & BOOST_SERIALIZATION_NVP(enable_iterative_generation);
        ar & BOOST_SERIALIZATION_NVP(enable_recursive_generation);
        ar & BOOST_SERIALIZATION_NVP(enable_simd_generation);
        ar & BOOST_SERIALIZATION_NVP(enable_inplace_transform);
        ar & BOOST_SERIALIZATION_NVP(enable_batch_processing);
        ar & BOOST_SERIALIZATION_NVP(batch_size_threshold);
        ar & BOOST_SERIALIZATION_NVP(gemm_tile_size);
        ar & BOOST_SERIALIZATION_NVP(gemm_micro_kernel_size);
        ar & BOOST_SERIALIZATION_NVP(enable_gemm_optimization);
        ar & BOOST_SERIALIZATION_NVP(enable_matrix_validation);
        ar & BOOST_SERIALIZATION_NVP(enable_orthogonality_check);
        ar & BOOST_SERIALIZATION_NVP(tolerance);
    }
};

//--------------------------------------------------------------------------
// MAIN CONFIGURATION CLASS
//--------------------------------------------------------------------------
class Configuration {
public:
    PerformanceConfig performance;
    LoggingConfig logging;
    AlgorithmConfig algorithm;
    
    // Additional custom settings
    std::map<std::string, ConfigValue> custom_settings;
    
    Configuration() = default;
    
    // Serialization support
    template<class Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar & BOOST_SERIALIZATION_NVP(performance);
        ar & BOOST_SERIALIZATION_NVP(logging);
        ar & BOOST_SERIALIZATION_NVP(algorithm);
        ar & BOOST_SERIALIZATION_NVP(custom_settings);
    }
    
    // Validation
    bool validate() const;
    
    // Get/set custom values
    template<typename T>
    boost::optional<T> get_custom(const std::string& key) const {
        auto it = custom_settings.find(key);
        if (it != custom_settings.end()) {
            try {
                return boost::get<T>(it->second);
            } catch (const boost::bad_get&) {
                return boost::none;
            }
        }
        return boost::none;
    }
    
    template<typename T>
    void set_custom(const std::string& key, const T& value) {
        custom_settings[key] = value;
    }
    
    // Save/load configuration
    void save_to_file(const std::string& filename) const;
    void load_from_file(const std::string& filename);
    void save_to_stream(std::ostream& os) const;
    void load_from_stream(std::istream& is);
};

//--------------------------------------------------------------------------
// CONFIGURATION MANAGER
//--------------------------------------------------------------------------
class ConfigurationManager {
public:
    static ConfigurationManager& instance() {
        static ConfigurationManager instance;
        return instance;
    }
    
    const Configuration& get_config() const { return config_; }
    Configuration& get_config() { return config_; }
    
    void set_config(const Configuration& config) { config_ = config; }
    
    // Load configuration from multiple sources
    void load_from_command_line(int argc, char* argv[]);
    void load_from_file(const std::string& filename);
    void load_from_environment();
    void load_defaults();
    
    // Save configuration
    void save_to_file(const std::string& filename) const;
    void save_to_default_location() const;
    
    // Validation and error reporting
    bool validate_config() const;
    std::vector<std::string> get_validation_errors() const;
    
    // Configuration presets
    void apply_debug_preset();
    void apply_release_preset();
    void apply_production_preset();
    void apply_benchmark_preset();
    
private:
    ConfigurationManager() = default;
    
    Configuration config_;
    boost::program_options::options_description desc_;
    
    void setup_options_description();
    void apply_environment_overrides();
};

//--------------------------------------------------------------------------
// CONVENIENCE FUNCTIONS
//--------------------------------------------------------------------------
inline ConfigurationManager& get_config_manager() {
    return ConfigurationManager::instance();
}

inline const Configuration& get_config() {
    return ConfigurationManager::instance().get_config();
}

inline void load_config_from_file(const std::string& filename) {
    ConfigurationManager::instance().load_from_file(filename);
}

inline void save_config_to_file(const std::string& filename) {
    ConfigurationManager::instance().save_to_file(filename);
}

//--------------------------------------------------------------------------
// CONFIGURATION PRESETS
//--------------------------------------------------------------------------
namespace presets {
    inline Configuration debug() {
        Configuration config;
        config.performance.enable_simd = true;
        config.performance.enable_parallel = true;
        config.performance.num_threads = 1; // Single thread for debugging
        config.logging.level = "debug";
        config.logging.console_output = true;
        config.algorithm.enable_matrix_validation = true;
        config.algorithm.tolerance = 1e-15;
        return config;
    }
    
    inline Configuration release() {
        Configuration config;
        config.performance.enable_simd = true;
        config.performance.enable_parallel = true;
        config.performance.num_threads = -1; // Auto-detect
        config.logging.level = "info";
        config.logging.console_output = false;
        config.logging.file_output = true;
        config.algorithm.enable_matrix_validation = true;
        return config;
    }
    
    inline Configuration production() {
        Configuration config;
        config.performance.enable_simd = true;
        config.performance.enable_parallel = true;
        config.performance.num_threads = -1;
        config.logging.level = "warn";
        config.logging.console_output = false;
        config.logging.file_output = true;
        config.logging.max_file_size = 104857600; // 100MB
        config.logging.max_files = 10;
        config.algorithm.enable_matrix_validation = false; // Disable for performance
        return config;
    }
    
    inline Configuration benchmark() {
        Configuration config;
        config.performance.enable_simd = true;
        config.performance.enable_parallel = true;
        config.performance.num_threads = -1;
        config.logging.level = "error";
        config.logging.console_output = false;
        config.logging.file_output = false;
        config.algorithm.enable_matrix_validation = false;
        config.algorithm.enable_orthogonality_check = false;
        return config;
    }
}

} // namespace config
} // namespace hatrix

#endif // HATRIX_CONFIG_HPP
