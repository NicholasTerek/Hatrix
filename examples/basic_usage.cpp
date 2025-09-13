//======================================================================
// basic_usage.cpp
//----------------------------------------------------------------------
// Basic usage examples for the Hadamard matrix library.
// Demonstrates core functionality with simple examples.
//
// LICENSE: MIT
// AUTHOR : Nicholas Terek
// VERSION: 1.0.0
//======================================================================

#include "../Hatrix/hadamard_matrix.hpp"
#include <iostream>
#include <vector>

int main() {
    std::cout << "Hadamard Matrix Library - Basic Usage Examples\n";
    std::cout << "==============================================\n\n";
    
    //--------------------------------------------------------------------------
    // Example 1: Generate and Display a Small Hadamard Matrix
    //--------------------------------------------------------------------------
    std::cout << "Example 1: Generate H(4)\n";
    std::cout << "-----------------------\n";
    
    auto H4 = hadamard::generate_recursive(4);
    hadamard::print(H4, std::cout, hadamard::format_t::VERBOSE);
    
    std::cout << "\nCompact format:\n";
    hadamard::print(H4, std::cout, hadamard::format_t::COMPACT);
    
    std::cout << "\nLaTeX format:\n";
    hadamard::print(H4, std::cout, hadamard::format_t::LATEX);
    
    //--------------------------------------------------------------------------
    // Example 2: Validate Hadamard Properties
    //--------------------------------------------------------------------------
    std::cout << "\n\nExample 2: Validation\n";
    std::cout << "--------------------\n";
    
    std::cout << "Is H(4) a valid Hadamard matrix? " 
              << (hadamard::is_hadamard(H4) ? "Yes" : "No") << "\n";
    
    std::cout << "Is H(4) orthogonal? " 
              << (hadamard::is_orthogonal(H4) ? "Yes" : "No") << "\n";
    
    // Check validation issues
    auto issues = hadamard::validate_matrix(H4);
    if (issues.empty()) {
        std::cout << "No validation issues found.\n";
    } else {
        std::cout << "Validation issues:\n";
        for (const auto& issue : issues) {
            std::cout << "  - " << issue << "\n";
        }
    }
    
    //--------------------------------------------------------------------------
    // Example 3: Matrix Operations
    //--------------------------------------------------------------------------
    std::cout << "\n\nExample 3: Matrix Operations\n";
    std::cout << "-----------------------------\n";
    
    // Transpose
    auto H4T = hadamard::transpose(H4);
    std::cout << "H(4)^T (transpose):\n";
    hadamard::print(H4T, std::cout, hadamard::format_t::COMPACT);
    
    // Matrix-vector multiplication
    hadamard::vector_t v = {1, 2, 3, 4};
    auto result = hadamard::multiply(H4, v);
    
    std::cout << "\nMatrix-vector multiplication H(4) * [1,2,3,4]:\n";
    std::cout << "[";
    for (size_t i = 0; i < result.size(); ++i) {
        std::cout << result[i];
        if (i + 1 < result.size()) std::cout << ", ";
    }
    std::cout << "]\n";
    
    // Matrix-matrix multiplication (should give n*I for Hadamard)
    auto H4H4T = hadamard::multiply(H4, H4T);
    std::cout << "\nH(4) * H(4)^T (should be 4*I):\n";
    hadamard::print(H4H4T, std::cout, hadamard::format_t::COMPACT);
    
    //--------------------------------------------------------------------------
    // Example 4: Walsh Orderings
    //--------------------------------------------------------------------------
    std::cout << "\n\nExample 4: Walsh Orderings\n";
    std::cout << "--------------------------\n";
    
    auto W4_natural = hadamard::generate_walsh(4, hadamard::ordering_t::NATURAL);
    auto W4_sequency = hadamard::generate_walsh(4, hadamard::ordering_t::SEQUENCY);
    auto W4_dyadic = hadamard::generate_walsh(4, hadamard::ordering_t::DYADIC);
    
    std::cout << "Natural ordering:\n";
    hadamard::print(W4_natural, std::cout, hadamard::format_t::COMPACT);
    
    std::cout << "\nSequency ordering:\n";
    hadamard::print(W4_sequency, std::cout, hadamard::format_t::COMPACT);
    
    std::cout << "\nDyadic ordering:\n";
    hadamard::print(W4_dyadic, std::cout, hadamard::format_t::COMPACT);
    
    //--------------------------------------------------------------------------
    // Example 5: Fast Walsh-Hadamard Transform
    //--------------------------------------------------------------------------
    std::cout << "\n\nExample 5: Fast Walsh-Hadamard Transform\n";
    std::cout << "----------------------------------------\n";
    
    hadamard::dvector_t signal = {1.0, 2.0, 3.0, 4.0};
    std::cout << "Original signal: [";
    for (size_t i = 0; i < signal.size(); ++i) {
        std::cout << signal[i];
        if (i + 1 < signal.size()) std::cout << ", ";
    }
    std::cout << "]\n";
    
    // Forward transform
    auto transformed = hadamard::fwht(signal);
    std::cout << "Transformed: [";
    for (size_t i = 0; i < transformed.size(); ++i) {
        std::cout << std::fixed << std::setprecision(2) << transformed[i];
        if (i + 1 < transformed.size()) std::cout << ", ";
    }
    std::cout << "]\n";
    
    // Inverse transform
    auto reconstructed = hadamard::ifwht(transformed);
    std::cout << "Reconstructed: [";
    for (size_t i = 0; i < reconstructed.size(); ++i) {
        std::cout << std::fixed << std::setprecision(2) << reconstructed[i];
        if (i + 1 < reconstructed.size()) std::cout << ", ";
    }
    std::cout << "]\n";
    
    //--------------------------------------------------------------------------
    // Example 6: Matrix Properties Analysis
    //--------------------------------------------------------------------------
    std::cout << "\n\nExample 6: Matrix Properties\n";
    std::cout << "----------------------------\n";
    
    auto props = hadamard::analyze_properties(H4);
    
    std::cout << "Matrix properties:\n";
    for (const auto& [key, value] : props) {
        std::cout << "  " << key << ": " << std::fixed << std::setprecision(3) << value << "\n";
    }
    
    //--------------------------------------------------------------------------
    // Example 7: Serialization
    //--------------------------------------------------------------------------
    std::cout << "\n\nExample 7: Serialization\n";
    std::cout << "------------------------\n";
    
    // Serialize to different formats
    std::string compact_data = hadamard::serialize(H4, hadamard::format_t::COMPACT);
    std::string csv_data = hadamard::serialize(H4, hadamard::format_t::CSV);
    
    std::cout << "Compact serialization:\n" << compact_data << "\n";
    std::cout << "CSV serialization:\n" << csv_data << "\n";
    
    // Deserialize and verify
    auto deserialized = hadamard::deserialize(compact_data, hadamard::format_t::COMPACT);
    std::cout << "Deserialized matrix matches original: " 
              << (deserialized == H4 ? "Yes" : "No") << "\n";
    
    //--------------------------------------------------------------------------
    // Example 8: File I/O
    //--------------------------------------------------------------------------
    std::cout << "\n\nExample 8: File I/O\n";
    std::cout << "-------------------\n";
    
    try {
        // Save to file
        hadamard::save_to_file(H4, "hadamard_4.txt", hadamard::format_t::COMPACT);
        std::cout << "Matrix saved to hadamard_4.txt\n";
        
        // Load from file
        auto loaded = hadamard::load_from_file("hadamard_4.txt", hadamard::format_t::COMPACT);
        std::cout << "Matrix loaded from file matches original: " 
                  << (loaded == H4 ? "Yes" : "No") << "\n";
        
        // Clean up
        std::remove("hadamard_4.txt");
        
    } catch (const std::exception& e) {
        std::cout << "File I/O error: " << e.what() << "\n";
    }
    
    //--------------------------------------------------------------------------
    // Example 9: Performance Benchmarking
    //--------------------------------------------------------------------------
    std::cout << "\n\nExample 9: Performance Benchmarking\n";
    std::cout << "------------------------------------\n";
    
    // Benchmark matrix generation
    double gen_time = hadamard::benchmark([]() {
        auto H = hadamard::generate_recursive(16);
        volatile auto _ = hadamard::is_hadamard(H);  // Prevent optimization
    }, 100);
    
    std::cout << "Generation time (H(16), 100 iterations): " 
              << std::fixed << std::setprecision(2) << gen_time << " μs\n";
    
    // Benchmark transform
    hadamard::dvector_t test_signal(16);
    std::iota(test_signal.begin(), test_signal.end(), 1.0);
    
    double fwht_time = hadamard::benchmark([test_signal]() mutable {
        hadamard::fwht(test_signal);
    }, 100);
    
    std::cout << "FWHT time (size 16, 100 iterations): " 
              << std::fixed << std::setprecision(2) << fwht_time << " μs\n";
    
    std::cout << "\nBasic usage examples completed!\n";
    
    return 0;
}
