//======================================================================
// advanced_usage.cpp
//----------------------------------------------------------------------
// Advanced usage examples for the Hadamard matrix library.
// Demonstrates complex scenarios and optimization techniques.
//
// LICENSE: MIT
// AUTHOR : Nicholas Terek
// VERSION: 1.0.0
//======================================================================

#include "../Hatrix/hadamard_matrix.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

int main() {
    std::cout << "Hadamard Matrix Library - Advanced Usage Examples\n";
    std::cout << "==================================================\n\n";
    
    //--------------------------------------------------------------------------
    // Example 1: Large Matrix Generation and Memory Analysis
    //--------------------------------------------------------------------------
    std::cout << "Example 1: Large Matrix Generation\n";
    std::cout << "-----------------------------------\n";
    
    std::vector<int> sizes = {64, 128, 256, 512};
    
    for (int size : sizes) {
        std::cout << "\nGenerating H(" << size << ")...\n";
        
        auto start = std::chrono::high_resolution_clock::now();
        auto H = hadamard::generate_iterative(size);  // Use iterative for large sizes
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        size_t memory_bytes = size * size * sizeof(int);
        double memory_mb = memory_bytes / (1024.0 * 1024.0);
        
        std::cout << "  Generation time: " << duration.count() << " ms\n";
        std::cout << "  Memory usage: " << std::fixed << std::setprecision(2) << memory_mb << " MB\n";
        std::cout << "  Is valid Hadamard: " << (hadamard::is_hadamard(H) ? "Yes" : "No") << "\n";
    }
    
    //--------------------------------------------------------------------------
    // Example 2: Signal Processing with FWHT
    //--------------------------------------------------------------------------
    std::cout << "\n\nExample 2: Signal Processing\n";
    std::cout << "-----------------------------\n";
    
    // Generate a test signal with noise
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> noise(-0.5, 0.5);
    
    int signal_size = 32;
    hadamard::dvector_t clean_signal(signal_size);
    hadamard::dvector_t noisy_signal(signal_size);
    
    // Create a signal: sine wave + noise
    for (int i = 0; i < signal_size; ++i) {
        clean_signal[i] = std::sin(2.0 * M_PI * i / signal_size) + 0.5 * std::sin(4.0 * M_PI * i / signal_size);
        noisy_signal[i] = clean_signal[i] + noise(rng);
    }
    
    std::cout << "Signal size: " << signal_size << "\n";
    std::cout << "Clean signal (first 8 samples): [";
    for (int i = 0; i < 8; ++i) {
        std::cout << std::fixed << std::setprecision(3) << clean_signal[i];
        if (i < 7) std::cout << ", ";
    }
    std::cout << "...]\n";
    
    // Apply FWHT
    auto clean_fwht = hadamard::fwht(clean_signal);
    auto noisy_fwht = hadamard::fwht(noisy_signal);
    
    // Find significant coefficients
    std::vector<int> significant_indices;
    double threshold = 0.1;
    
    for (int i = 0; i < signal_size; ++i) {
        if (std::abs(clean_fwht[i]) > threshold) {
            significant_indices.push_back(i);
        }
    }
    
    std::cout << "Significant coefficients (|coeff| > " << threshold << "): " 
              << significant_indices.size() << " out of " << signal_size << "\n";
    
    // Noise filtering: zero out small coefficients
    auto filtered_fwht = noisy_fwht;
    int filtered_count = 0;
    for (int i = 0; i < signal_size; ++i) {
        if (std::abs(filtered_fwht[i]) < threshold) {
            filtered_fwht[i] = 0.0;
            filtered_count++;
        }
    }
    
    auto filtered_signal = hadamard::ifwht(filtered_fwht);
    
    std::cout << "Filtered coefficients: " << filtered_count << " zeroed out\n";
    
    // Calculate SNR improvement
    double original_noise_power = 0.0, filtered_noise_power = 0.0;
    for (int i = 0; i < signal_size; ++i) {
        double noise_component = noisy_signal[i] - clean_signal[i];
        double filtered_noise = filtered_signal[i] - clean_signal[i];
        original_noise_power += noise_component * noise_component;
        filtered_noise_power += filtered_noise * filtered_noise;
    }
    
    double snr_improvement = 10.0 * std::log10(original_noise_power / filtered_noise_power);
    std::cout << "SNR improvement: " << std::fixed << std::setprecision(2) << snr_improvement << " dB\n";
    
    //--------------------------------------------------------------------------
    // Example 3: Matrix Decomposition and Reconstruction
    //--------------------------------------------------------------------------
    std::cout << "\n\nExample 3: Matrix Decomposition\n";
    std::cout << "--------------------------------\n";
    
    auto H8 = hadamard::generate_recursive(8);
    
    // Decompose into submatrices
    int half = 4;
    hadamard::matrix_t H8_00(half, hadamard::vector_t(half));
    hadamard::matrix_t H8_01(half, hadamard::vector_t(half));
    hadamard::matrix_t H8_10(half, hadamard::vector_t(half));
    hadamard::matrix_t H8_11(half, hadamard::vector_t(half));
    
    for (int i = 0; i < half; ++i) {
        for (int j = 0; j < half; ++j) {
            H8_00[i][j] = H8[i][j];
            H8_01[i][j] = H8[i][j + half];
            H8_10[i][j] = H8[i + half][j];
            H8_11[i][j] = H8[i + half][j + half];
        }
    }
    
    std::cout << "H(8) decomposed into 4 submatrices of size 4x4\n";
    std::cout << "H(8)[0:4,0:4] (top-left):\n";
    hadamard::print(H8_00, std::cout, hadamard::format_t::COMPACT);
    
    // Verify Sylvester construction
    std::cout << "Verifying Sylvester construction H(8)[0:4,0:4] == H(8)[0:4,4:8]: " 
              << (H8_00 == H8_01 ? "Yes" : "No") << "\n";
    
    std::cout << "Verifying Sylvester construction H(8)[0:4,0:4] == H(8)[4:8,0:4]: " 
              << (H8_00 == H8_10 ? "Yes" : "No") << "\n";
    
    std::cout << "Verifying Sylvester construction H(8)[4:8,4:8] == -H(8)[0:4,0:4]: " 
              << (H8_11 == H8_00 ? "No (should be negative)" : "Yes") << "\n";
    
    //--------------------------------------------------------------------------
    // Example 4: Custom Matrix Validation
    //--------------------------------------------------------------------------
    std::cout << "\n\nExample 4: Custom Matrix Validation\n";
    std::cout << "------------------------------------\n";
    
    // Create a custom matrix that's almost Hadamard
    hadamard::matrix_t custom = {{1, 1, 1, 1},
                                 {1, -1, 1, -1},
                                 {1, 1, -1, -1},
                                 {1, -1, -1, 1}};  // This is actually valid H(4)
    
    std::cout << "Custom matrix:\n";
    hadamard::print(custom, std::cout, hadamard::format_t::COMPACT);
    
    auto issues = hadamard::validate_matrix(custom);
    if (issues.empty()) {
        std::cout << "Custom matrix is valid!\n";
    } else {
        std::cout << "Validation issues:\n";
        for (const auto& issue : issues) {
            std::cout << "  - " << issue << "\n";
        }
    }
    
    // Test with invalid matrix
    hadamard::matrix_t invalid = {{1, 1, 1, 1},
                                  {1, -1, 1, -1},
                                  {1, 1, -1, -1},
                                  {1, -1, -1, 2}};  // Invalid: contains 2
    
    std::cout << "\nInvalid matrix (contains 2):\n";
    auto invalid_issues = hadamard::validate_matrix(invalid);
    std::cout << "Validation issues:\n";
    for (const auto& issue : invalid_issues) {
        std::cout << "  - " << issue << "\n";
    }
    
    //--------------------------------------------------------------------------
    // Example 5: Performance Comparison
    //--------------------------------------------------------------------------
    std::cout << "\n\nExample 5: Performance Comparison\n";
    std::cout << "-----------------------------------\n";
    
    std::vector<int> test_sizes = {8, 16, 32, 64, 128};
    
    std::cout << "Size\tRecursive(ms)\tIterative(ms)\tRatio\n";
    std::cout << std::string(50, '-') << "\n";
    
    for (int size : test_sizes) {
        // Benchmark recursive
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; ++i) {
            auto H_rec = hadamard::generate_recursive(size);
            volatile auto _ = H_rec.size();  // Prevent optimization
        }
        auto end = std::chrono::high_resolution_clock::now();
        double time_rec = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 10000.0;
        
        // Benchmark iterative
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10; ++i) {
            auto H_iter = hadamard::generate_iterative(size);
            volatile auto _ = H_iter.size();  // Prevent optimization
        }
        end = std::chrono::high_resolution_clock::now();
        double time_iter = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 10000.0;
        
        double ratio = time_rec / time_iter;
        
        std::cout << size << "\t" 
                  << std::fixed << std::setprecision(2) << time_rec << "\t"
                  << std::fixed << std::setprecision(2) << time_iter << "\t"
                  << std::fixed << std::setprecision(2) << ratio << "\n";
    }
    
    //--------------------------------------------------------------------------
    // Example 6: Batch Processing
    //--------------------------------------------------------------------------
    std::cout << "\n\nExample 6: Batch Processing\n";
    std::cout << "----------------------------\n";
    
    // Process multiple signals in batch
    int batch_size = 10;
    int signal_length = 16;
    
    std::vector<hadamard::dvector_t> signals(batch_size);
    std::vector<hadamard::dvector_t> transforms(batch_size);
    
    // Generate random signals
    std::uniform_real_distribution<double> signal_dist(-1.0, 1.0);
    for (int b = 0; b < batch_size; ++b) {
        signals[b].resize(signal_length);
        for (int i = 0; i < signal_length; ++i) {
            signals[b][i] = signal_dist(rng);
        }
    }
    
    // Process batch
    auto batch_start = std::chrono::high_resolution_clock::now();
    for (int b = 0; b < batch_size; ++b) {
        transforms[b] = hadamard::fwht(signals[b]);
    }
    auto batch_end = std::chrono::high_resolution_clock::now();
    
    auto batch_duration = std::chrono::duration_cast<std::chrono::microseconds>(batch_end - batch_start);
    double avg_time_per_signal = batch_duration.count() / static_cast<double>(batch_size);
    
    std::cout << "Processed " << batch_size << " signals of length " << signal_length << "\n";
    std::cout << "Total time: " << batch_duration.count() << " μs\n";
    std::cout << "Average time per signal: " << std::fixed << std::setprecision(2) << avg_time_per_signal << " μs\n";
    
    // Verify round-trip for all signals
    int correct_roundtrips = 0;
    for (int b = 0; b < batch_size; ++b) {
        auto reconstructed = hadamard::ifwht(transforms[b]);
        bool is_correct = true;
        for (int i = 0; i < signal_length; ++i) {
            if (std::abs(reconstructed[i] - signals[b][i]) > 1e-10) {
                is_correct = false;
                break;
            }
        }
        if (is_correct) correct_roundtrips++;
    }
    
    std::cout << "Correct round-trips: " << correct_roundtrips << "/" << batch_size << "\n";
    
    std::cout << "\nAdvanced usage examples completed!\n";
    
    return 0;
}
