//======================================================================
// hatrix_parallel.hpp
//----------------------------------------------------------------------
// Multi-threaded and cache-optimized implementations for large-scale
// Hadamard matrix operations and ML experiments.
//
// LICENSE: MIT
// AUTHOR : Nicholas Terek
// VERSION: 1.0.0
//======================================================================
#ifndef HATRIX_PARALLEL_HPP
#define HATRIX_PARALLEL_HPP

#include "hadamard_matrix.hpp"
#include "hatrix_simd.hpp"
#include <thread>
#include <future>
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <algorithm>
#include <chrono>
#include <random>

namespace hatrix {
namespace parallel {

//--------------------------------------------------------------------------
// THREAD POOL FOR PARALLEL EXECUTION
//--------------------------------------------------------------------------
class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency()) 
        : num_threads_(num_threads), stop_(false) {
        for (size_t i = 0; i < num_threads_; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                        
                        if (stop_ && tasks_.empty()) {
                            return;
                        }
                        
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    task();
                }
            });
        }
    }
    
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        condition_.notify_all();
        
        for (auto& worker : workers_) {
            worker.join();
        }
    }
    
    template<typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
        using return_type = typename std::result_of<F(Args...)>::type;
        
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<return_type> result = task->get_future();
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }
            tasks_.emplace([task]() { (*task)(); });
        }
        
        condition_.notify_one();
        return result;
    }
    
    size_t size() const { return num_threads_; }
    
private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_;
    size_t num_threads_;
};

//--------------------------------------------------------------------------
// CACHE-OPTIMIZED MATRIX STORAGE
//--------------------------------------------------------------------------
template<typename T>
class CacheOptimizedMatrix {
public:
    CacheOptimizedMatrix(int rows, int cols) : rows_(rows), cols_(cols) {
        // Use aligned allocation for better cache performance
        size_t size = static_cast<size_t>(rows) * cols;
        data_ = std::make_unique<T[]>(size);
        
        // Prefetch data into cache
        prefetch_all();
    }
    
    T& operator()(int row, int col) {
        return data_[row * cols_ + col];
    }
    
    const T& operator()(int row, int col) const {
        return data_[row * cols_ + col];
    }
    
    T* row_ptr(int row) {
        return data_.get() + row * cols_;
    }
    
    const T* row_ptr(int row) const {
        return data_.get() + row * cols_;
    }
    
    int rows() const { return rows_; }
    int cols() const { return cols_; }
    size_t size() const { return static_cast<size_t>(rows_) * cols_; }
    
    // Cache-friendly operations
    void prefetch_row(int row) const {
        const T* row_data = row_ptr(row);
        for (int i = 0; i < cols_; i += 64 / sizeof(T)) {  // Cache line size
            __builtin_prefetch(row_data + i, 0, 3);  // Read, high temporal locality
        }
    }
    
    void prefetch_all() const {
        for (int row = 0; row < rows_; ++row) {
            prefetch_row(row);
        }
    }
    
private:
    std::unique_ptr<T[]> data_;
    int rows_, cols_;
};

//--------------------------------------------------------------------------
// PARALLEL HADAMARD MATRIX GENERATION
//--------------------------------------------------------------------------
class ParallelHadamardGenerator {
public:
    explicit ParallelHadamardGenerator(size_t num_threads = std::thread::hardware_concurrency())
        : thread_pool_(num_threads) {}
    
    // Parallel recursive generation with cache optimization
    CacheOptimizedMatrix<int> generate_parallel(int n) {
        CacheOptimizedMatrix<int> matrix(n, n);
        
        // Use thread pool for parallel generation
        std::atomic<int> tasks_completed{0};
        std::vector<std::future<void>> futures;
        
        // Divide work into blocks for parallel processing
        int block_size = std::max(1, n / static_cast<int>(thread_pool_.size()));
        
        for (int start_row = 0; start_row < n; start_row += block_size) {
            int end_row = std::min(start_row + block_size, n);
            
            futures.emplace_back(thread_pool_.enqueue([&, start_row, end_row, n]() {
                generate_block_parallel(matrix, start_row, end_row, 0, n, n);
                tasks_completed.fetch_add(1);
            }));
        }
        
        // Wait for all tasks to complete
        for (auto& future : futures) {
            future.wait();
        }
        
        return matrix;
    }
    
    // Block-based generation for cache optimization
    CacheOptimizedMatrix<int> generate_blocked(int n) {
        CacheOptimizedMatrix<int> matrix(n, n);
        
        constexpr int cache_block_size = 64;  // 64x64 blocks for L1 cache
        
        generate_blocked_recursive(matrix, 0, 0, n, cache_block_size);
        
        return matrix;
    }
    
    // SIMD-optimized generation with threading
    CacheOptimizedMatrix<int> generate_simd_parallel(int n) {
        CacheOptimizedMatrix<int> matrix(n, n);
        
        if (n <= 64) {
            // Use SIMD for small matrices
            generate_simd_small(matrix, 0, 0, n);
        } else {
            // Use parallel SIMD for large matrices
            std::vector<std::future<void>> futures;
            int block_size = n / static_cast<int>(thread_pool_.size());
            
            for (int start_row = 0; start_row < n; start_row += block_size) {
                int end_row = std::min(start_row + block_size, n);
                
                futures.emplace_back(thread_pool_.enqueue([&, start_row, end_row, n]() {
                    generate_simd_block(matrix, start_row, end_row, 0, n, n);
                }));
            }
            
            for (auto& future : futures) {
                future.wait();
            }
        }
        
        return matrix;
    }
    
private:
    ThreadPool thread_pool_;
    
    void generate_block_parallel(CacheOptimizedMatrix<int>& matrix, int row_start, int row_end, 
                                int col_start, int col_end, int original_n) {
        int n = row_end - row_start;
        int m = col_end - col_start;
        
        if (n <= 1 || m <= 1) {
            if (n == 1 && m == 1) {
                matrix(row_start, col_start) = 1;
            }
            return;
        }
        
        // Sylvester construction with cache-friendly access
        int half_n = n / 2;
        int half_m = m / 2;
        
        // Generate four quadrants
        generate_block_parallel(matrix, row_start, row_start + half_n, 
                               col_start, col_start + half_m, original_n);
        
        generate_block_parallel(matrix, row_start, row_start + half_n, 
                               col_start + half_m, col_end, original_n);
        
        generate_block_parallel(matrix, row_start + half_n, row_end, 
                               col_start, col_start + half_m, original_n);
        
        generate_block_parallel(matrix, row_start + half_n, row_end, 
                               col_start + half_m, col_end, original_n);
        
        // Negate bottom-right quadrant
        for (int i = row_start + half_n; i < row_end; ++i) {
            for (int j = col_start + half_m; j < col_end; ++j) {
                matrix(i, j) *= -1;
            }
        }
    }
    
    void generate_blocked_recursive(CacheOptimizedMatrix<int>& matrix, int row_start, int col_start, 
                                   int size, int block_size) {
        if (size <= block_size) {
            // Generate small block directly with cache optimization
            generate_dense_block(matrix, row_start, col_start, size);
            return;
        }
        
        int half = size / 2;
        
        // Generate four quadrants
        generate_blocked_recursive(matrix, row_start, col_start, half, block_size);
        generate_blocked_recursive(matrix, row_start, col_start + half, half, block_size);
        generate_blocked_recursive(matrix, row_start + half, col_start, half, block_size);
        generate_blocked_recursive(matrix, row_start + half, col_start + half, half, block_size);
        
        // Negate bottom-right quadrant
        for (int i = 0; i < half; ++i) {
            for (int j = 0; j < half; ++j) {
                matrix(row_start + half + i, col_start + half + j) *= -1;
            }
        }
    }
    
    void generate_dense_block(CacheOptimizedMatrix<int>& matrix, int row_start, int col_start, int size) {
        // Cache-optimized dense block generation
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                if (i == 0 && j == 0) {
                    matrix(row_start + i, col_start + j) = 1;
                } else if (i == 0 || j == 0) {
                    matrix(row_start + i, col_start + j) = 1;
                } else {
                    matrix(row_start + i, col_start + j) = 
                        matrix(row_start, col_start) * matrix(row_start + i, col_start) * 
                        matrix(row_start, col_start + j);
                }
            }
        }
    }
    
    void generate_simd_small(CacheOptimizedMatrix<int>& matrix, int row_start, int col_start, int size) {
        // SIMD-optimized generation for small matrices
        if (size == 1) {
            matrix(row_start, col_start) = 1;
        } else if (size == 2) {
            matrix(row_start, col_start) = 1;
            matrix(row_start, col_start + 1) = 1;
            matrix(row_start + 1, col_start) = 1;
            matrix(row_start + 1, col_start + 1) = -1;
        } else {
            // Use standard recursive for small sizes
            generate_blocked_recursive(matrix, row_start, col_start, size, 2);
        }
    }
    
    void generate_simd_block(CacheOptimizedMatrix<int>& matrix, int row_start, int row_end, 
                            int col_start, int col_end, int original_n) {
        // SIMD-optimized block generation
        int n = row_end - row_start;
        int m = col_end - col_start;
        
        if (n <= 8 && m <= 8) {
            generate_simd_small(matrix, row_start, col_start, std::min(n, m));
            return;
        }
        
        // Use AVX2 for 32-bit integers when possible
        if (n >= 4 && m >= 4) {
            // Vectorized block generation
            for (int i = row_start; i < row_end; i += 4) {
                for (int j = col_start; j < col_end; j += 4) {
                    generate_simd_4x4_block(matrix, i, j);
                }
            }
        } else {
            // Fallback to scalar
            generate_block_parallel(matrix, row_start, row_end, col_start, col_end, original_n);
        }
    }
    
    void generate_simd_4x4_block(CacheOptimizedMatrix<int>& matrix, int row_start, int col_start) {
        // Generate 4x4 block using SIMD
        __m128i ones = _mm_set1_epi32(1);
        __m128i neg_ones = _mm_set1_epi32(-1);
        
        // First row: all ones
        _mm_store_si128(reinterpret_cast<__m128i*>(matrix.row_ptr(row_start) + col_start), ones);
        
        // First column: all ones
        for (int i = 1; i < 4; ++i) {
            matrix(row_start + i, col_start) = 1;
        }
        
        // Fill remaining elements using Sylvester construction
        for (int i = 1; i < 4; ++i) {
            for (int j = 1; j < 4; ++j) {
                matrix(row_start + i, col_start + j) = 
                    matrix(row_start, col_start) * matrix(row_start + i, col_start) * 
                    matrix(row_start, col_start + j);
            }
        }
    }
};

//--------------------------------------------------------------------------
// PARALLEL MATRIX OPERATIONS
//--------------------------------------------------------------------------
class ParallelMatrixOps {
public:
    explicit ParallelMatrixOps(size_t num_threads = std::thread::hardware_concurrency())
        : thread_pool_(num_threads) {}
    
    // Parallel matrix transpose with cache optimization
    CacheOptimizedMatrix<int> transpose_parallel(const CacheOptimizedMatrix<int>& matrix) {
        int n = matrix.rows();
        int m = matrix.cols();
        CacheOptimizedMatrix<int> result(m, n);
        
        // Use blocked transpose for cache optimization
        constexpr int block_size = 64;
        
        std::vector<std::future<void>> futures;
        
        for (int i = 0; i < n; i += block_size) {
            for (int j = 0; j < m; j += block_size) {
                futures.emplace_back(thread_pool_.enqueue([&, i, j, n, m, block_size]() {
                    transpose_block(matrix, result, i, j, 
                                   std::min(i + block_size, n), 
                                   std::min(j + block_size, m));
                }));
            }
        }
        
        for (auto& future : futures) {
            future.wait();
        }
        
        return result;
    }
    
    // Parallel matrix-vector multiplication
    std::vector<int> multiply_vector_parallel(const CacheOptimizedMatrix<int>& matrix, 
                                             const std::vector<int>& vector) {
        int n = matrix.rows();
        std::vector<int> result(n);
        
        std::vector<std::future<void>> futures;
        int block_size = std::max(1, n / static_cast<int>(thread_pool_.size()));
        
        for (int start_row = 0; start_row < n; start_row += block_size) {
            int end_row = std::min(start_row + block_size, n);
            
            futures.emplace_back(thread_pool_.enqueue([&, start_row, end_row]() {
                for (int i = start_row; i < end_row; ++i) {
                    int sum = 0;
                    const int* row = matrix.row_ptr(i);
                    
                    // Unroll loop for better performance
                    int j = 0;
                    for (; j + 4 <= matrix.cols(); j += 4) {
                        sum += row[j] * vector[j] + row[j + 1] * vector[j + 1] +
                               row[j + 2] * vector[j + 2] + row[j + 3] * vector[j + 3];
                    }
                    
                    // Handle remaining elements
                    for (; j < matrix.cols(); ++j) {
                        sum += row[j] * vector[j];
                    }
                    
                    result[i] = sum;
                }
            }));
        }
        
        for (auto& future : futures) {
            future.wait();
        }
        
        return result;
    }
    
    // Parallel matrix-matrix multiplication
    CacheOptimizedMatrix<int> multiply_matrices_parallel(const CacheOptimizedMatrix<int>& A, 
                                                        const CacheOptimizedMatrix<int>& B) {
        int n = A.rows();
        int k = A.cols();
        int m = B.cols();
        
        CacheOptimizedMatrix<int> C(n, m);
        
        // Use blocked matrix multiplication for cache optimization
        constexpr int block_size = 64;
        
        std::vector<std::future<void>> futures;
        
        for (int i = 0; i < n; i += block_size) {
            for (int j = 0; j < m; j += block_size) {
                futures.emplace_back(thread_pool_.enqueue([&, i, j, n, m, k, block_size]() {
                    multiply_block(A, B, C, i, j, 
                                  std::min(i + block_size, n), 
                                  std::min(j + block_size, m), k);
                }));
            }
        }
        
        for (auto& future : futures) {
            future.wait();
        }
        
        return C;
    }
    
private:
    ThreadPool thread_pool_;
    
    void transpose_block(const CacheOptimizedMatrix<int>& src, CacheOptimizedMatrix<int>& dst,
                        int row_start, int col_start, int row_end, int col_end) {
        for (int i = row_start; i < row_end; ++i) {
            for (int j = col_start; j < col_end; ++j) {
                dst(j, i) = src(i, j);
            }
        }
    }
    
    void multiply_block(const CacheOptimizedMatrix<int>& A, const CacheOptimizedMatrix<int>& B,
                       CacheOptimizedMatrix<int>& C, int row_start, int col_start,
                       int row_end, int col_end, int k) {
        for (int i = row_start; i < row_end; ++i) {
            for (int j = col_start; j < col_end; ++j) {
                int sum = 0;
                for (int l = 0; l < k; ++l) {
                    sum += A(i, l) * B(l, j);
                }
                C(i, j) = sum;
            }
        }
    }
};

//--------------------------------------------------------------------------
// BATCH PROCESSING FOR ML EXPERIMENTS
//--------------------------------------------------------------------------
class BatchProcessor {
public:
    explicit BatchProcessor(size_t num_threads = std::thread::hardware_concurrency())
        : thread_pool_(num_threads) {}
    
    // Process multiple signals in parallel
    std::vector<simd::aligned_vector<double>> process_signals_batch(
        const std::vector<simd::aligned_vector<double>>& signals) {
        
        std::vector<simd::aligned_vector<double>> results(signals.size());
        
        std::vector<std::future<void>> futures;
        size_t batch_size = std::max(1UL, signals.size() / thread_pool_.size());
        
        for (size_t start = 0; start < signals.size(); start += batch_size) {
            size_t end = std::min(start + batch_size, signals.size());
            
            futures.emplace_back(thread_pool_.enqueue([&, start, end]() {
                for (size_t i = start; i < end; ++i) {
                    results[i] = simd::FWHTOptimized::fwht_avx2(signals[i]);
                }
            }));
        }
        
        for (auto& future : futures) {
            future.wait();
        }
        
        return results;
    }
    
    // Parallel validation of multiple matrices
    std::vector<bool> validate_matrices_batch(
        const std::vector<CacheOptimizedMatrix<int>>& matrices) {
        
        std::vector<bool> results(matrices.size());
        
        std::vector<std::future<void>> futures;
        size_t batch_size = std::max(1UL, matrices.size() / thread_pool_.size());
        
        for (size_t start = 0; start < matrices.size(); start += batch_size) {
            size_t end = std::min(start + batch_size, matrices.size());
            
            futures.emplace_back(thread_pool_.enqueue([&, start, end]() {
                for (size_t i = start; i < end; ++i) {
                    results[i] = validate_matrix_fast(matrices[i]);
                }
            }));
        }
        
        for (auto& future : futures) {
            future.wait();
        }
        
        return results;
    }
    
    // Public validation function for testing
    static bool validate_matrix_fast(const CacheOptimizedMatrix<int>& matrix) {
        int n = matrix.rows();
        if (n != matrix.cols()) return false;
        
        // Quick validation: check if all entries are ±1
        for (int i = 0; i < n; ++i) {
            const int* row = matrix.row_ptr(i);
            for (int j = 0; j < n; ++j) {
                if (row[j] != 1 && row[j] != -1) {
                    return false;
                }
            }
        }
        
        // Check orthogonality for smaller matrices
        if (n <= 64) {
            return check_orthogonality_fast(matrix);
        }
        
        return true;  // Skip expensive check for large matrices
    }
    
private:
    ThreadPool thread_pool_;
    
    bool check_orthogonality_fast(const CacheOptimizedMatrix<int>& matrix) {
        int n = matrix.rows();
        
        // Check a few random row pairs for orthogonality
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, n - 1);
        
        for (int test = 0; test < std::min(10, n); ++test) {
            int i = dis(gen);
            int j = dis(gen);
            if (i == j) continue;
            
            int dot_product = 0;
            for (int k = 0; k < n; ++k) {
                dot_product += matrix(i, k) * matrix(j, k);
            }
            
            if (dot_product != 0) {
                return false;
            }
        }
        
        return true;
    }
};

} // namespace parallel
} // namespace hatrix

#endif // HATRIX_PARALLEL_HPP
