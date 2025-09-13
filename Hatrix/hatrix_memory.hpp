//======================================================================
// hatrix_memory.hpp
//----------------------------------------------------------------------
// High-quality memory management system for the Hatrix library.
// Provides aligned allocation, RAII wrappers, and memory pool management.
//
// LICENSE: MIT
// AUTHOR : Nicholas Terek
// VERSION: 1.0.0
//======================================================================
#ifndef HATRIX_MEMORY_HPP
#define HATRIX_MEMORY_HPP

#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <boost/align/aligned_allocator.hpp>
#include <boost/container/small_vector.hpp>
#include <boost/pool/pool.hpp>
#include <boost/pool/object_pool.hpp>
#include <boost/pool/singleton_pool.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>
#include <boost/smart_ptr/make_shared.hpp>
#include <boost/smart_ptr/unique_ptr.hpp>
#include <boost/smart_ptr/make_unique.hpp>
#include <boost/smart_ptr/scoped_ptr.hpp>
#include <boost/smart_ptr/intrusive_ptr.hpp>
#include <boost/intrusive_ptr.hpp>
#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include "hatrix_exceptions.hpp"
#include "hatrix_logging.hpp"

namespace hatrix {
namespace memory {

//--------------------------------------------------------------------------
// ALIGNMENT CONSTANTS
//--------------------------------------------------------------------------
static constexpr std::size_t DEFAULT_ALIGNMENT = 64;  // Cache line size
static constexpr std::size_t SIMD_ALIGNMENT = 32;     // AVX alignment
static constexpr std::size_t AVX512_ALIGNMENT = 64;   // AVX-512 alignment

//--------------------------------------------------------------------------
// ALIGNED ALLOCATOR
//--------------------------------------------------------------------------
template<typename T, std::size_t Alignment = DEFAULT_ALIGNMENT>
using aligned_allocator = boost::alignment::aligned_allocator<T, Alignment>;

template<typename T, std::size_t Alignment = DEFAULT_ALIGNMENT>
using aligned_vector = std::vector<T, aligned_allocator<T, Alignment>>;

//--------------------------------------------------------------------------
// MEMORY POOL MANAGER
//--------------------------------------------------------------------------
class MemoryPoolManager {
public:
    static MemoryPoolManager& instance() {
        static MemoryPoolManager instance;
        return instance;
    }
    
    // Get aligned memory pool for specific size
    template<std::size_t Size>
    boost::object_pool<void>& get_pool() {
        std::lock_guard<std::mutex> lock(pools_mutex_);
        auto& pool = pools_[Size];
        if (!pool) {
            pool = std::make_unique<boost::object_pool<void>>(Size, 1024);
            HATRIX_DEBUG("Created memory pool for size: {} bytes", Size);
        }
        return *pool;
    }
    
    // Allocate aligned memory
    template<typename T>
    T* allocate_aligned(std::size_t count, std::size_t alignment = DEFAULT_ALIGNMENT) {
        std::size_t size = count * sizeof(T);
        
        void* ptr = std::aligned_alloc(alignment, size);
        if (!ptr) {
            HATRIX_THROW(exceptions::MemoryAllocationError, size);
        }
        
        HATRIX_DEBUG("Allocated {} bytes aligned to {} for {} objects of type {}",
                    size, alignment, count, typeid(T).name());
        
        allocated_bytes_ += size;
        return static_cast<T*>(ptr);
    }
    
    // Deallocate aligned memory
    template<typename T>
    void deallocate_aligned(T* ptr, std::size_t count, std::size_t alignment = DEFAULT_ALIGNMENT) {
        if (ptr) {
            std::size_t size = count * sizeof(T);
            std::free(ptr);
            allocated_bytes_ -= size;
            
            HATRIX_DEBUG("Deallocated {} bytes for {} objects of type {}",
                        size, count, typeid(T).name());
        }
    }
    
    // Get memory usage statistics
    std::size_t get_allocated_bytes() const { return allocated_bytes_.load(); }
    std::size_t get_pool_count() const { return pools_.size(); }
    
    // Cleanup all pools
    void cleanup() {
        std::lock_guard<std::mutex> lock(pools_mutex_);
        pools_.clear();
        HATRIX_INFO("Memory pools cleaned up");
    }
    
private:
    MemoryPoolManager() = default;
    
    std::unordered_map<std::size_t, std::unique_ptr<boost::object_pool<void>>> pools_;
    std::mutex pools_mutex_;
    std::atomic<std::size_t> allocated_bytes_{0};
};

//--------------------------------------------------------------------------
// RAII MEMORY WRAPPER
//--------------------------------------------------------------------------
template<typename T, std::size_t Alignment = DEFAULT_ALIGNMENT>
class AlignedArray {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using iterator = T*;
    using const_iterator = const T*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    
    explicit AlignedArray(size_type count) 
        : data_(MemoryPoolManager::instance().allocate_aligned<T>(count, Alignment))
        , size_(count) {
        
        HATRIX_DEBUG("Created AlignedArray with {} elements of type {}", 
                    count, typeid(T).name());
        
        // Initialize with default values
        std::uninitialized_default_construct_n(data_, count);
    }
    
    template<typename U>
    AlignedArray(size_type count, const U& value) 
        : data_(MemoryPoolManager::instance().allocate_aligned<T>(count, Alignment))
        , size_(count) {
        
        HATRIX_DEBUG("Created AlignedArray with {} elements of type {} initialized with value",
                    count, typeid(T).name());
        
        // Initialize with value
        std::uninitialized_fill_n(data_, count, static_cast<T>(value));
    }
    
    ~AlignedArray() {
        if (data_) {
            // Destroy objects
            std::destroy_n(data_, size_);
            
            // Deallocate memory
            MemoryPoolManager::instance().deallocate_aligned(data_, size_, Alignment);
            
            HATRIX_DEBUG("Destroyed AlignedArray with {} elements", size_);
        }
    }
    
    // Non-copyable but movable
    AlignedArray(const AlignedArray&) = delete;
    AlignedArray& operator=(const AlignedArray&) = delete;
    
    AlignedArray(AlignedArray&& other) noexcept 
        : data_(other.data_)
        , size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }
    
    AlignedArray& operator=(AlignedArray&& other) noexcept {
        if (this != &other) {
            // Destroy current data
            if (data_) {
                std::destroy_n(data_, size_);
                MemoryPoolManager::instance().deallocate_aligned(data_, size_, Alignment);
            }
            
            // Move from other
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }
    
    // Access operators
    reference operator[](size_type index) {
        HATRIX_DEBUG_ASSERT(index < size_, "Index {} out of bounds for size {}", index, size_);
        return data_[index];
    }
    
    const_reference operator[](size_type index) const {
        HATRIX_DEBUG_ASSERT(index < size_, "Index {} out of bounds for size {}", index, size_);
        return data_[index];
    }
    
    reference at(size_type index) {
        if (index >= size_) {
            HATRIX_THROW(exceptions::MathematicalError, 
                        fmt::format("Index {} out of bounds for size {}", index, size_));
        }
        return data_[index];
    }
    
    const_reference at(size_type index) const {
        if (index >= size_) {
            HATRIX_THROW(exceptions::MathematicalError, 
                        fmt::format("Index {} out of bounds for size {}", index, size_));
        }
        return data_[index];
    }
    
    // Iterator support
    iterator begin() { return data_; }
    iterator end() { return data_ + size_; }
    const_iterator begin() const { return data_; }
    const_iterator end() const { return data_ + size_; }
    const_iterator cbegin() const { return data_; }
    const_iterator cend() const { return data_ + size_; }
    
    // Capacity
    size_type size() const { return size_; }
    bool empty() const { return size_ == 0; }
    
    // Data access
    pointer data() { return data_; }
    const_pointer data() const { return data_; }
    
    // Memory information
    static constexpr size_type alignment() { return Alignment; }
    bool is_aligned() const {
        return reinterpret_cast<std::uintptr_t>(data_) % Alignment == 0;
    }
    
private:
    pointer data_;
    size_type size_;
};

//--------------------------------------------------------------------------
// SMART POINTER TYPES
//--------------------------------------------------------------------------
template<typename T, std::size_t Alignment = DEFAULT_ALIGNMENT>
using aligned_unique_ptr = std::unique_ptr<T, std::function<void(T*)>>;

template<typename T, std::size_t Alignment = DEFAULT_ALIGNMENT>
aligned_unique_ptr<T, Alignment> make_aligned_unique(std::size_t count) {
    auto deleter = [count](T* ptr) {
        if (ptr) {
            std::destroy_n(ptr, count);
            MemoryPoolManager::instance().deallocate_aligned(ptr, count, Alignment);
        }
    };
    
    auto ptr = MemoryPoolManager::instance().allocate_aligned<T>(count, Alignment);
    std::uninitialized_default_construct_n(ptr, count);
    
    return aligned_unique_ptr<T, Alignment>(ptr, deleter);
}

//--------------------------------------------------------------------------
// MATRIX-SPECIFIC MEMORY TYPES
//--------------------------------------------------------------------------
template<typename T, std::size_t Alignment = DEFAULT_ALIGNMENT>
using aligned_matrix = std::vector<AlignedArray<T, Alignment>>;

template<typename T, std::size_t Alignment = DEFAULT_ALIGNMENT>
using aligned_small_matrix = boost::container::small_vector<AlignedArray<T, Alignment>, 8>;

//--------------------------------------------------------------------------
// MEMORY UTILITIES
//--------------------------------------------------------------------------
namespace utils {
    
    // Check if pointer is aligned
    template<std::size_t Alignment>
    bool is_aligned(const void* ptr) {
        return reinterpret_cast<std::uintptr_t>(ptr) % Alignment == 0;
    }
    
    // Get next aligned address
    template<std::size_t Alignment>
    void* align_up(void* ptr) {
        auto addr = reinterpret_cast<std::uintptr_t>(ptr);
        auto aligned_addr = (addr + Alignment - 1) & ~(Alignment - 1);
        return reinterpret_cast<void*>(aligned_addr);
    }
    
    // Calculate aligned size
    template<std::size_t Alignment>
    std::size_t align_size(std::size_t size) {
        return (size + Alignment - 1) & ~(Alignment - 1);
    }
    
    // Memory bandwidth measurement
    class MemoryBandwidth {
    public:
        static double measure_bandwidth(std::size_t size_bytes, std::size_t iterations = 1000) {
            HATRIX_PERFORMANCE_TIMER_INFO("memory_bandwidth_measurement");
            
            auto data = make_aligned_unique<double>(size_bytes / sizeof(double));
            volatile double sum = 0.0;
            
            auto start = std::chrono::high_resolution_clock::now();
            
            for (std::size_t i = 0; i < iterations; ++i) {
                for (std::size_t j = 0; j < size_bytes / sizeof(double); ++j) {
                    sum += data[j];
                }
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
            
            double time_seconds = duration.count() / 1e9;
            double bytes_per_second = (size_bytes * iterations) / time_seconds;
            double mbps = bytes_per_second / (1024.0 * 1024.0);
            
            HATRIX_INFO("Memory bandwidth measurement: {:.2f} MB/s for {} bytes over {} iterations",
                       mbps, size_bytes, iterations);
            
            return mbps;
        }
    };
    
} // namespace utils

//--------------------------------------------------------------------------
// CONVENIENCE FUNCTIONS
//--------------------------------------------------------------------------
inline MemoryPoolManager& get_memory_manager() {
    return MemoryPoolManager::instance();
}

template<typename T, std::size_t Alignment = DEFAULT_ALIGNMENT>
AlignedArray<T, Alignment> make_aligned_array(std::size_t count) {
    return AlignedArray<T, Alignment>(count);
}

template<typename T, std::size_t Alignment = DEFAULT_ALIGNMENT>
AlignedArray<T, Alignment> make_aligned_array(std::size_t count, const T& value) {
    return AlignedArray<T, Alignment>(count, value);
}

} // namespace memory
} // namespace hatrix

#endif // HATRIX_MEMORY_HPP
