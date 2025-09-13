//======================================================================
// hatrix_containers.hpp
//----------------------------------------------------------------------
// High-quality containers inspired by Boost.
// Provides optimized data structures without external dependencies.
//
// LICENSE: MIT
// AUTHOR : Nicholas Terek
// VERSION: 1.0.0
//======================================================================
#ifndef HATRIX_CONTAINERS_HPP
#define HATRIX_CONTAINERS_HPP

#include <vector>
#include <array>
#include <memory>
#include <algorithm>
#include <type_traits>
#include <initializer_list>
#include <stdexcept>
#include <cassert>
#include <cstring>

namespace hatrix {
namespace containers {

//--------------------------------------------------------------------------
// SMALL VECTOR - Inspired by boost::container::small_vector
//--------------------------------------------------------------------------
template<typename T, size_t N>
class small_vector {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using iterator = T*;
    using const_iterator = const T*;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    
    // Constructors
    small_vector() : size_(0), capacity_(N), data_(stack_data_) {}
    
    explicit small_vector(size_type count) : size_(0), capacity_(N), data_(stack_data_) {
        resize(count);
    }
    
    small_vector(size_type count, const T& value) : size_(0), capacity_(N), data_(stack_data_) {
        resize(count, value);
    }
    
    template<typename InputIt>
    small_vector(InputIt first, InputIt last) : size_(0), capacity_(N), data_(stack_data_) {
        assign(first, last);
    }
    
    small_vector(std::initializer_list<T> init) : size_(0), capacity_(N), data_(stack_data_) {
        assign(init.begin(), init.end());
    }
    
    small_vector(const small_vector& other) : size_(0), capacity_(N), data_(stack_data_) {
        assign(other.begin(), other.end());
    }
    
    small_vector(small_vector&& other) noexcept : size_(0), capacity_(N), data_(stack_data_) {
        swap(other);
    }
    
    ~small_vector() {
        clear();
        if (data_ != stack_data_) {
            delete[] data_;
        }
    }
    
    // Assignment operators
    small_vector& operator=(const small_vector& other) {
        if (this != &other) {
            assign(other.begin(), other.end());
        }
        return *this;
    }
    
    small_vector& operator=(small_vector&& other) noexcept {
        if (this != &other) {
            clear();
            if (data_ != stack_data_) {
                delete[] data_;
            }
            size_ = other.size_;
            capacity_ = other.capacity_;
            data_ = other.data_;
            other.size_ = 0;
            other.capacity_ = N;
            other.data_ = other.stack_data_;
        }
        return *this;
    }
    
    small_vector& operator=(std::initializer_list<T> init) {
        assign(init.begin(), init.end());
        return *this;
    }
    
    // Element access
    reference at(size_type pos) {
        if (pos >= size_) {
            throw std::out_of_range("small_vector::at");
        }
        return data_[pos];
    }
    
    const_reference at(size_type pos) const {
        if (pos >= size_) {
            throw std::out_of_range("small_vector::at");
        }
        return data_[pos];
    }
    
    reference operator[](size_type pos) {
        assert(pos < size_);
        return data_[pos];
    }
    
    const_reference operator[](size_type pos) const {
        assert(pos < size_);
        return data_[pos];
    }
    
    reference front() {
        assert(!empty());
        return data_[0];
    }
    
    const_reference front() const {
        assert(!empty());
        return data_[0];
    }
    
    reference back() {
        assert(!empty());
        return data_[size_ - 1];
    }
    
    const_reference back() const {
        assert(!empty());
        return data_[size_ - 1];
    }
    
    pointer data() { return data_; }
    const_pointer data() const { return data_; }
    
    // Iterators
    iterator begin() { return data_; }
    iterator end() { return data_ + size_; }
    const_iterator begin() const { return data_; }
    const_iterator end() const { return data_ + size_; }
    const_iterator cbegin() const { return data_; }
    const_iterator cend() const { return data_ + size_; }
    
    // Capacity
    bool empty() const { return size_ == 0; }
    size_type size() const { return size_; }
    size_type max_size() const { return std::numeric_limits<size_type>::max(); }
    size_type capacity() const { return capacity_; }
    
    void reserve(size_type new_cap) {
        if (new_cap > capacity_) {
            reallocate(new_cap);
        }
    }
    
    void shrink_to_fit() {
        if (size_ < capacity_) {
            reallocate(size_);
        }
    }
    
    // Modifiers
    void clear() {
        for (size_type i = 0; i < size_; ++i) {
            data_[i].~T();
        }
        size_ = 0;
    }
    
    iterator insert(const_iterator pos, const T& value) {
        return insert(pos, 1, value);
    }
    
    iterator insert(const_iterator pos, T&& value) {
        size_type index = pos - begin();
        if (size_ >= capacity_) {
            reallocate(capacity_ * 2);
        }
        
        // Move elements
        for (size_type i = size_; i > index; --i) {
            new (data_ + i) T(std::move(data_[i - 1]));
            data_[i - 1].~T();
        }
        
        new (data_ + index) T(std::move(value));
        size_++;
        return begin() + index;
    }
    
    iterator insert(const_iterator pos, size_type count, const T& value) {
        size_type index = pos - begin();
        
        if (size_ + count > capacity_) {
            reallocate(std::max(capacity_ * 2, size_ + count));
        }
        
        // Move elements
        for (size_type i = size_ + count - 1; i >= index + count; --i) {
            new (data_ + i) T(std::move(data_[i - count]));
            data_[i - count].~T();
        }
        
        // Insert new elements
        for (size_type i = 0; i < count; ++i) {
            new (data_ + index + i) T(value);
        }
        
        size_ += count;
        return begin() + index;
    }
    
    template<typename InputIt>
    iterator insert(const_iterator pos, InputIt first, InputIt last) {
        size_type count = std::distance(first, last);
        size_type index = pos - begin();
        
        if (size_ + count > capacity_) {
            reallocate(std::max(capacity_ * 2, size_ + count));
        }
        
        // Move elements
        for (size_type i = size_ + count - 1; i >= index + count; --i) {
            new (data_ + i) T(std::move(data_[i - count]));
            data_[i - count].~T();
        }
        
        // Insert new elements
        size_type i = 0;
        for (auto it = first; it != last; ++it, ++i) {
            new (data_ + index + i) T(*it);
        }
        
        size_ += count;
        return begin() + index;
    }
    
    iterator erase(const_iterator pos) {
        return erase(pos, pos + 1);
    }
    
    iterator erase(const_iterator first, const_iterator last) {
        size_type count = last - first;
        size_type index = first - begin();
        
        // Destroy elements
        for (size_type i = index; i < index + count; ++i) {
            data_[i].~T();
        }
        
        // Move elements
        for (size_type i = index; i < size_ - count; ++i) {
            new (data_ + i) T(std::move(data_[i + count]));
            data_[i + count].~T();
        }
        
        size_ -= count;
        return begin() + index;
    }
    
    void push_back(const T& value) {
        if (size_ >= capacity_) {
            reallocate(capacity_ * 2);
        }
        new (data_ + size_) T(value);
        size_++;
    }
    
    void push_back(T&& value) {
        if (size_ >= capacity_) {
            reallocate(capacity_ * 2);
        }
        new (data_ + size_) T(std::move(value));
        size_++;
    }
    
    void pop_back() {
        assert(!empty());
        data_[size_ - 1].~T();
        size_--;
    }
    
    void resize(size_type count) {
        resize(count, T{});
    }
    
    void resize(size_type count, const T& value) {
        if (count > size_) {
            if (count > capacity_) {
                reallocate(count);
            }
            for (size_type i = size_; i < count; ++i) {
                new (data_ + i) T(value);
            }
        } else if (count < size_) {
            for (size_type i = count; i < size_; ++i) {
                data_[i].~T();
            }
        }
        size_ = count;
    }
    
    void swap(small_vector& other) noexcept {
        std::swap(size_, other.size_);
        std::swap(capacity_, other.capacity_);
        std::swap(data_, other.data_);
        std::swap(stack_data_, other.stack_data_);
    }
    
    template<typename InputIt>
    void assign(InputIt first, InputIt last) {
        clear();
        size_type count = std::distance(first, last);
        if (count > capacity_) {
            reallocate(count);
        }
        for (auto it = first; it != last; ++it) {
            new (data_ + size_) T(*it);
            size_++;
        }
    }
    
    void assign(size_type count, const T& value) {
        clear();
        if (count > capacity_) {
            reallocate(count);
        }
        for (size_type i = 0; i < count; ++i) {
            new (data_ + i) T(value);
        }
        size_ = count;
    }
    
private:
    void reallocate(size_type new_capacity) {
        T* new_data;
        if (new_capacity <= N) {
            new_data = stack_data_;
        } else {
            new_data = new T[new_capacity];
        }
        
        // Move existing elements
        for (size_type i = 0; i < size_; ++i) {
            new (new_data + i) T(std::move(data_[i]));
            data_[i].~T();
        }
        
        if (data_ != stack_data_) {
            delete[] data_;
        }
        
        data_ = new_data;
        capacity_ = new_capacity;
    }
    
    size_type size_;
    size_type capacity_;
    T* data_;
    alignas(T) char stack_data_[N * sizeof(T)];
};

//--------------------------------------------------------------------------
// ALIGNED VECTOR - Inspired by boost::alignment::aligned_allocator
//--------------------------------------------------------------------------
template<typename T, size_t Alignment = 64>
class aligned_vector {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using iterator = T*;
    using const_iterator = const T*;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    
    // Constructors
    aligned_vector() : size_(0), capacity_(0), data_(nullptr) {}
    
    explicit aligned_vector(size_type count) : size_(0), capacity_(0), data_(nullptr) {
        resize(count);
    }
    
    aligned_vector(size_type count, const T& value) : size_(0), capacity_(0), data_(nullptr) {
        resize(count, value);
    }
    
    template<typename InputIt>
    aligned_vector(InputIt first, InputIt last) : size_(0), capacity_(0), data_(nullptr) {
        assign(first, last);
    }
    
    aligned_vector(std::initializer_list<T> init) : size_(0), capacity_(0), data_(nullptr) {
        assign(init.begin(), init.end());
    }
    
    aligned_vector(const aligned_vector& other) : size_(0), capacity_(0), data_(nullptr) {
        assign(other.begin(), other.end());
    }
    
    aligned_vector(aligned_vector&& other) noexcept 
        : size_(other.size_), capacity_(other.capacity_), data_(other.data_) {
        other.size_ = 0;
        other.capacity_ = 0;
        other.data_ = nullptr;
    }
    
    ~aligned_vector() {
        clear();
        if (data_) {
            aligned_free(data_);
        }
    }
    
    // Assignment operators
    aligned_vector& operator=(const aligned_vector& other) {
        if (this != &other) {
            assign(other.begin(), other.end());
        }
        return *this;
    }
    
    aligned_vector& operator=(aligned_vector&& other) noexcept {
        if (this != &other) {
            clear();
            if (data_) {
                aligned_free(data_);
            }
            size_ = other.size_;
            capacity_ = other.capacity_;
            data_ = other.data_;
            other.size_ = 0;
            other.capacity_ = 0;
            other.data_ = nullptr;
        }
        return *this;
    }
    
    aligned_vector& operator=(std::initializer_list<T> init) {
        assign(init.begin(), init.end());
        return *this;
    }
    
    // Element access
    reference at(size_type pos) {
        if (pos >= size_) {
            throw std::out_of_range("aligned_vector::at");
        }
        return data_[pos];
    }
    
    const_reference at(size_type pos) const {
        if (pos >= size_) {
            throw std::out_of_range("aligned_vector::at");
        }
        return data_[pos];
    }
    
    reference operator[](size_type pos) {
        assert(pos < size_);
        return data_[pos];
    }
    
    const_reference operator[](size_type pos) const {
        assert(pos < size_);
        return data_[pos];
    }
    
    reference front() {
        assert(!empty());
        return data_[0];
    }
    
    const_reference front() const {
        assert(!empty());
        return data_[0];
    }
    
    reference back() {
        assert(!empty());
        return data_[size_ - 1];
    }
    
    const_reference back() const {
        assert(!empty());
        return data_[size_ - 1];
    }
    
    pointer data() { return data_; }
    const_pointer data() const { return data_; }
    
    // Iterators
    iterator begin() { return data_; }
    iterator end() { return data_ + size_; }
    const_iterator begin() const { return data_; }
    const_iterator end() const { return data_ + size_; }
    const_iterator cbegin() const { return data_; }
    const_iterator cend() const { return data_ + size_; }
    
    // Capacity
    bool empty() const { return size_ == 0; }
    size_type size() const { return size_; }
    size_type max_size() const { return std::numeric_limits<size_type>::max(); }
    size_type capacity() const { return capacity_; }
    
    void reserve(size_type new_cap) {
        if (new_cap > capacity_) {
            reallocate(new_cap);
        }
    }
    
    void shrink_to_fit() {
        if (size_ < capacity_) {
            reallocate(size_);
        }
    }
    
    // Modifiers
    void clear() {
        for (size_type i = 0; i < size_; ++i) {
            data_[i].~T();
        }
        size_ = 0;
    }
    
    void push_back(const T& value) {
        if (size_ >= capacity_) {
            reallocate(capacity_ == 0 ? 1 : capacity_ * 2);
        }
        new (data_ + size_) T(value);
        size_++;
    }
    
    void push_back(T&& value) {
        if (size_ >= capacity_) {
            reallocate(capacity_ == 0 ? 1 : capacity_ * 2);
        }
        new (data_ + size_) T(std::move(value));
        size_++;
    }
    
    void pop_back() {
        assert(!empty());
        data_[size_ - 1].~T();
        size_--;
    }
    
    void resize(size_type count) {
        resize(count, T{});
    }
    
    void resize(size_type count, const T& value) {
        if (count > size_) {
            if (count > capacity_) {
                reallocate(count);
            }
            for (size_type i = size_; i < count; ++i) {
                new (data_ + i) T(value);
            }
        } else if (count < size_) {
            for (size_type i = count; i < size_; ++i) {
                data_[i].~T();
            }
        }
        size_ = count;
    }
    
    void swap(aligned_vector& other) noexcept {
        std::swap(size_, other.size_);
        std::swap(capacity_, other.capacity_);
        std::swap(data_, other.data_);
    }
    
    template<typename InputIt>
    void assign(InputIt first, InputIt last) {
        clear();
        size_type count = std::distance(first, last);
        if (count > capacity_) {
            reallocate(count);
        }
        for (auto it = first; it != last; ++it) {
            new (data_ + size_) T(*it);
            size_++;
        }
    }
    
    void assign(size_type count, const T& value) {
        clear();
        if (count > capacity_) {
            reallocate(count);
        }
        for (size_type i = 0; i < count; ++i) {
            new (data_ + i) T(value);
        }
        size_ = count;
    }
    
    // Alignment utilities
    bool is_aligned() const {
        return data_ && (reinterpret_cast<uintptr_t>(data_) % Alignment == 0);
    }
    
    static constexpr size_type alignment() { return Alignment; }
    
private:
    void reallocate(size_type new_capacity) {
        T* new_data = static_cast<T*>(aligned_alloc(Alignment, new_capacity * sizeof(T)));
        if (!new_data) {
            throw std::bad_alloc();
        }
        
        // Move existing elements
        for (size_type i = 0; i < size_; ++i) {
            new (new_data + i) T(std::move(data_[i]));
            data_[i].~T();
        }
        
        if (data_) {
            aligned_free(data_);
        }
        
        data_ = new_data;
        capacity_ = new_capacity;
    }
    
    static void* aligned_alloc(size_t alignment, size_t size) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, alignment, size) != 0) {
            return nullptr;
        }
        return ptr;
    }
    
    static void aligned_free(void* ptr) {
        std::free(ptr);
    }
    
    size_type size_;
    size_type capacity_;
    T* data_;
};

//--------------------------------------------------------------------------
// CONVENIENCE TYPE ALIASES
//--------------------------------------------------------------------------
template<typename T, size_t N = 8>
using small_vector_t = small_vector<T, N>;

template<typename T, size_t Alignment = 64>
using aligned_vector_t = aligned_vector<T, Alignment>;

} // namespace containers
} // namespace hatrix

#endif // HATRIX_CONTAINERS_HPP
