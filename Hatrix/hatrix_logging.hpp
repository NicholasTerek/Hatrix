//======================================================================
// hatrix_logging.hpp
//----------------------------------------------------------------------
// High-quality logging system inspired by spdlog.
// Provides structured logging, performance monitoring, and multiple outputs
// without external dependencies.
//
// LICENSE: MIT
// AUTHOR : Nicholas Terek
// VERSION: 1.0.0
//======================================================================
#ifndef HATRIX_LOGGING_HPP
#define HATRIX_LOGGING_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <mutex>
#include <thread>
#include <chrono>
#include <atomic>
#include <queue>
#include <condition_variable>
#include <sstream>
#include <iomanip>
#include <vector>
#include <map>
#include <functional>
#include <cassert>
#include <cstring>
#include <ctime>

namespace hatrix {
namespace logging {

//--------------------------------------------------------------------------
// LOG LEVELS
//--------------------------------------------------------------------------
enum class Level : int {
    TRACE = 0,
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERROR = 4,
    CRITICAL = 5,
    OFF = 6
};

inline const char* level_to_string(Level level) {
    switch (level) {
        case Level::TRACE: return "TRACE";
        case Level::DEBUG: return "DEBUG";
        case Level::INFO: return "INFO";
        case Level::WARN: return "WARN";
        case Level::ERROR: return "ERROR";
        case Level::CRITICAL: return "CRITICAL";
        case Level::OFF: return "OFF";
        default: return "UNKNOWN";
    }
}

inline const char* level_to_color(Level level) {
    switch (level) {
        case Level::TRACE: return "\033[37m";  // White
        case Level::DEBUG: return "\033[36m";  // Cyan
        case Level::INFO: return "\033[32m";   // Green
        case Level::WARN: return "\033[33m";   // Yellow
        case Level::ERROR: return "\033[31m";  // Red
        case Level::CRITICAL: return "\033[35m"; // Magenta
        default: return "\033[0m";             // Reset
    }
}

inline const char* level_reset_color() {
    return "\033[0m";
}

//--------------------------------------------------------------------------
// LOG MESSAGE
//--------------------------------------------------------------------------
struct LogMessage {
    Level level;
    std::string timestamp;
    std::string thread_id;
    std::string source_file;
    int source_line;
    std::string message;
    std::string formatted_message;
    
    LogMessage(Level l, const std::string& file, int line, const std::string& msg)
        : level(l), source_file(file), source_line(line), message(msg) {
        
        // Generate timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        ss << "." << std::setfill('0') << std::setw(3) << ms.count();
        timestamp = ss.str();
        
        // Get thread ID
        std::stringstream tid;
        tid << std::this_thread::get_id();
        thread_id = tid.str();
        
        // Extract filename from path
        size_t pos = source_file.find_last_of("/\\");
        if (pos != std::string::npos) {
            source_file = source_file.substr(pos + 1);
        }
    }
};

//--------------------------------------------------------------------------
// LOG SINK INTERFACE
//--------------------------------------------------------------------------
class LogSink {
public:
    virtual ~LogSink() = default;
    virtual void log(const LogMessage& msg) = 0;
    virtual void flush() = 0;
    virtual void set_level(Level level) = 0;
    virtual Level get_level() const = 0;
};

//--------------------------------------------------------------------------
// CONSOLE SINK
//--------------------------------------------------------------------------
class ConsoleSink : public LogSink {
public:
    explicit ConsoleSink(bool colored = true) : colored_(colored), level_(Level::TRACE) {}
    
    void log(const LogMessage& msg) override {
        if (msg.level < level_) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (colored_) {
            std::cout << level_to_color(msg.level);
        }
        
        std::cout << "[" << msg.timestamp << "] "
                  << "[" << level_to_string(msg.level) << "] "
                  << "[" << msg.thread_id << "] "
                  << "[" << msg.source_file << ":" << msg.source_line << "] "
                  << msg.message;
        
        if (colored_) {
            std::cout << level_reset_color();
        }
        
        std::cout << std::endl;
    }
    
    void flush() override {
        std::lock_guard<std::mutex> lock(mutex_);
        std::cout.flush();
    }
    
    void set_level(Level level) override { level_ = level; }
    Level get_level() const override { return level_; }
    
private:
    std::mutex mutex_;
    bool colored_;
    Level level_;
};

//--------------------------------------------------------------------------
// FILE SINK
//--------------------------------------------------------------------------
class FileSink : public LogSink {
public:
    explicit FileSink(const std::string& filename, Level level = Level::TRACE)
        : filename_(filename), level_(level) {
        file_.open(filename_, std::ios::app);
    }
    
    ~FileSink() {
        if (file_.is_open()) {
            file_.close();
        }
    }
    
    void log(const LogMessage& msg) override {
        if (msg.level < level_) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (file_.is_open()) {
            file_ << "[" << msg.timestamp << "] "
                  << "[" << level_to_string(msg.level) << "] "
                  << "[" << msg.thread_id << "] "
                  << "[" << msg.source_file << ":" << msg.source_line << "] "
                  << msg.message << std::endl;
        }
    }
    
    void flush() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_.is_open()) {
            file_.flush();
        }
    }
    
    void set_level(Level level) override { level_ = level; }
    Level get_level() const override { return level_; }
    
private:
    std::string filename_;
    std::ofstream file_;
    std::mutex mutex_;
    Level level_;
};

//--------------------------------------------------------------------------
// ROTATING FILE SINK
//--------------------------------------------------------------------------
class RotatingFileSink : public LogSink {
public:
    RotatingFileSink(const std::string& filename, size_t max_size, size_t max_files)
        : base_filename_(filename), max_size_(max_size), max_files_(max_files), level_(Level::TRACE) {
        open_file();
    }
    
    void log(const LogMessage& msg) override {
        if (msg.level < level_) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (file_.is_open()) {
            file_ << "[" << msg.timestamp << "] "
                  << "[" << level_to_string(msg.level) << "] "
                  << "[" << msg.thread_id << "] "
                  << "[" << msg.source_file << ":" << msg.source_line << "] "
                  << msg.message << std::endl;
            
            // Check if we need to rotate
            if (file_.tellp() > static_cast<std::streampos>(max_size_)) {
                rotate_file();
            }
        }
    }
    
    void flush() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (file_.is_open()) {
            file_.flush();
        }
    }
    
    void set_level(Level level) override { level_ = level; }
    Level get_level() const override { return level_; }
    
private:
    void open_file() {
        if (file_.is_open()) {
            file_.close();
        }
        file_.open(base_filename_, std::ios::app);
    }
    
    void rotate_file() {
        file_.close();
        
        // Rotate existing files
        for (int i = max_files_ - 1; i > 0; --i) {
            std::string old_name = base_filename_ + "." + std::to_string(i);
            std::string new_name = base_filename_ + "." + std::to_string(i + 1);
            
            std::ifstream old_file(old_name);
            if (old_file.good()) {
                old_file.close();
                std::remove(new_name.c_str());
                std::rename(old_name.c_str(), new_name.c_str());
            }
        }
        
        // Move current file to .1
        std::string first_rotated = base_filename_ + ".1";
        std::remove(first_rotated.c_str());
        std::rename(base_filename_.c_str(), first_rotated.c_str());
        
        // Open new file
        open_file();
    }
    
    std::string base_filename_;
    std::ofstream file_;
    std::mutex mutex_;
    size_t max_size_;
    size_t max_files_;
    Level level_;
};

//--------------------------------------------------------------------------
// ASYNC LOGGER
//--------------------------------------------------------------------------
class AsyncLogger {
public:
    AsyncLogger(size_t queue_size = 8192) 
        : queue_size_(queue_size), running_(true), queue_(), mutex_(), condition_() {
        
        worker_thread_ = std::thread(&AsyncLogger::worker_loop, this);
    }
    
    ~AsyncLogger() {
        stop();
    }
    
    void log(const LogMessage& msg) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        if (queue_.size() >= queue_size_) {
            // Drop oldest message if queue is full
            queue_.pop();
        }
        
        queue_.push(msg);
        condition_.notify_one();
    }
    
    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            running_ = false;
        }
        condition_.notify_all();
        
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }
    
    void add_sink(std::shared_ptr<LogSink> sink) {
        std::lock_guard<std::mutex> lock(sinks_mutex_);
        sinks_.push_back(sink);
    }
    
private:
    void worker_loop() {
        while (running_) {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this] { return !queue_.empty() || !running_; });
            
            while (!queue_.empty()) {
                LogMessage msg = queue_.front();
                queue_.pop();
                lock.unlock();
                
                // Send to all sinks
                {
                    std::lock_guard<std::mutex> sinks_lock(sinks_mutex_);
                    for (auto& sink : sinks_) {
                        sink->log(msg);
                    }
                }
                
                lock.lock();
            }
        }
    }
    
    size_t queue_size_;
    std::atomic<bool> running_;
    std::queue<LogMessage> queue_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::thread worker_thread_;
    
    std::vector<std::shared_ptr<LogSink>> sinks_;
    std::mutex sinks_mutex_;
};

//--------------------------------------------------------------------------
// MAIN LOGGER CLASS
//--------------------------------------------------------------------------
class Logger {
public:
    explicit Logger(const std::string& name) : name_(name), level_(Level::INFO) {}
    
    void add_sink(std::shared_ptr<LogSink> sink) {
        std::lock_guard<std::mutex> lock(sinks_mutex_);
        sinks_.push_back(sink);
    }
    
    void set_level(Level level) {
        level_ = level;
        std::lock_guard<std::mutex> lock(sinks_mutex_);
        for (auto& sink : sinks_) {
            sink->set_level(level);
        }
    }
    
    Level get_level() const { return level_; }
    
    void log(Level level, const std::string& file, int line, const std::string& message) {
        if (level < level_) return;
        
        LogMessage msg(level, file, line, message);
        
        std::lock_guard<std::mutex> lock(sinks_mutex_);
        for (auto& sink : sinks_) {
            sink->log(msg);
        }
    }
    
    void flush() {
        std::lock_guard<std::mutex> lock(sinks_mutex_);
        for (auto& sink : sinks_) {
            sink->flush();
        }
    }
    
private:
    std::string name_;
    Level level_;
    std::vector<std::shared_ptr<LogSink>> sinks_;
    std::mutex sinks_mutex_;
};

//--------------------------------------------------------------------------
// LOGGER MANAGER
//--------------------------------------------------------------------------
class LoggerManager {
public:
    static LoggerManager& instance() {
        static LoggerManager instance;
        return instance;
    }
    
    std::shared_ptr<Logger> get_logger(const std::string& name) {
        std::lock_guard<std::mutex> lock(loggers_mutex_);
        
        auto it = loggers_.find(name);
        if (it != loggers_.end()) {
            return it->second;
        }
        
        auto logger = std::make_shared<Logger>(name);
        loggers_[name] = logger;
        return logger;
    }
    
    void set_default_logger(std::shared_ptr<Logger> logger) {
        default_logger_ = logger;
    }
    
    std::shared_ptr<Logger> get_default_logger() {
        if (!default_logger_) {
            default_logger_ = get_logger("hatrix");
        }
        return default_logger_;
    }
    
    void shutdown() {
        std::lock_guard<std::mutex> lock(loggers_mutex_);
        for (auto& pair : loggers_) {
            pair.second->flush();
        }
        loggers_.clear();
    }
    
private:
    LoggerManager() = default;
    
    std::map<std::string, std::shared_ptr<Logger>> loggers_;
    std::mutex loggers_mutex_;
    std::shared_ptr<Logger> default_logger_;
};

//--------------------------------------------------------------------------
// PERFORMANCE TIMER
//--------------------------------------------------------------------------
class PerformanceTimer {
public:
    explicit PerformanceTimer(const std::string& operation_name, Level log_level = Level::DEBUG)
        : operation_name_(operation_name)
        , log_level_(log_level)
        , start_time_(std::chrono::high_resolution_clock::now()) {
        
        auto logger = LoggerManager::instance().get_default_logger();
        logger->log(log_level_, __FILE__, __LINE__, 
                   "Starting operation: " + operation_name_);
    }
    
    ~PerformanceTimer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time_);
        
        auto logger = LoggerManager::instance().get_default_logger();
        logger->log(log_level_, __FILE__, __LINE__, 
                   "Completed operation: " + operation_name_ + 
                   " in " + std::to_string(duration.count() / 1000.0) + " ms");
    }
    
    void checkpoint(const std::string& checkpoint_name) {
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(current_time - start_time_);
        
        auto logger = LoggerManager::instance().get_default_logger();
        logger->log(log_level_, __FILE__, __LINE__, 
                   "Checkpoint '" + checkpoint_name + "' in " + operation_name_ + 
                   ": " + std::to_string(elapsed.count() / 1000.0) + " ms");
    }
    
private:
    std::string operation_name_;
    Level log_level_;
    std::chrono::high_resolution_clock::time_point start_time_;
};

//--------------------------------------------------------------------------
// CONVENIENCE FUNCTIONS
//--------------------------------------------------------------------------
inline std::shared_ptr<Logger> get_logger(const std::string& name = "hatrix") {
    return LoggerManager::instance().get_logger(name);
}

inline void set_default_logger(std::shared_ptr<Logger> logger) {
    LoggerManager::instance().set_default_logger(logger);
}

inline void shutdown_logging() {
    LoggerManager::instance().shutdown();
}

//--------------------------------------------------------------------------
// LOGGING MACROS
//--------------------------------------------------------------------------
#define HATRIX_LOG(logger, level, ...) \
    do { \
        auto logger_ptr = hatrix::logging::get_logger(logger); \
        if (logger_ptr->get_level() <= level) { \
            std::stringstream ss; \
            ss << __VA_ARGS__; \
            logger_ptr->log(level, __FILE__, __LINE__, ss.str()); \
        } \
    } while(0)

#define HATRIX_TRACE(...) HATRIX_LOG("hatrix", hatrix::logging::Level::TRACE, __VA_ARGS__)
#define HATRIX_DEBUG(...) HATRIX_LOG("hatrix", hatrix::logging::Level::DEBUG, __VA_ARGS__)
#define HATRIX_INFO(...)  HATRIX_LOG("hatrix", hatrix::logging::Level::INFO, __VA_ARGS__)
#define HATRIX_WARN(...)  HATRIX_LOG("hatrix", hatrix::logging::Level::WARN, __VA_ARGS__)
#define HATRIX_ERROR(...) HATRIX_LOG("hatrix", hatrix::logging::Level::ERROR, __VA_ARGS__)
#define HATRIX_CRITICAL(...) HATRIX_LOG("hatrix", hatrix::logging::Level::CRITICAL, __VA_ARGS__)

#define HATRIX_PERFORMANCE_TIMER(name) \
    hatrix::logging::PerformanceTimer timer(name)

#define HATRIX_PERFORMANCE_TIMER_DEBUG(name) \
    hatrix::logging::PerformanceTimer timer(name, hatrix::logging::Level::DEBUG)

#define HATRIX_PERFORMANCE_TIMER_INFO(name) \
    hatrix::logging::PerformanceTimer timer(name, hatrix::logging::Level::INFO)

} // namespace logging
} // namespace hatrix

#endif // HATRIX_LOGGING_HPP
