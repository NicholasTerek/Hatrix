//======================================================================
// hatrix_format.hpp
//----------------------------------------------------------------------
// High-quality formatting system inspired by fmt.
// Provides type-safe, fast string formatting without external dependencies.
//
// LICENSE: MIT
// AUTHOR : Nicholas Terek
// VERSION: 1.0.0
//======================================================================
#ifndef HATRIX_FORMAT_HPP
#define HATRIX_FORMAT_HPP

#include <string>
#include <sstream>
#include <iomanip>
#include <type_traits>
#include <limits>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <memory>
#include <vector>
#include <array>

namespace hatrix {
namespace format {

//--------------------------------------------------------------------------
// FORMAT SPECIFIER
//--------------------------------------------------------------------------
struct FormatSpec {
    char type = 's';           // s, d, f, e, g, x, o, b
    int width = 0;             // Minimum field width
    int precision = -1;        // Precision for floating point
    char fill = ' ';           // Fill character
    bool left_align = false;   // Left alignment
    bool show_sign = false;    // Show + for positive numbers
    bool zero_pad = false;     // Zero padding
    bool alternate = false;    // Alternate form (#)
    
    static FormatSpec parse(const std::string& spec) {
        FormatSpec fmt;
        if (spec.empty()) return fmt;
        
        size_t pos = 0;
        
        // Parse fill and alignment
        if (pos < spec.length() - 1 && spec[pos + 1] == '<') {
            fmt.fill = spec[pos];
            fmt.left_align = true;
            pos += 2;
        } else if (pos < spec.length() - 1 && spec[pos + 1] == '>') {
            fmt.fill = spec[pos];
            pos += 2;
        } else if (pos < spec.length() - 1 && spec[pos + 1] == '^') {
            fmt.fill = spec[pos];
            pos += 2;
        } else if (spec[pos] == '<') {
            fmt.left_align = true;
            pos++;
        } else if (spec[pos] == '>') {
            pos++;
        } else if (spec[pos] == '^') {
            pos++;
        }
        
        // Parse sign
        if (pos < spec.length() && spec[pos] == '+') {
            fmt.show_sign = true;
            pos++;
        } else if (pos < spec.length() && spec[pos] == '-') {
            pos++; // Skip, handled by left_align
        } else if (pos < spec.length() && spec[pos] == ' ') {
            pos++; // Space for positive numbers
        }
        
        // Parse alternate form
        if (pos < spec.length() && spec[pos] == '#') {
            fmt.alternate = true;
            pos++;
        }
        
        // Parse zero padding
        if (pos < spec.length() && spec[pos] == '0') {
            fmt.zero_pad = true;
            pos++;
        }
        
        // Parse width
        if (pos < spec.length() && std::isdigit(spec[pos])) {
            fmt.width = 0;
            while (pos < spec.length() && std::isdigit(spec[pos])) {
                fmt.width = fmt.width * 10 + (spec[pos] - '0');
                pos++;
            }
        }
        
        // Parse precision
        if (pos < spec.length() && spec[pos] == '.') {
            pos++;
            if (pos < spec.length() && std::isdigit(spec[pos])) {
                fmt.precision = 0;
                while (pos < spec.length() && std::isdigit(spec[pos])) {
                    fmt.precision = fmt.precision * 10 + (spec[pos] - '0');
                    pos++;
                }
            }
        }
        
        // Parse type
        if (pos < spec.length()) {
            fmt.type = spec[pos];
        }
        
        return fmt;
    }
};

//--------------------------------------------------------------------------
// TYPE TRAITS
//--------------------------------------------------------------------------
template<typename T>
struct is_integral_type : std::bool_constant<
    std::is_integral_v<T> && !std::is_same_v<T, bool> && !std::is_same_v<T, char>
> {};

template<typename T>
struct is_floating_type : std::bool_constant<std::is_floating_point_v<T>> {};

template<typename T>
struct is_string_type : std::bool_constant<
    std::is_same_v<T, std::string> || 
    std::is_same_v<T, const char*> ||
    std::is_same_v<T, char*>
> {};

//--------------------------------------------------------------------------
// FORMATTER BASE
//--------------------------------------------------------------------------
template<typename T>
class Formatter {
public:
    static std::string format(const T& value, const FormatSpec& spec) {
        if constexpr (is_integral_type<T>::value) {
            return format_integral(value, spec);
        } else if constexpr (is_floating_type<T>::value) {
            return format_floating(value, spec);
        } else if constexpr (is_string_type<T>::value) {
            return format_string(value, spec);
        } else {
            return format_default(value, spec);
        }
    }
    
private:
    static std::string format_integral(const T& value, const FormatSpec& spec) {
        std::stringstream ss;
        
        // Handle different integer types
        if (spec.type == 'd' || spec.type == 's') {
            ss << value;
        } else if (spec.type == 'x' || spec.type == 'X') {
            if (spec.alternate) ss << "0x";
            ss << std::hex;
            if (spec.type == 'X') ss << std::uppercase;
            ss << value;
        } else if (spec.type == 'o') {
            if (spec.alternate) ss << "0";
            ss << std::oct << value;
        } else if (spec.type == 'b') {
            if (spec.alternate) ss << "0b";
            format_binary(value, ss);
        } else {
            ss << value;
        }
        
        std::string result = ss.str();
        return apply_padding(result, spec);
    }
    
    static std::string format_floating(const T& value, const FormatSpec& spec) {
        std::stringstream ss;
        
        if (spec.precision >= 0) {
            ss << std::fixed << std::setprecision(spec.precision);
        }
        
        if (spec.type == 'f') {
            ss << std::fixed;
        } else if (spec.type == 'e' || spec.type == 'E') {
            ss << std::scientific;
            if (spec.type == 'E') ss << std::uppercase;
        } else if (spec.type == 'g' || spec.type == 'G') {
            ss << std::defaultfloat;
            if (spec.type == 'G') ss << std::uppercase;
        }
        
        ss << value;
        std::string result = ss.str();
        
        // Handle sign
        if (spec.show_sign && value >= 0) {
            result = "+" + result;
        }
        
        return apply_padding(result, spec);
    }
    
    static std::string format_string(const T& value, const FormatSpec& spec) {
        std::string str;
        if constexpr (std::is_same_v<T, std::string>) {
            str = value;
        } else {
            str = std::string(value);
        }
        
        // Apply precision (truncate)
        if (spec.precision >= 0 && spec.precision < static_cast<int>(str.length())) {
            str = str.substr(0, spec.precision);
        }
        
        return apply_padding(str, spec);
    }
    
    static std::string format_default(const T& value, const FormatSpec& spec) {
        std::stringstream ss;
        ss << value;
        return apply_padding(ss.str(), spec);
    }
    
    static void format_binary(T value, std::stringstream& ss) {
        if (value == 0) {
            ss << "0";
            return;
        }
        
        std::string binary;
        while (value > 0) {
            binary = (value & 1 ? "1" : "0") + binary;
            value >>= 1;
        }
        ss << binary;
    }
    
    static std::string apply_padding(const std::string& str, const FormatSpec& spec) {
        if (spec.width <= 0 || static_cast<int>(str.length()) >= spec.width) {
            return str;
        }
        
        int padding = spec.width - str.length();
        std::string result;
        
        if (spec.left_align) {
            result = str + std::string(padding, spec.fill);
        } else {
            result = std::string(padding, spec.fill) + str;
        }
        
        return result;
    }
};

//--------------------------------------------------------------------------
// FORMAT STRING PARSER
//--------------------------------------------------------------------------
class FormatStringParser {
public:
    struct FormatItem {
        std::string literal;
        std::string spec;
        int arg_index = -1;
        bool is_placeholder = false;
    };
    
    static std::vector<FormatItem> parse(const std::string& format_str) {
        std::vector<FormatItem> items;
        std::string current_literal;
        
        for (size_t i = 0; i < format_str.length(); ++i) {
            if (format_str[i] == '{' && i + 1 < format_str.length()) {
                // Check for escaped brace
                if (format_str[i + 1] == '{') {
                    current_literal += '{';
                    i++; // Skip next brace
                    continue;
                }
                
                // Save current literal
                if (!current_literal.empty()) {
                    items.push_back({current_literal, "", -1, false});
                    current_literal.clear();
                }
                
                // Parse placeholder
                size_t end = i + 1;
                while (end < format_str.length() && format_str[end] != '}') {
                    end++;
                }
                
                if (end < format_str.length()) {
                    std::string placeholder = format_str.substr(i + 1, end - i - 1);
                    FormatItem item;
                    item.is_placeholder = true;
                    
                    // Parse argument index and spec
                    size_t colon_pos = placeholder.find(':');
                    if (colon_pos != std::string::npos) {
                        std::string index_str = placeholder.substr(0, colon_pos);
                        item.spec = placeholder.substr(colon_pos + 1);
                        item.arg_index = std::stoi(index_str);
                    } else {
                        item.arg_index = std::stoi(placeholder);
                    }
                    
                    items.push_back(item);
                    i = end;
                } else {
                    // Unclosed brace, treat as literal
                    current_literal += format_str[i];
                }
            } else if (format_str[i] == '}' && i + 1 < format_str.length() && format_str[i + 1] == '}') {
                current_literal += '}';
                i++; // Skip next brace
            } else {
                current_literal += format_str[i];
            }
        }
        
        // Add remaining literal
        if (!current_literal.empty()) {
            items.push_back({current_literal, "", -1, false});
        }
        
        return items;
    }
};

//--------------------------------------------------------------------------
// MAIN FORMAT FUNCTION
//--------------------------------------------------------------------------
template<typename... Args>
std::string format(const std::string& format_str, Args&&... args) {
    auto items = FormatStringParser::parse(format_str);
    std::string result;
    
    // Convert arguments to tuple for indexed access
    auto arg_tuple = std::make_tuple(std::forward<Args>(args)...);
    int arg_index = 0;
    
    for (const auto& item : items) {
        if (item.is_placeholder) {
            int index = item.arg_index >= 0 ? item.arg_index : arg_index++;
            if (index < sizeof...(args)) {
                FormatSpec spec = FormatSpec::parse(item.spec);
                std::string formatted = format_arg(arg_tuple, index, spec);
                result += formatted;
            } else {
                result += "{" + std::to_string(index) + "}";
            }
        } else {
            result += item.literal;
        }
    }
    
    return result;
}

//--------------------------------------------------------------------------
// ARGUMENT FORMATTING
//--------------------------------------------------------------------------
template<typename Tuple, size_t Index>
std::string format_arg_impl(const Tuple& tuple, const FormatSpec& spec) {
    using T = std::tuple_element_t<Index, Tuple>;
    return Formatter<T>::format(std::get<Index>(tuple), spec);
}

template<typename Tuple>
std::string format_arg(const Tuple& tuple, int index, const FormatSpec& spec) {
    // This is a simplified version - in practice, you'd need more sophisticated
    // template metaprogramming to handle arbitrary tuple indices
    if (index == 0) return format_arg_impl<Tuple, 0>(tuple, spec);
    if (index == 1) return format_arg_impl<Tuple, 1>(tuple, spec);
    if (index == 2) return format_arg_impl<Tuple, 2>(tuple, spec);
    if (index == 3) return format_arg_impl<Tuple, 3>(tuple, spec);
    if (index == 4) return format_arg_impl<Tuple, 4>(tuple, spec);
    if (index == 5) return format_arg_impl<Tuple, 5>(tuple, spec);
    if (index == 6) return format_arg_impl<Tuple, 6>(tuple, spec);
    if (index == 7) return format_arg_impl<Tuple, 7>(tuple, spec);
    if (index == 8) return format_arg_impl<Tuple, 8>(tuple, spec);
    if (index == 9) return format_arg_impl<Tuple, 9>(tuple, spec);
    
    return "{" + std::to_string(index) + "}";
}

//--------------------------------------------------------------------------
// CONVENIENCE FUNCTIONS
//--------------------------------------------------------------------------
template<typename... Args>
std::string format_to_string(const std::string& format_str, Args&&... args) {
    return format(format_str, std::forward<Args>(args)...);
}

template<typename... Args>
void print(const std::string& format_str, Args&&... args) {
    std::cout << format(format_str, std::forward<Args>(args)...);
}

template<typename... Args>
void println(const std::string& format_str, Args&&... args) {
    std::cout << format(format_str, std::forward<Args>(args)...) << std::endl;
}

//--------------------------------------------------------------------------
// SPECIALIZED FORMATTERS
//--------------------------------------------------------------------------
template<>
class Formatter<bool> {
public:
    static std::string format(const bool& value, const FormatSpec& spec) {
        std::string str = value ? "true" : "false";
        return FormatSpec::apply_padding(str, spec);
    }
};

template<>
class Formatter<char> {
public:
    static std::string format(const char& value, const FormatSpec& spec) {
        return std::string(1, value);
    }
};

template<>
class Formatter<const char*> {
public:
    static std::string format(const char* value, const FormatSpec& spec) {
        return Formatter<std::string>::format(std::string(value), spec);
    }
};

template<>
class Formatter<char*> {
public:
    static std::string format(char* value, const FormatSpec& spec) {
        return Formatter<std::string>::format(std::string(value), spec);
    }
};

//--------------------------------------------------------------------------
// MACROS FOR CONVENIENCE
//--------------------------------------------------------------------------
#define HATRIX_FORMAT(...) hatrix::format::format(__VA_ARGS__)
#define HATRIX_PRINT(...) hatrix::format::print(__VA_ARGS__)
#define HATRIX_PRINTLN(...) hatrix::format::println(__VA_ARGS__)

} // namespace format
} // namespace hatrix

#endif // HATRIX_FORMAT_HPP
