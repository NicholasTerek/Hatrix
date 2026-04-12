// Hatrix
// Copyright (c) 2025 Hatrix contributors
// Licensed under the MIT License. See LICENSE for details.

#pragma once

#if defined(_WIN32) || defined(__CYGWIN__)
#if defined(HATRIX_BUILDING_LIBRARY)
#define HATRIX_EXPORT __declspec(dllexport)
#else
#define HATRIX_EXPORT __declspec(dllimport)
#endif
#else
#define HATRIX_EXPORT
#endif
