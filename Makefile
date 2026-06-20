CXX = g++
CXXFLAGS = -std=c++17 -Wall -Iinclude -g

TARGET = few_body_nc
SRC_DIR = src
BUILD_DIR = build


#SRCS = src/main.cpp src/body.cpp src/solver.cpp src/io.cpp src/leapfrog.cpp src/globals.cpp
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
SRCS += $(wildcard $(SRC_DIR)/io/*.cpp)
SRCS += $(wildcard $(SRC_DIR)/integrators/*.cpp)
SRCS += $(wildcard $(SRC_DIR)/integrators-helper/*.cpp)
SRCS += $(wildcard $(SRC_DIR)/integrators-helper/hb15/*.cpp)
SRCS += $(wildcard $(SRC_DIR)/numerics/*.cpp)
SRCS += $(wildcard $(SRC_DIR)/dynamics/*.cpp) 
SRCS += $(wildcard $(SRC_DIR)/core/*.cpp)
SRCS += $(wildcard $(SRC_DIR)/analysis/*.cpp)
#SRCS+= src/io/io.cpp src/io/output.cpp src/io/input.cpp

OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))

OBJS := $(OBJS:io/%=$(BUILD_DIR)/io/%)
OBJS := $(OBJS:integrators/%=$(BUILD_DIR)/integrators/%)
OBJS := $(OBJS:propagator/%=$(BUILD_DIR)/propagator/%)
OBJS := $(OBJS:numerics/%=$(BUILD_DIR)/numerics/%)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@if not exist "$(dir $@)" mkdir "$(dir $@)" 2>nul || true
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	@if exist $(BUILD_DIR) rmdir /S /Q $(BUILD_DIR)
	@if exist $(TARGET).exe del /Q $(TARGET).exe
	@if exist *.csv del /Q *.csv
	@if exist benchmark_plots rmdir /S /Q benchmark_plots
	@if exist python\__pycache__ rmdir /S /Q python\__pycache__
