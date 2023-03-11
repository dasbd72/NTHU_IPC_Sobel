NVFLAGS := -std=c++11 -O3 -Xptxas=-v -arch=sm_61 --use_fast_math
LDFLAGS := -lpng -lz
TARGET := hw3

.PHONY: all
all: $(TARGET)

.PHONY: clean
clean:
	rm -f $(TARGET)

$(TARGET): hw3.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?


