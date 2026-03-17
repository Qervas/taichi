# Makefile for gpu_mesh — GPU particle-to-mesh surface reconstruction
# Build: make
# Clean: make clean

NVCC      := nvcc
ARCH      := sm_80
CFLAGS    := -O3 -arch=$(ARCH)
TARGET    := gpu_mesh
SRC       := gpu_mesh.cu

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
