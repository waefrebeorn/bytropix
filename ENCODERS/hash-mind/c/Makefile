# Makefile — HashMind C Training System
# Targets:
#   make          — build train (with -O3)
#   make debug    — build with -O0 -g (debugging)
#   make test     — build and run training
#   make cuda     — build with CUDA support
#   make clean    — remove build artifacts
#   make dist     — clean build

CC = gcc
CFLAGS = -O3 -Wall -Wextra -Wno-unused-parameter -march=native -ffast-math
LDFLAGS = -lm
INCDIR = -Iinclude
SRCDIR = src
BUILDDIR = build

SRC = $(SRCDIR)/tokenizer.c $(SRCDIR)/rolling_hash.c $(SRCDIR)/hashmind_model.c $(SRCDIR)/hashmind_data.c $(SRCDIR)/train.c
OBJ = $(SRC:$(SRCDIR)/%.c=$(BUILDDIR)/%.o)
TARGET = train

# CUDA (optional — RTX 5050 = sm_89)
NVCC = nvcc
CUDA_FLAGS = -O3 -arch=sm_89

.PHONY: all debug test cuda clean dist

all: $(TARGET)

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(BUILDDIR)/%.o: $(SRCDIR)/%.c $(wildcard include/*.h) | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCDIR) -c $< -o $@

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

debug: CFLAGS = -O0 -g -Wall -Wextra -Wno-unused-parameter
debug: clean $(TARGET)

test: $(TARGET)
	./$(TARGET) --lr 0.001 --epochs 5 --gen-every 1000

cuda: | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCDIR) -c $(SRCDIR)/tokenizer.c -o $(BUILDDIR)/tokenizer.o $(LDFLAGS)
	$(CC) $(CFLAGS) $(INCDIR) -c $(SRCDIR)/rolling_hash.c -o $(BUILDDIR)/rolling_hash.o $(LDFLAGS)
	$(CC) $(CFLAGS) $(INCDIR) -c $(SRCDIR)/hashmind_model.c -o $(BUILDDIR)/hashmind_model.o $(LDFLAGS)
	$(CC) $(CFLAGS) $(INCDIR) -c $(SRCDIR)/hashmind_data.c -o $(BUILDDIR)/hashmind_data.o $(LDFLAGS)
	$(CC) $(CFLAGS) $(INCDIR) -c $(SRCDIR)/train.c -o $(BUILDDIR)/train.o $(LDFLAGS)
	$(NVCC) $(CUDA_FLAGS) -c ggml-cuda/wubu-cuda.cu -o $(BUILDDIR)/wubu-cuda.o
	$(CC) $(CFLAGS) -o $(TARGET) $(BUILDDIR)/*.o $(LDFLAGS)

clean:
	rm -rf $(BUILDDIR) $(TARGET)

dist: clean all
