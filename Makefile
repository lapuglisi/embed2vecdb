CPP = g++
LD = g++

SOURCES = main.cpp app-llama.cpp utils.cpp llama-utils.cpp
OBJECTS = $(SOURCES:.cpp=.o)

DEVLIBS_ROOT = /mnt/storage/dev/libs

INCLUDES = -I./ -I./qdrant -I/mnt/storage/ggml-org/llama.cpp/include -I$(DEVLIBS_ROOT)/include

CPPFLAGS = $(INCLUDES) -O2 -pipe -march=native -ggdb -std=c++17

LDFLAGS = -L/mnt/storage/ggml-org/llama.cpp/lib -lllama

TARGET = embed2vecdb

all: $(TARGET)

debug:
	@$(MAKE) CPPFLAGS="$(CPPFLAGS) -D_DEBUG" all

$(TARGET): $(OBJECTS) $(QDRANT_OBJS)
	$(LD) -o $(TARGET) $(LDFLAGS) $(OBJECTS) $(QDRANT_OBJS)

.cpp.o:
	$(CPP) $(CPPFLAGS) -c $< -o $@

.cc.o:
	$(CPP) $(CPPFLAGS) -c $< -o $@

clean:
	@rm -fv $(OBJECTS)

purge: clean
	@rm -fv $(TARGET)

