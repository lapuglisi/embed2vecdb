CPP = g++
LD = g++

SOURCES = main.cpp app-llama.cpp utils.cpp llama-utils.cpp $(wildcard qdrant/*.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

LLAMACPP_ROOT = /mnt/development/ggml-org/llama.cpp
DEVLIBS_ROOT = /mnt/storage/dev/libs

INCLUDES = -I./include -I./qdrant -I$(LLAMACPP_ROOT)/include -I$(DEVLIBS_ROOT)/include

CPPFLAGS = $(INCLUDES) -O2 -pipe -march=native -ggdb -std=c++17

LDFLAGS = -ggdb -L$(LLAMACPP_ROOT)/lib -L$(DEVLIBS_ROOT)/lib -lllama -lcurl -luuid

TARGET = embed2vecdb

all: $(TARGET)

debug:
	@$(MAKE) CPPFLAGS="$(CPPFLAGS) -D_DEBUG" all

$(TARGET): $(OBJECTS)
	$(LD) -o $(TARGET) $(LDFLAGS) $(OBJECTS)

.cpp.o:
	$(CPP) $(CPPFLAGS) -c $< -o $@

.cc.o:
	$(CPP) $(CPPFLAGS) -c $< -o $@

clean:
	@rm -fv $(OBJECTS)

purge: clean
	@rm -fv $(TARGET)
