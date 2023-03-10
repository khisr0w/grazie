# Unity build

# Directories
SRC_DIR := src
BIN_DIR := binary
EXECUTABLE := grazie.exe
DEBUG_DIR := $(BIN_DIR)/debug
RELEASE_DIR := $(BIN_DIR)/release

# Compiler and flags
CC := cl
CFLAGS_COMMON := /EHa /nologo /FC /Zo /WX /W4 /Gm- /wd5208
CFLAGS_DEBUG := /Od /MTd /Z7 /Zo /DDEBUG
CFLAGS_RELEASE := /O2 /Oi /MT /DRELEASE

# Source files
# SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
SOURCES := $(SRC_DIR)/main.cpp

# Object files
# OBJECTS_DEBUG := $(patsubst $(SRC_DIR)/%.cpp,$(DEBUG_DIR)/%.obj,$(SOURCES))
# OBJECTS_RELEASE := $(patsubst $(SRC_DIR)/%.cpp,$(RELEASE_DIR)/%.obj,$(SOURCES))

OBJECTS_DEBUG := $(DEBUG_DIR)/main.obj
OBJECTS_RELEASE := $(RELEASE_DIR)/main.obj

# Executables
EXECUTABLE_DEUBG := $(DEBUG_DIR)/$(EXECUTABLE)
EXECUTABLE_RELEASE := $(RELEASE_DIR)/$(EXECUTABLE)

# Phony targets
.PHONY: all clean debug release

# Default target
all: debug release

# Debug target
debug: $(DEBUG_DIR) $(OBJECTS_DEBUG) $(EXECUTABLE_DEUBG)
	@echo Debug build complete...

$(DEBUG_DIR):
	@echo Starting debug build...
	@mkdir -p $(DEBUG_DIR)

$(DEBUG_DIR)/main.obj: $(SRC_DIR)/main.cpp
	@echo "    [Debug] Compiling Objects..."
	@$(CC) $(CFLAGS_COMMON) $(CFLAGS_DEBUG) /Fo$@ /c $<

# Link debug executable
$(EXECUTABLE_DEUBG): $(OBJECTS_DEBUG)
	@echo "    [Debug] Linking Objects for executable..."
	@$(CC) $(CFLAGS_COMMON) $(CFLAGS_DEBUG) /Fe$@ $^


# Release target
release: $(RELEASE_DIR) $(OBJECTS_RELEASE) $(EXECUTABLE_RELEASE)
	@echo Release build complete...

$(RELEASE_DIR):
	@echo Starting release build...
	@mkdir -p $(RELEASE_DIR)

$(RELEASE_DIR)/main.obj: $(SRC_DIR)/main.cpp
	@echo "    Compiling Objects..."
	@$(CC) $(CFLAGS_COMMON) $(CFLAGS_RELEASE) /Fo$@ /c $<

# Link release executable
$(EXECUTABLE_RELEASE): $(OBJECTS_RELEASE)
	@echo "    Linking Objects for executable..."
	@$(CC) $(CFLAGS_COMMON) $(CFLAGS_RELEASE) /Fe$@ $^

# Clean target
clean:
	@rm -rf $(BIN_DIR)/*
	@echo Cleaned!
