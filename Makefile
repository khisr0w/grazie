# Unity build

# Directories
SRC_DIR := src
BIN_DIR := binary
EXECUTABLE := grazie
DEBUG_DIR := $(BIN_DIR)/debug
RELEASE_DIR := $(BIN_DIR)/release

# Compiler and flags
CC := clang
CFLAGS_COMMON := -fno-caret-diagnostics -Wno-null-dereference #/EHa /nologo /FC /Zo /WX /W4 /Gm- /wd5208 /wd4505
CFLAGS_DEBUG := -g #/Od /MTd /Z7 /Zo /DDEBUG
CFLAGS_RELEASE := #/O2 /Oi /MT /DRELEASE

ifeq ($(OS),Windows_NT)
EXECUTABLE := grazie.exe
CC := cl
CFLAGS_COMMON := /EHa /nologo /FC /Zo /WX /W4 /Gm- /wd5208 /wd4505 /wd4127 /DGRAZIE_PLT_WIN /DGRAZIE_ASSERT
CFLAGS_DEBUG := /Od /MTd /Z7 /Zo /DGRAZIE_DEBUG
CFLAGS_RELEASE := /O2 /Oi /MT /DGRAZIE_RELEASE
endif

# Source files
# SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
SOURCES := $(SRC_DIR)/main.c

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

$(DEBUG_DIR)/main.obj: $(SRC_DIR)/main.c
	@echo "    [Debug] Compiling Objects..."

ifeq ($(OS),Windows_NT)
	@$(CC) $(CFLAGS_COMMON) $(CFLAGS_DEBUG)  /Fo$@ /c $<
else
	@$(CC) $(CFLAGS_COMMON) $(CFLAGS_DEBUG) -c $< -o $@ 
endif

# Link debug executable
$(EXECUTABLE_DEUBG): $(OBJECTS_DEBUG)
	@echo "    [Debug] Linking Objects for executable..."
ifeq ($(OS),Windows_NT)
	@$(CC) $(CFLAGS_COMMON) $(CFLAGS_DEBUG) /Fe$@ $^
else
	@$(CC) $(CFLAGS_COMMON) $(CFLAGS_DEBUG) $^ -o $@
endif


# Release target
release: $(RELEASE_DIR) $(OBJECTS_RELEASE) $(EXECUTABLE_RELEASE)
	@echo Release build complete...

$(RELEASE_DIR):
	@echo Starting release build...
	@mkdir -p $(RELEASE_DIR)

$(RELEASE_DIR)/main.obj: $(SRC_DIR)/main.c
	@echo "    Compiling Objects..."
ifeq ($(OS),Windows_NT)
	@$(CC) $(CFLAGS_COMMON) $(CFLAGS_RELEASE) /Fo$@ /c $<
else
	@$(CC) $(CFLAGS_COMMON) $(CFLAGS_RELEASE) -c $< -o $@ 
endif

# Link release executable
$(EXECUTABLE_RELEASE): $(OBJECTS_RELEASE)
	@echo "    Linking Objects for executable..."
ifeq ($(OS),Windows_NT)
	@$(CC) $(CFLAGS_COMMON) $(CFLAGS_RELEASE) /Fe$@ $^
else
	@$(CC) $(CFLAGS_COMMON) $(CFLAGS_RELEASE) $^ -o $@
endif

# Clean target
clean:
	@rm -rf $(BIN_DIR)/*
	@echo Cleaned!

# =======================| Test script |========================

TEST_BIN_DIR := $(BIN_DIR)/tests
TEST_SRC_DIR := tests
TEST_SRC := $(wildcard $(TEST_SRC_DIR)/*.c)
TEST_OBJ := $(patsubst $(TEST_SRC_DIR)/%.c, $(TEST_BIN_DIR)/%.o, $(TEST_SRC))

TEST_EXE := $(patsubst $(TEST_BIN_DIR)/%.o, $(TEST_BIN_DIR)/%, $(TEST_OBJ))

ifeq ($(OS),Windows_NT)
TEST_EXE := $(patsubst $(TEST_BIN_DIR)/%.o, $(TEST_BIN_DIR)/%.exe, $(TEST_OBJ))
endif

test: $(TEST_BIN_DIR) $(TEST_EXE)

$(TEST_BIN_DIR):
	@echo Starting test...
	@mkdir -p $@

$(TEST_EXE): $(TEST_OBJ)
ifeq ($(OS),Windows_NT)
	@$(CC) $(CFLAGS_COMMON) $(CFLAGS_DEBUG) /Fe$@ $^
else
	@$(CC) $(CFLAGS_COMMON) $(CFLAGS_DEBUG) $^ -o $@
endif

$(TEST_OBJ): $(TEST_SRC)
ifeq ($(OS),Windows_NT)
	@$(CC) $(CFLAGS_COMMON) $(CFLAGS_RELEASE) /Fo$@ /c $<
else
	@$(CC) $(CFLAGS_COMMON) $(CFLAGS_RELEASE) -c $< -o $@ 
endif

# =============================================================

