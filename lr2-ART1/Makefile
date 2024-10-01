CC = gcc
ERRORS = -Wall
# -Wall
# -Wextra
# -Werror
# -Wpedantic
# -Wno-unused-variable
# -Wfatal-errors
CFLAGS = -c $(ERRORS) -fPIC -std=c11
# CFLAGS += -D _DEBUG
BUILD_DIR = build
EXE_NAME = rcmnd
SOURCES = $(wildcard *.c)
SOURCES += $(wildcard *.cpp)
HEADERS = $(wildcard *.h)
HEADERS += $(wildcard *.hpp)
C_INCLUDE_PATH = -I.
C_LIBS =

TARGET = 
ifeq ($(OS),Windows_NT)
TARGET = windows
else
TARGET = linux
endif

ifeq ($(TARGET),windows)
RM = rm
COPY = copy
SYS_FLAGS = -D _WIN32
EXE_EXTENSION = .exe
ARCHIVE_EXTENSION = .lib
OBJECT_EXTENSION = .obj
ECHO = @echo
SYS_MSG = "Windows_NT detected!"
C_LIBS +=
else
RM = rm -rf
COPY = cp
SYS_FLAGS = -D  
EXE_EXTENSION =
ARCHIVE_EXTENSION = .a
OBJECT_EXTENSION = .o
ECHO = @echo
SYS_MSG = "Linux detected!"
C_LIBS +=
endif


build: build_dir platform $(SOURCES)
# Compile the sources
	$(CC) $(CFLAGS) $(C_INCLUDE_PATH) $(SYS_FLAGS) $(SOURCES) -o $(BUILD_DIR)/$(EXE_NAME)$(OBJECT_EXTENSION) $(C_LIBS)
# Link the object files
	$(CC) $(BUILD_DIR)/$(EXE_NAME)$(OBJECT_EXTENSION) -o $(BUILD_DIR)/$(EXE_NAME)$(EXE_EXTENSION) $(C_LIBS)


run: build
	$(BUILD_DIR)/$(EXE_NAME)$(EXE_EXTENSION)

build_dir:
	@if not exist $(BUILD_DIR) mkdir $(BUILD_DIR)

platform:
	$(ECHO) $(SYS_MSG)