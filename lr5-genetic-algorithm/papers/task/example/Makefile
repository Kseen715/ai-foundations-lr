CC = gcc
ERRORS = -Wall
# -Wall
# -Wextra
# -Werror
# -Wpedantic
# -Wno-unused-variable
# -Wfatal-errors
CFLAGS = -c $(ERRORS) -fPIC -std=c11 -O3
# CFLAGS += -D _DEBUG
BUILD_DIR = build
EXE_NAME = backprop
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
RM = rm -Recurse -Force
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

OBJECTS = $(SOURCES:.c=$(OBJECT_EXTENSION))

all: $(EXE_NAME)

%$(OBJECT_EXTENSION): %.c build_dir
	$(CC) $(CFLAGS) $(C_INCLUDE_PATH) $(SYS_FLAGS) -c $< -o $(BUILD_DIR)/$@

$(EXE_NAME): build_dir $(OBJECTS)
	$(CC) $(addprefix $(BUILD_DIR)/,$(OBJECTS)) -o $(BUILD_DIR)/$(EXE_NAME)$(EXE_EXTENSION) $(C_LIBS)

run: $(EXE_NAME)
	cd $(BUILD_DIR) && $(EXE_NAME)$(EXE_EXTENSION) && cd ../

build_dir: platform
	@if not exist $(BUILD_DIR) mkdir $(BUILD_DIR)

platform:
	$(ECHO) $(SYS_MSG)