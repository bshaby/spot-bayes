# Build the RW-integration shared library (Shi's GSL closed-form + integrals).
#
# GSL is located via gsl-config if available, else via the active conda prefix
# ($CONDA_PREFIX), else system defaults. Override on the command line, e.g.:
#   make GSL_PREFIX=/path/to/gsl
#
# Produces rw_inte_cpp.so in this directory; the Python wrapper (rw_inte.py)
# searches here for it.

CXX ?= g++
CXXFLAGS ?= -O3 -std=c++11 -fPIC

# Resolve GSL include/lib dirs.
GSL_CONFIG := $(shell command -v gsl-config 2>/dev/null)
ifneq ($(GSL_CONFIG),)
  GSL_CFLAGS := $(shell gsl-config --cflags)
  GSL_LIBS   := $(shell gsl-config --libs)
else ifdef CONDA_PREFIX
  GSL_CFLAGS := -I$(CONDA_PREFIX)/include
  GSL_LIBS   := -L$(CONDA_PREFIX)/lib -lgsl -lgslcblas -lm
else
  GSL_CFLAGS :=
  GSL_LIBS   := -lgsl -lgslcblas -lm
endif

TARGET := rw_inte_cpp.so
SRC := rw_inte_cpp.cpp

.PHONY: all clean
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(GSL_CFLAGS) -shared $(SRC) -o $(TARGET) $(GSL_LIBS)
	@# embed rpath so the .so finds libgsl at runtime without LD_LIBRARY_PATH
	@if [ -n "$(CONDA_PREFIX)" ]; then \
	  patchelf --set-rpath $(CONDA_PREFIX)/lib $(TARGET) 2>/dev/null || true; \
	fi

clean:
	rm -f $(TARGET)
