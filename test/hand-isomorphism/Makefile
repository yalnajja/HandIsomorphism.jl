#
# Makefile
# original author: Kevin Waugh (waugh@cs.cmu.edu)
# date: April 7, 2013
#
# modified by: Yacine Al-Najjar (yalnajja@gmail.com)
# date: August 22, 2026
# changes: Added shared library (.so) build target for Julia FFI
#
# see README.md and LICENSE.txt for more details.
#
CFLAGS     ?= -std=c99 -Wall -O3 -DNDEBUG -fPIC
LDLIBS     ?=-lm

SRC        :=$(addprefix src/,deck.c hand_index.c)
OBJ        :=$(SRC:.c=.o)

LIB        :=libhand_index.so
TARGETS    :=src/check $(LIB)
TARGET_SRC :=$(addprefix src/,check-main.c)
TARGET_OBJ :=$(TARGET_SRC:.c=.o)

.PHONY: all clean check

all: $(TARGETS)

clean:
	rm -f $(TARGETS)
	rm -f $(TARGET_OBJ) $(OBJ)

check: src/check
	./src/check

# Rule for building the shared library (.so)
$(LIB): $(OBJ)
	$(CC) -shared $(LDFLAGS) -o $@ $^ $(LOADLIBES) $(LDLIBS)

src/check: src/check-main.o $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^ $(LOADLIBES) $(LDLIBS)

src/check-main.o src/hand_index.o: src/deck.h src/hand_index.h
src/deck.o: src/deck.h