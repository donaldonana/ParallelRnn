CFLAGS= -O3 -g3 -Wall
CC= gcc
SRC = src
LIB = lib
OBJ = obj
BIN = bin
OBJECTS = $(OBJ)/utilities.o  $(OBJ)/lstm.o  $(OBJ)/layers.o
HEADERS = $(SRC)/utilities.h  $(SRC)/lstm.h  $(SRC)/std_conf.h


# defines the set of configuration variables for the Makefile
include Makefile.config

ifdef LR
CONFIG_FLAGS += -DLR=$(LR)
endif

ifdef EPOCH
CONFIG_FLAGS += -DEPOCH=$(EPOCH)
endif

ifdef MINI_BATCH_SIZE
CONFIG_FLAGS += -DMINI_BATCH_SIZE=$(MINI_BATCH_SIZE)
endif

ifdef NUM_THREADS
CONFIG_FLAGS += -DNUM_THREADS=$(NUM_THREADS)
endif

ifdef MUTEX
CONFIG_FLAGS += -DMUTEX=$(MUTEX)
endif

ifdef HIDEN_SIZE
CONFIG_FLAGS += -DHIDEN_SIZE=$(HIDEN_SIZE)
endif
 
all : app test

$(OBJ)/layers.o: $(SRC)/layers.c $(HEADERS)
	$(CC) -c $(CONFIG_FLAGS) $(CFLAGS) $(SRC)/layers.c  -o  $(OBJ)/layers.o -lm -lpthread

$(OBJ)/utilities.o: $(SRC)/utilities.c $(HEADERS)
	$(CC) -c $(CONFIG_FLAGS) $(CFLAGS) $(SRC)/utilities.c  -o  $(OBJ)/utilities.o -lm -lpthread

$(OBJ)/lstm.o: $(SRC)/lstm.c $(HEADERS)
	$(CC) -c $(CONFIG_FLAGS) $(CFLAGS) $(SRC)/lstm.c  -o  $(OBJ)/lstm.o -lm -lpthread
 
$(OBJ)/app.o: $(SRC)/app.c $(HEADERS)
	$(CC) -c $(CONFIG_FLAGS) $(CFLAGS) $(SRC)/app.c  -o $(OBJ)/app.o -lm -lpthread

$(OBJ)/test.o: $(SRC)/test.c $(HEADERS)
	$(CC) -c $(CONFIG_FLAGS) $(CFLAGS) $(SRC)/test.c -o $(OBJ)/test.o -lm -lpthread

app: ${OBJECTS} $(OBJ)/app.o $(HEADERS)
	$(CC) -o ./app.exe ${OBJECTS} $(OBJ)/app.o -lm -lpthread

test: ${OBJECTS} $(OBJ)/test.o $(HEADERS)
	$(CC) -o $(BIN)/test.exe ${OBJECTS} $(OBJ)/test.o -lm -lpthread

clean: 
	rm -rf $(OBJ)/*.o
	rm -rf $(BIN)/*.exe

empty: 
	del /F /Q $(OBJ)\*.o
	del /F /Q $(BIN)\*.exe

run-test: test
	./$(BIN)/test.exe

run-app: app
	./app.exe
