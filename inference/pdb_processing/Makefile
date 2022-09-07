CC = g++
#LIB = /cs/labs/dina/dina/libs/gamb
#DLIB = /cs/labs/dina/dina/libs/DockingLib
LIB = /cs/labs/dina/punims/gamb
DLIB = /cs/labs/dina/punims/DockingLib



CCFLAGS = -std=c++17 -I$(LIB) -I$(DLIB) -Wall -O2
LDFLAGS = -static -L$(LIB) -L$(DLIB) -ldockingLib -lgamb++ -lstdc++ -lstdc++fs -lm -lboost_program_options



# User defined classes and modules. (no file suffixes)

CLASSES = generatePDB

# Prepare object and source file list using pattern substitution func.
ALL  = $(CLASSES)
OBJS = $(patsubst %, %.o,  $(ALL))
SRCS = $(patsubst %, %.cc, $(ALL))

TARGET = generatePDB

$(TARGET): $(OBJS)
	$(CC) $(OBJS) $(LDFLAGS)  -o $(TARGET)

%.o: %.cc
	$(CC) $(CCFLAGS) -c $*.cc

clean:
	/bin/rm -f *.o *~ \#* core

depend:
	makedepend -- $(CCFLAGS) -- $(SRCS)
# DO NOT DELETE THIS LINE -- make depend depends on it.
