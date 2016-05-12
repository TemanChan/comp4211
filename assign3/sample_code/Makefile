CC = g++
FLAGS = -std=c++11
TARGET = main
OBJS = grid.o main.o state.o
${TARGET}: ${OBJS}
	${CC} ${FLAGS} -o ${TARGET} $^
grid.o: grid.cpp grid.h state.h parameter.h
	${CC} ${FLAGS} -c grid.cpp
main.o: main.cpp grid.h state.h parameter.h
	${CC} ${FLAGS} -c main.cpp
state.o: state.cpp state.h parameter.h
	${CC} ${FLAGS} -c state.cpp
clean:
	rm -f ${TARGET} ${OBJS}
