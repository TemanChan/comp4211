CC = g++
FLAGS = -std=c++11
TARGET = main
OBJS = main.o Net.o Neuron.o TrainingData.o
${TARGET}: ${OBJS}
	${CC} ${FLAGS} -o ${TARGET} $^
main.o: main.cpp TrainingData.h Net.h Neuron.h
	${CC} ${FLAGS} -c main.cpp
Net.o: Net.cpp Net.h Neuron.h
	${CC} ${FLAGS} -c Net.cpp
Neuron.o: Neuron.cpp Neuron.h errMsg.h
	${CC} ${FLAGS} -c Neuron.cpp
TrainingData.o: TrainingData.cpp TrainingData.h
	${CC} ${FLAGS} -c TrainingData.cpp
clean:
	rm -f ${TARGET} ${OBJS}
