CC = g++
TARGET = main
OBJS = main.o Net.o Neuron.o TrainingData.o
${TARGET}: ${OBJS}
	${CC}  -o ${TARGET} $^
	./${TARGET} > output.txt
main.o: main.cpp TrainingData.h Net.h Neuron.h
	${CC}  -c main.cpp
Net.o: Net.cpp Net.h Neuron.h
	${CC}  -c Net.cpp
Neuron.o: Neuron.cpp Neuron.h Net.h errMsg.h
	${CC}  -c Neuron.cpp
TrainingData.o: TrainingData.cpp TrainingData.h
	${CC}  -c TrainingData.cpp
clean:
	rm -f ${TARGET} ${OBJS}
