#include <exception>
#include <cstring>

class MyException: public exception
{
public:
	MyException(const char *msg){
		strcpy(errMsg, msg);
	}
	~MyException() throw(){
		delete errMsg;
		errMsg = NULL;
	}
	virtual const char * what() const throw(){
		return errMsg;
	}
private:
	char *errMsg;
};
