#include <iostream>

#ifndef LOGGER
#define LOGGER
enum LogLevel{
    FATAL,
    WARNING,
    INFO,
    DEBUG,
};

LogLevel logLevel = LogLevel::INFO;

void LoggerPrint(std::string message, LogLevel level){

    if(logLevel >= level){
        std::cout << message;
    }

}

#endif