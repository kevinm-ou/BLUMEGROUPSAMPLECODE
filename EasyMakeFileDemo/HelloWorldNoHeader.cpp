#include <iostream>
#include <iomanip>
//This file will not be linked via header to main
//Instead it will work by extern
void HelloDoerteGroup2(int someInt){
    std::cout<<"Hello group, how did you find out my favorite number is: "<<someInt<<std::endl;
}
