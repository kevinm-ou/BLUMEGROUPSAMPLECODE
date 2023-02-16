#include <iostream>
#include <iomanip>
#include "HelloWorld.h"
//Two calls to outside subroutines
//Only one has a header call
extern void HelloDoerteGroup2(int someInt);

int main(int argc, char *argv[])
{
    HelloDoerteGroup();
    HelloDoerteGroup2(241);
    return 0;
}