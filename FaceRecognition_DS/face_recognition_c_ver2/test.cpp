#include<bits/stdc++.h>

static bool check_nv_facemask(std::string str){
    size_t found = str.find("facemask");
    if (found != std::string::npos)
        return true;
    return false;
}
int main()
{
       std::cout<<check_nv_facemask("hao");
    std::cout<<check_nv_facemask("hao_facemask");
    return 0;

 }