#include <stdio.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <stdlib.h>

int main(int argc,char* argv[]){
    if(argc!=2){
        char s[30];
        sprintf(s,"使用方法: %s 文件名\n",argv[0]);
        fputs(s,stderr);
        exit(1);
    }

    struct stat statbuf;

    int fd = open(argv[1],O_RDWR);
    stat(argv[1],&statbuf);
    // 获取文件大小
    int length = statbuf.st_size;

    void* bufp = mmap(NULL,length,PROT_READ,MAP_SHARED,fd,0);
    //char* s=bufp;
    //printf("bupf:%s",s);
    int len = write(1,bufp,length);

    close(fd);
    munmap(bufp,length);
    return 0;
}
