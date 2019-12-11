//
// Created by cmf on 2019/11/5.
//

#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv) {
//    int srch = 720, srcw = 1280;
//    int target_size = 1300;

    int srch = atoi(argv[1]), srcw = atoi(argv[2]);
    int target_size = atoi(argv[3]);

    int max_long_edge = srch >= srcw ? srch : srcw;
    int max_short_edge = srch <= srcw ? srch : srcw;

    float scale = target_size * 1.0 / max_long_edge;

    int targeth = srch*scale, targetw = srcw*scale;
    printf("max_long_edge = %d, scale = %f, targeth = %d, targetw = %d\r\n",
            max_long_edge, scale, targeth, targetw);
}