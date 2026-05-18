#include "wubu_model.h"
#include <stdio.h>

int main() {
    wubu_model_t mdl;
    printf("Calling wubu_model_init...\n");
    fflush(stdout);
    if (!wubu_model_init(&mdl, "/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf")) {
        printf("FAILED\n");
        return 1;
    }
    printf("SUCCESS\n");
    wubu_model_free(&mdl);
    return 0;
}
