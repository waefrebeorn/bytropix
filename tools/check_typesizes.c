
#include <stdio.h>
#include "ggml.h"

int main() {
    printf("Q5_K type_size = %ld\n", (long)ggml_type_size(GGML_TYPE_Q5_K));
    printf("Q5_K blck_size = %ld\n", (long)ggml_blck_size(GGML_TYPE_Q5_K));
    printf("Q6_K type_size = %ld\n", (long)ggml_type_size(GGML_TYPE_Q6_K));
    printf("Q6_K blck_size = %ld\n", (long)ggml_blck_size(GGML_TYPE_Q6_K));
    printf("IQ2_XXS type_size = %ld\n", (long)ggml_type_size(GGML_TYPE_IQ2_XXS));
    printf("IQ2_XXS blck_size = %ld\n", (long)ggml_blck_size(GGML_TYPE_IQ2_XXS));
    printf("IQ3_XXS type_size = %ld\n", (long)ggml_type_size(GGML_TYPE_IQ3_XXS));
    printf("IQ3_XXS blck_size = %ld\n", (long)ggml_blck_size(GGML_TYPE_IQ3_XXS));
    printf("Q4_K type_size = %ld\n", (long)ggml_type_size(GGML_TYPE_Q4_K));
    printf("Q4_K blck_size = %ld\n", (long)ggml_blck_size(GGML_TYPE_Q4_K));
    return 0;
}
