/*
 * gen_fixture_safetensors.c -- write a tiny F32 safetensors file for tests.
 * Tensors: "a" [2,3] = 1..6, "b" [4] = 7..10.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void put_u64(unsigned char *p, unsigned long long v) {
    for (int i = 0; i < 8; i++) p[i] = (unsigned char)((v >> (8*i)) & 0xFF);
}

int main(void) {
    const char *header =
        "{\"a\":{\"dtype\":\"F32\",\"shape\":[2,3],\"data_offsets\":[0,24]},"
        "\"b\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[24,40]},"
        "\"__metadata__\":{\"info\":\"fixture\"}}";
    size_t hlen = strlen(header);
    size_t total_header = 8 + hlen;
    size_t padded = (total_header + 7) & ~((size_t)7);
    size_t raw_off = padded;

    FILE *f = fopen("fixture.safetensors", "wb");
    if (!f) { fprintf(stderr, "open fail\n"); return 1; }

    unsigned char lenbuf[8];
    put_u64(lenbuf, (unsigned long long)hlen);
    fwrite(lenbuf, 1, 8, f);
    fwrite(header, 1, hlen, f);
    // pad to 8
    for (size_t i = total_header; i < padded; i++) fputc(0, f);

    float a[6] = {1,2,3,4,5,6};
    float b[4] = {7,8,9,10};
    fwrite(a, sizeof(float), 6, f);
    fwrite(b, sizeof(float), 4, f);
    fclose(f);
    printf("wrote fixture.safetensors (raw_off=%zu)\n", raw_off);
    return 0;
}
