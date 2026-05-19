#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define D_MODEL 2048

int main(int argc, char **argv) {
    const char *ref_dir = argc > 1 ? argv[1] : "/tmp/dump_layers_ref";
    const char *our_dir = argc > 2 ? argv[2] : "/tmp/dump_layers_our";
    int n_layers = argc > 3 ? atoi(argv[3]) : 40;
    
    float all_ref[D_MODEL], all_our[D_MODEL];
    double total_dot = 0, total_n1 = 0, total_n2 = 0;
    
    for (int l = 0; l < n_layers; l++) {
        char ref_path[512], our_path[512];
        snprintf(ref_path, sizeof(ref_path), "%s/ref_layer_%d.bin", ref_dir, l);
        snprintf(our_path, sizeof(our_path), "%s/our_layer_%d.bin", our_dir, l);
        
        FILE *f_ref = fopen(ref_path, "rb");
        FILE *f_our = fopen(our_path, "rb");
        if (!f_ref || !f_our) {
            printf("L%02d: FILE MISSING (%s)\n", l, !f_ref ? ref_path : our_path);
            if (f_ref) fclose(f_ref);
            continue;
        }
        
        size_t nr = fread(all_ref, sizeof(float), D_MODEL, f_ref);
        size_t no = fread(all_our, sizeof(float), D_MODEL, f_our);
        fclose(f_ref); fclose(f_our);
        
        if (nr != D_MODEL || no != D_MODEL) {
            printf("L%02d: READ ERROR (got %zu/%zu floats)\n", l, nr, no);
            continue;
        }
        
        double dot = 0, n1 = 0, n2 = 0, max_diff = 0;
        for (int i = 0; i < D_MODEL; i++) {
            dot += (double)all_ref[i] * (double)all_our[i];
            n1  += (double)all_ref[i] * (double)all_ref[i];
            n2  += (double)all_our[i] * (double)all_our[i];
            double diff = fabs((double)all_ref[i] - (double)all_our[i]);
            if (diff > max_diff) max_diff = diff;
        }
        double cos = dot / (sqrt(n1) * sqrt(n2) + 1e-30);
        printf("L%02d: cos=%7.4f max_diff=%.6f\n", l, cos, max_diff);
        
        total_dot += dot;
        total_n1  += n1;
        total_n2  += n2;
    }
    
    double total_cos = total_dot / (sqrt(total_n1) * sqrt(total_n2) + 1e-30);
    printf("---\nOVERALL: cos=%7.4f\n", total_cos);
    return 0;
}
