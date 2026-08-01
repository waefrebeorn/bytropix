/*
 * wubu_verify.h -- C11 type-check + invariant assertion gate (AX09).
 */
#ifndef WUBU_VERIFY_H
#define WUBU_VERIFY_H
#define WUBU_VERIFY_MAX_ASSERTS 256
typedef struct {
    int n_asserts;
    int n_passed;
    int n_failed;
} wubu_verify_t;
int  wubu_verify_init(wubu_verify_t *v);
int  wubu_verify_assert_int(wubu_verify_t *v, int cond, const char *expr_str);
int  wubu_verify_assert_ptr(wubu_verify_t *v, const void *ptr, const char *expr_str);
int  wubu_verify_assert_range(wubu_verify_t *v, long val, long lo, long hi, const char *expr_str);
int  wubu_verify_all_passed(const wubu_verify_t *v);
int  wubu_verify_count(const wubu_verify_t *v, int *out_passed, int *out_failed);
#endif
