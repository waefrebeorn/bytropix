/*
 * wubu_symreg.h -- Symbolic regression: equation discovery from data (EE01).
 */
#ifndef WUBU_SYMREG_H
#define WUBU_SYMREG_H

#define WUBU_SYMREG_MAX_TERMS 32
#define WUBU_SYMREG_MAX_EXPR 256
#define WUBU_SYMREG_MAX_POP 200
#define WUBU_SYMREG_NDIM 15  /* sweep config dims */

typedef struct {
    int n_samples;
    int n_vars;
    const double *X;  /* [n_samples][n_vars] */
    const double *y;  /* [n_samples] */
} wubu_symreg_data_t;

typedef struct {
    char expr[WUBU_SYMREG_MAX_EXPR];  /* discovered equation string */
    double mse;
    int   complexity;  /* number of nodes */
    int   found;
} wubu_symreg_result_t;

/* Genetic-programming symbolic regression. Searches for expr minimizing MSE.
   Uses seeded PRNG (no global state). Returns best expression found. */
int wubu_symreg_fit(const wubu_symreg_data_t *data, unsigned seed,
                     int max_iters, wubu_symreg_result_t *out);

/* Evaluate a discovered expression on input vector x[n_vars].
   Supports: + - * / sin exp sqrt, vars x0..xN, constants. */
double wubu_symreg_eval(const char *expr, const double *x, int n_vars);

#endif