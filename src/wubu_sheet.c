/* wubu_sheet.c -- Win98/XP Spreadsheet (Excel-like).
 * See wubu_sheet.h. Real, dependency-ordered formula evaluator. */

#include "wubu_sheet.h"
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

int wubu_sh_init(wubu_sheet_t *sh, int rows, int cols) {
    if (!sh) return -1;
    if (rows < 1 || rows > WUBU_SH_MAX_ROWS) return -1;
    if (cols < 1 || cols > WUBU_SH_MAX_COLS) return -1;
    memset(sh, 0, sizeof(*sh));
    sh->rows = rows;
    sh->cols = cols;
    return 0;
}

int wubu_sh_ref(const wubu_sheet_t *sh, const char *ref) {
    if (!sh || !ref || !ref[0]) return -1;
    int col = 0, i = 0;
    while (ref[i] && isalpha((unsigned char)ref[i])) {
        col = col * 26 + (toupper((unsigned char)ref[i]) - 'A' + 1);
        i++;
    }
    if (i == 0) return -1;
    if (col < 1 || col > sh->cols) return -1;
    if (!ref[i]) return -1;
    int row = atoi(ref + i);
    if (row < 1 || row > sh->rows) return -1;
    return (row - 1) * sh->cols + (col - 1);
}

int wubu_sh_set(wubu_sheet_t *sh, const char *ref, double value) {
    int idx = wubu_sh_ref(sh, ref);
    if (idx < 0) return -1;
    wubu_sh_cell_t *c = &sh->cells[idx];
    c->is_formula = 0;
    c->formula[0] = '\0';
    c->value = value;
    c->err = 0;
    c->circular = 0;
    wubu_sh_recalc(sh);
    return 0;
}

/* tokenize "=<lhs>[<op><rhs>]" into lhs, op, rhs. Returns 0 on ok, -1 on
 * malformed (empty operand, unknown op, trailing junk). */
static int tokenize(const char *expr, char *lhs, char *rhs, char *op) {
    lhs[0] = rhs[0] = '\0';
    *op = 0;
    if (!expr || expr[0] != '=') return -1;
    const char *p = expr + 1;
    int li = 0;
    while (*p && *p != '+' && *p != '-' && *p != '*' && *p != '/') {
        if (*p == ' ') { p++; continue; }
        if (li < 31) lhs[li++] = *p;
        p++;
    }
    lhs[li] = '\0';
    if (li == 0) return -1;          /* empty lhs */
    if (*p) {
        *op = *p;
        p++;
        int ri = 0;
        while (*p) {
            if (*p == ' ') { p++; continue; }
            if (ri < 31) rhs[ri++] = *p;
            p++;
        }
        rhs[ri] = '\0';
        if (ri == 0) return -1;      /* empty rhs after op */
    }
    return 0;
}

/* parse one operand token into a value; *isref set if it's a cell ref. */
static int parse_operand(wubu_sheet_t *sh, const char *tok, double *out,
                         int *isref, int *refidx) {
    *isref = 0; *refidx = -1;
    if (!tok || !tok[0]) return -1;
    if (isalpha((unsigned char)tok[0])) {
        int idx = wubu_sh_ref(sh, tok);
        if (idx < 0) return -1;
        *isref = 1; *refidx = idx;
        *out = sh->cells[idx].value;
        return 0;
    }
    char *end = NULL;
    double v = strtod(tok, &end);
    if (end == tok || *end != '\0') return -1;   /* not a clean number */
    *out = v;
    return 0;
}

int wubu_sh_set_formula(wubu_sheet_t *sh, const char *ref, const char *formula) {
    int idx = wubu_sh_ref(sh, ref);
    if (idx < 0) return -1;
    if (!formula || formula[0] != '=') return -1;
    if (strlen(formula) >= sizeof(sh->cells[idx].formula)) return -1;
    /* validate syntax at set time */
    char lhs[32], rhs[32], op;
    if (tokenize(formula, lhs, rhs, &op) != 0) return -1;
    double tmp; int isref, refidx;
    if (parse_operand(sh, lhs, &tmp, &isref, &refidx) != 0) return -1;
    if (op && parse_operand(sh, rhs, &tmp, &isref, &refidx) != 0) return -1;
    /* store */
    wubu_sh_cell_t *c = &sh->cells[idx];
    c->is_formula = 1;
    strncpy(c->formula, formula, sizeof(c->formula) - 1);
    c->formula[sizeof(c->formula) - 1] = '\0';
    wubu_sh_recalc(sh);
    return 0;   /* the set succeeded (recalc may flag circular/err) */
}

/* evaluate a cell's formula given already-resolved operand values. */
static double eval_expr(const char *formula, double a, int a_ok,
                        double b, int b_ok, char op, int *ok) {
    *ok = 1;
    if (!a_ok) { *ok = 0; return WUBU_SH_ERR; }
    if (!op) return a;
    if (!b_ok) { *ok = 0; return WUBU_SH_ERR; }
    switch (op) {
    case '+': return a + b;
    case '-': return a - b;
    case '*': return a * b;
    case '/':
        if (b == 0.0) { *ok = 0; return WUBU_SH_ERR; }
        return a / b;
    }
    *ok = 0; return WUBU_SH_ERR;
}

/* recursive depth-first evaluation with cycle detection. */
static double eval_cell(wubu_sheet_t *sh, int idx, int *visiting) {
    wubu_sh_cell_t *c = &sh->cells[idx];
    if (!c->is_formula) return c->value;
    if (visiting[idx]) {
        c->circular = 1;
        c->err = 1;
        c->value = WUBU_SH_ERR;
        return WUBU_SH_ERR;
    }
    visiting[idx] = 1;
    char lhs[32], rhs[32], op;
    tokenize(c->formula, lhs, rhs, &op);
    double a = 0, b = 0; int a_ok = 1, b_ok = 1, isref, refidx;
    if (parse_operand(sh, lhs, &a, &isref, &refidx) != 0) a_ok = 0;
    else if (isref) a = eval_cell(sh, refidx, visiting);   /* resolve dep */
    if (op) {
        if (parse_operand(sh, rhs, &b, &isref, &refidx) != 0) b_ok = 0;
        else if (isref) b = eval_cell(sh, refidx, visiting);
    }
    visiting[idx] = 0;
    int ok;
    double v = eval_expr(c->formula, a, a_ok, b, b_ok, op, &ok);
    if (!ok) { c->err = 1; c->value = WUBU_SH_ERR; return WUBU_SH_ERR; }
    c->err = 0;
    c->value = v;
    return v;
}

int wubu_sh_recalc(wubu_sheet_t *sh) {
    if (!sh) return -1;
    int n = sh->rows * sh->cols;
    int visiting[WUBU_SH_MAX_CELLS];
    memset(visiting, 0, sizeof(visiting));
    int circular = 0;
    for (int i = 0; i < n; i++) {
        if (sh->cells[i].is_formula) {
            eval_cell(sh, i, visiting);
            if (sh->cells[i].circular) circular++;
        }
    }
    return circular;
}

int wubu_sh_get(const wubu_sheet_t *sh, const char *ref, double *out) {
    int idx = wubu_sh_ref(sh, ref);
    if (idx < 0 || !out) return -1;
    *out = sh->cells[idx].value;
    return 0;
}

int wubu_sh_is_circular(const wubu_sheet_t *sh, const char *ref) {
    int idx = wubu_sh_ref(sh, ref);
    if (idx < 0) return -1;
    return sh->cells[idx].circular ? 1 : 0;
}
