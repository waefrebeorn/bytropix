#ifndef WUBU_SHEET_H
#define WUBU_SHEET_H

/* wubu_sheet.h -- Win98/XP Spreadsheet (Excel-like).
 *
 * A cell grid addressed in A1 notation (column letters + row number). Cells
 * hold either a literal number or a formula: `=<ref>`, `<ref><op><ref|num>`
 * where op is + - * / and a ref is another cell. Setting a value or formula
 * re-evaluates the whole sheet in dependency order (refs before dependents),
 * detecting circular references (a cell depending on itself, directly or via a
 * chain) which evaluate to an error sentinel. Deterministic, no stubs. */

#include <stddef.h>

#define WUBU_SH_MAX_ROWS 64
#define WUBU_SH_MAX_COLS 26        /* A..Z */
#define WUBU_SH_MAX_CELLS (WUBU_SH_MAX_ROWS * WUBU_SH_MAX_COLS)
#define WUBU_SH_ERR (-1e30)        /* error sentinel (e.g. circular ref, div/0) */

typedef struct {
    double value;           /* computed numeric value */
    char formula[32];       /* "" = literal; else "=<expr>" */
    int is_formula;
    int err;                /* 1 = error (circular / div0 / bad operand) */
    int circular;           /* 1 = part of a circular reference */
} wubu_sh_cell_t;

typedef struct {
    wubu_sh_cell_t cells[WUBU_SH_MAX_CELLS];
    int rows, cols;
} wubu_sheet_t;

/* Initialise a rows x cols sheet (1..MAX). Returns 0/-1. */
int wubu_sh_init(wubu_sheet_t *sh, int rows, int cols);

/* Parse an A1 ref ("A1","B12") into a cell index, or -1 if invalid/out of
 * bounds for this sheet. */
int wubu_sh_ref(const wubu_sheet_t *sh, const char *ref);

/* Set a literal numeric value at an A1 ref. Returns 0/-1. */
int wubu_sh_set(wubu_sheet_t *sh, const char *ref, double value);

/* Set a formula at an A1 ref (include the leading '='). Supports
 * `<ref><op><ref|num>` and bare `<ref>` and bare `<num>`. Returns 0/-1
 * (bad ref/syntax). Re-evaluates the sheet. */
int wubu_sh_set_formula(wubu_sheet_t *sh, const char *ref, const char *formula);

/* Re-evaluate every formula cell in dependency order (after bulk edits).
 * Returns 0, or the number of circular-reference cells encountered. */
int wubu_sh_recalc(wubu_sheet_t *sh);

/* Read a cell's computed value at an A1 ref (WUBU_SH_ERR on error).
 * Returns 0/-1 on bad ref. */
int wubu_sh_get(const wubu_sheet_t *sh, const char *ref, double *out);

/* Is the cell at `ref` part of a circular reference? 1/0/-1. */
int wubu_sh_is_circular(const wubu_sheet_t *sh, const char *ref);

#endif
