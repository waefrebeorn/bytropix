/* wubu_fmt.h -- the format-constraint reward checker (the Atropos
 * Answer-Format env doctrine): the binary format rewards decoupled from
 * the semantics. Constraint types: JSON-valid, delimiter-enforced
 * (<think>/</think>), length-capped, prefix-exact. The verifiable
 * format reward for the trajectory-level GRPO. */
#ifndef WUBU_FMT_H
#define WUBU_FMT_H

enum {
    WUBU_FMT_JSON = 0,     /* the output parses as a JSON object/array */
    WUBU_FMT_THINK,        /* <think>...</think> strictly opened+closed */
    WUBU_FMT_LEN_MAX,      /* the output length <= limit */
    WUBU_FMT_LEN_MIN,      /* the output length >= limit */
    WUBU_FMT_PREFIX        /* the output starts with the given prefix */
};

/* Check one constraint on the output.
 * type: WUBU_FMT_*; out: the output text; limit: the numeric limit
 * (lengths) or 0; extra: the string argument (the prefix for PREFIX).
 * Returns 1 when the format holds, 0 otherwise. */
int wubu_fmt_check(int type, const char *out, int limit, const char *extra);

/* The combined format reward: the fraction of the constraints held
 * (each of the n types applied in order). */
float wubu_fmt_reward(const int *types, int n, const char *out,
                      const int *limits, const char **extras);

#endif
