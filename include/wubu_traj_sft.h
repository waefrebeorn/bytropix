/* wubu_traj_sft.h -- the trajectory -> masked-observation SFT converter
 * (the Hermes 69%-output-token + Orchard obs-masking doctrine): a raw
 * agent trajectory (think / tool_call / tool_response / observation
 * cycles) becomes an ordered segment list where ONLY the assistant-side
 * segments (think, assistant reply, tool_call) carry a train mask -- the
 * user turns, tool responses and observations are context, not labels. */
#ifndef WUBU_TRAJ_SFT_H
#define WUBU_TRAJ_SFT_H

enum {
    WUBU_SEG_USER = 0,   /* the user turn            -- masked (context) */
    WUBU_SEG_OBS,        /* the environment obs      -- masked (context) */
    WUBU_SEG_TOOL_RESP,  /* the tool response        -- masked (context) */
    WUBU_SEG_ASSISTANT,  /* the assistant reply      -- TRAINS */
    WUBU_SEG_THINK,      /* the reasoning trace      -- TRAINS */
    WUBU_SEG_TOOL_CALL,  /* the tool call            -- TRAINS */
    WUBU_SEG_END
};

typedef struct {
    int type;          /* WUBU_SEG_* */
    const char *text;  /* the segment body (NUL-terminated) */
    int train;         /* 1 = the segment's tokens train the loss */
    const char *base;  /* the internal copy's base (for the free) */
} wubu_sft_seg_t;

/* Parse a raw trajectory and fill the segment list.
 * traj: the raw text with one segment per line, prefixed by one of
 *   [user] [obs] [tool_resp] [assistant] [think] [tool_call]
 *   (the prefix is stripped; empty/untagged lines are skipped).
 * segs/out: the filled segment array (max max_segs). The segment texts
 *   point into ONE internal copy -- release it with
 *   wubu_traj_sft_segs_free(segs, n). The input buffer is never
 *   modified (deterministic re-parsing).
 * Returns the segment count (or 0 on a bad input). */
int wubu_traj_sft_convert(const char *traj, wubu_sft_seg_t *segs,
                          int max_segs);

/* Free the internal copy owned by the segments (safe to call with n<=0). */
void wubu_traj_sft_segs_free(wubu_sft_seg_t *segs, int n);

/* The training-mask fraction: the number of train segments / total
 * (the Hermes "69% output tokens" analogue at the segment level). */
float wubu_traj_sft_train_frac(const wubu_sft_seg_t *segs, int n);

#endif
