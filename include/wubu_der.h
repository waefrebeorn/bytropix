/*
 * wubu_der.h -- Dark Experience Replay (BB07). C11.
 *
 * Buzzega et al 2020: the replay buffer stores the TEACHER's soft
 * targets ("dark knowledge") alongside the transitions; the combined
 * loss is the task loss plus the distillation KL over the replayed
 * samples:  L = L_task + alpha * KL(teacher || student).
 * The dark knowledge (the logits' relative confidences) preserves the
 * old-task manifold far better than hard-label replay.
 */
#ifndef WUBU_DER_H
#define WUBU_DER_H

#include <stdint.h>

#define WUBU_DER_BUFSZ 256
#define WUBU_DER_DIMS  16

typedef struct {
    float logits[WUBU_DER_BUFSZ][WUBU_DER_DIMS]; /* teacher soft targets */
    uint8_t used[WUBU_DER_BUFSZ];
    int head;      /* next slot (ring) */
    int count;     /* occupied slots */
} wubu_der_buffer_t;

/* Push a teacher logits vector into the ring buffer (oldest evicted). */
int wubu_der_push(wubu_der_buffer_t *b, const float *teacher_logits, int ndim);
/* The softmax-cross-entropy distillation loss of the student's logits
 * against the teacher's (temperature-softened), averaged over the
 * replayed samples. Returns 0 with an empty buffer. */
float wubu_der_loss(const wubu_der_buffer_t *b, const float *student_logits,
                    int ndim, float temperature);
/* The combined loss: ce + alpha * der_loss (the alpha weighting). */
float wubu_der_total(float ce, float der, float alpha);

#endif
