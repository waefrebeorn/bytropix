/* barun_save.h -- export trained Barun checkpoints as real safetensors. */
#ifndef BARUN_SAVE_H
#define BARUN_SAVE_H

#include "wubu_barun.h"

/* write the model in the released 137-tensor safetensors layout. */
int barun_save_safetensors(const barun_model_t *m, const char *path);

#endif
