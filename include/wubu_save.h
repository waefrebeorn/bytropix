/* wubu_save.h -- export trained WuBu checkpoints as real safetensors. */
#ifndef WUBU_SAVE_H
#define WUBU_SAVE_H

#include "wubu.h"

/* write the model in the released 137-tensor safetensors layout. */
int wubu_save_safetensors(const wubu_model_t *m, const char *path);

#endif
