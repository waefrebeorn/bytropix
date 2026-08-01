/*
 * wubu_audio.h -- Audio encoder: mel-spectrogram + real FFT (CC02).
 */
#ifndef WUBU_AUDIO_H
#define WUBU_AUDIO_H

#define WUBU_AUDIO_SAMPLE_RATE 16000
#define WUBU_AUDIO_FRAME_SIZE 512   /* STFT window */
#define WUBU_AUDIO_HOP 160          /* 10ms hop at 16kHz */
#define WUBU_AUDIO_N_MELS 40
#define WUBU_AUDIO_MAX_FRAMES 64    /* ~4 seconds */

typedef struct {
    float mel_filterbank[WUBU_AUDIO_N_MELS][WUBU_AUDIO_FRAME_SIZE / 2 + 1];
    int init;
} wubu_audio_t;

int  wubu_audio_init(wubu_audio_t *a);
/* Encode PCM samples → mel spectrogram. Returns frames written. */
int  wubu_audio_encode(const wubu_audio_t *a, const float *pcm, int n_samples,
                        float *out_mel, int max_frames);
/* Hann window for STFT. */
void wubu_audio_hann(float *win, int n);

#endif