/*
 * wubu_audio.c -- Audio encoder: mel-spectrogram + real FFT from scratch (CC02). C11.
 *
 * Convergence (librosa-from-scratch 7-hop: STFT, Hann window, mel filterbank,
 * hz→mel, power spectrogram, log-scale):
 *   - CC02: raw PCM 16kHz → frames of 512 (Hann window) → real FFT →
 *     power spectrogram → mel filterbank (40 bins, hz→mel) → log-mel.
 *   Implements a decimation-in-time radix-2 FFT (no deps).
 */
#include "wubu_audio.h"
#include <math.h>
#include <string.h>

static float hz_to_mel(float hz) {
    return 2595.0f * log10f(1.0f + hz / 700.0f);
}
static float mel_to_hz(float mel) {
    return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

/* Bit-reversal permutation */
static void bit_reverse(float *re, float *im, int n) {
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float tr = re[i]; re[i] = re[j]; re[j] = tr;
            float ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }
}

/* Radix-2 DIT FFT (n must be power of 2). In-place on re[], im[]. */
static void fft_radix2(float *re, float *im, int n, int inverse) {
    bit_reverse(re, im, n);
    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * (float)M_PI / len * (inverse ? -1 : 1);
        float wlen_re = cosf(ang), wlen_im = sinf(ang);
        for (int i = 0; i < n; i += len) {
            float w_re = 1.0f, w_im = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                float tr = w_re * re[i + j + len/2] - w_im * im[i + j + len/2];
                float ti = w_re * im[i + j + len/2] + w_im * re[i + j + len/2];
                re[i + j + len/2] = re[i + j] - tr;
                im[i + j + len/2] = im[i + j] - ti;
                re[i + j] += tr;
                im[i + j] += ti;
                float tmp = w_re * wlen_re - w_im * wlen_im;
                w_im = w_re * wlen_im + w_im * wlen_re;
                w_re = tmp;
            }
        }
    }
    if (inverse) { for (int i = 0; i < n; i++) { re[i] /= n; im[i] /= n; } }
}

void wubu_audio_hann(float *win, int n) {
    for (int i = 0; i < n; i++)
        win[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (n - 1)));
}

int wubu_audio_init(wubu_audio_t *a) {
    if (!a) return -1;
    int n_fft = WUBU_AUDIO_FRAME_SIZE;
    int n_freq = n_fft / 2 + 1;
    int n_mels = WUBU_AUDIO_N_MELS;
    float f_min = 0.0f, f_max = (float)(WUBU_AUDIO_SAMPLE_RATE / 2);
    /* Build mel filterbank */
    float mel_min = hz_to_mel(f_min), mel_max = hz_to_mel(f_max);
    float mel_points[WUBU_AUDIO_N_MELS + 2];
    for (int i = 0; i < n_mels + 2; i++)
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (n_mels + 1);
    float hz_points[WUBU_AUDIO_N_MELS + 2];
    for (int i = 0; i < n_mels + 2; i++)
        hz_points[i] = mel_to_hz(mel_points[i]);
    int bin[WUBU_AUDIO_N_MELS + 2];
    for (int i = 0; i < n_mels + 2; i++)
        bin[i] = (int)(hz_points[i] / f_max * (n_freq - 1));
    memset(a->mel_filterbank, 0, sizeof(a->mel_filterbank));
    for (int m = 0; m < n_mels; m++) {
        for (int k = bin[m]; k <= bin[m + 2]; k++) {
            if (k < 0 || k >= n_freq) continue;
            float val = 0.0f;
            if (k >= bin[m] && k <= bin[m + 1])
                val = (k - bin[m]) / (float)(bin[m + 1] - bin[m] + 1e-10f);
            else if (k >= bin[m + 1] && k <= bin[m + 2])
                val = (bin[m + 2] - k) / (float)(bin[m + 2] - bin[m + 1] + 1e-10f);
            a->mel_filterbank[m][k] = val;
        }
    }
    a->init = 1;
    return 0;
}

int wubu_audio_encode(const wubu_audio_t *a, const float *pcm, int n_samples,
                      float *out_mel, int max_frames) {
    if (!a || !pcm || !out_mel || n_samples <= 0) return -1;
    int n_fft = WUBU_AUDIO_FRAME_SIZE;
    int hop = WUBU_AUDIO_HOP;
    int n_freq = n_fft / 2 + 1;
    int n_mels = WUBU_AUDIO_N_MELS;
    float win[WUBU_AUDIO_FRAME_SIZE];
    wubu_audio_hann(win, n_fft);
    float re_buf[WUBU_AUDIO_FRAME_SIZE], im_buf[WUBU_AUDIO_FRAME_SIZE];
    int n_frames = 0;
    for (int i = 0; i + n_fft <= n_samples && n_frames < max_frames; i += hop) {
        /* Apply window */
        for (int j = 0; j < n_fft; j++) {
            re_buf[j] = pcm[i + j] * win[j];
            im_buf[j] = 0.0f;
        }
        fft_radix2(re_buf, im_buf, n_fft, 0);
        /* Power spectrogram → mel */
        for (int m = 0; m < n_mels; m++) {
            float s = 0.0f;
            for (int k = 0; k < n_freq; k++) {
                float power = re_buf[k] * re_buf[k] + im_buf[k] * im_buf[k];
                s += a->mel_filterbank[m][k] * power;
            }
            /* Log-mel (dB) */
            out_mel[n_frames * n_mels + m] = logf(s + 1e-10f);
        }
        n_frames++;
    }
    return n_frames;
}
