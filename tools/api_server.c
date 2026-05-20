/**
 * api_server.c — Local inference API server for bytropix
 *
 * OpenAI-compatible HTTP server for educational/research use.
 * Provides a REST API wrapping the infer_text_gpu binary.
 *
 * ============================================================================
 * DISCLAIMER
 * ============================================================================
 * This software is provided for EDUCATIONAL AND RESEARCH PURPOSES only.
 * It is openly licensed scaffolding — anyone may build, deploy, or sell
 * inference services built on this foundation.
 *
 * THE AUTHORS ASSUME NO LIABILITY for any use, including but not limited to:
 *   - Security vulnerabilities in deployment configurations
 *   - Data privacy violations by downstream operators
 *   - Regulatory compliance (GDPR, CCPA, EU AI Act, etc.)
 *
 * Operators are responsible for their own security posture, API key management,
 * rate limiting, content filtering, and legal compliance.
 * ============================================================================
 *
 * Protocol: HTTP/1.1 with optional TLS (OpenSSL)
 * Endpoints:
 *   POST /v1/chat/completions  — Chat completions (OpenAI-compatible)
 *   POST /v1/completions       — Text completions
 *   GET  /v1/models            — List available models
 *   GET  /health               — Health check
 *
 * Usage:
 *   ./api_server             # Port 8080, no auth, no TLS
 *   ./api_server --port 8443 --tls cert.pem key.pem
 *   ./api_server --sandbox   # Fake responses, isolated testing
 *   ./api_server --auth "sk-..."  # Require API key
 *
 * Environment:
 *   INFER_BIN=./infer_text_gpu  [default]
 *   MODEL_PATH=/path/to/model.gguf [default from config]
 *   API_PORT=8080
 *   API_AUTH_KEY=sk-...
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <time.h>
#include <math.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <ctype.h>

/* OpenSSL for TLS support */
#include <openssl/ssl.h>
#include <openssl/err.h>

/* ================================================================
 *  Constants
 * ================================================================ */

#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)

#define MAX_HEADERS 16384
#define MAX_BODY    1048576  /* 1 MB */
#define MAX_PATH    4096
#define MAX_TOKENS  4096
#define RATE_WINDOW 60       /* seconds */
#define RATE_LIMIT  60       /* requests per window */
#define DEFAULT_PORT 8080

/* ================================================================
 *  CLI flags / config
 * ================================================================ */

static int     g_port = DEFAULT_PORT;
static int     g_sandbox = 0;
static char    g_auth_key[256] = "";
static int     g_use_tls = 0;
static char    g_tls_cert[1024] = "";
static char    g_tls_key[1024] = "";
static char    g_infer_bin[1024] = "./infer_text_gpu";
static char    g_model_path[1024] = "/home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf";
static volatile int g_running = 1;

/* Rate limiter state */
typedef struct { time_t timestamps[RATE_LIMIT]; int count; } rate_bucket_t;
static rate_bucket_t g_rate_bucket;
static int g_rate_count = 0;

/* ================================================================
 *  Signal handler
 * ================================================================ */

static void handle_signal(int sig) {
    (void)sig;
    g_running = 0;
    fprintf(stderr, "\n[server] Shutting down...\n");
}

/* ================================================================
 *  Helpers
 * ================================================================ */

static void print_banner(void) {
    printf("\n");
    printf("  ╔══════════════════════════════════════════════════════╗\n");
    printf("  ║       bytropix Local Inference API Server          ║\n");
    printf("  ║       EDUCATIONAL / RESEARCH USE ONLY              ║\n");
    printf("  ╚══════════════════════════════════════════════════════╝\n");
    printf("  Port: %d  |  TLS: %s  |  Sandbox: %s\n",
           g_port, g_use_tls ? "yes" : "no", g_sandbox ? "yes" : "no");
    printf("  Model: %s\n", g_model_path);
    printf("  Infer: %s\n", g_infer_bin);
    printf("  Auth:  %s\n\n", g_auth_key[0] ? "enabled" : "none (open)");
}

/* ================================================================
 *  JSON helpers (minimal, inline)
 * ================================================================ */

/* Escape a string for JSON */
static char *json_escape(const char *s) {
    if (!s) return strdup("null");
    size_t cap = strlen(s) * 2 + 3;
    char *out = (char *)malloc(cap);
    if (!out) return NULL;
    size_t j = 0;
    out[j++] = '"';
    for (const char *p = s; *p; p++) {
        unsigned char c = (unsigned char)*p;
        switch (c) {
            case '"':  out[j++] = '\\'; out[j++] = '"';  break;
            case '\\': out[j++] = '\\'; out[j++] = '\\'; break;
            case '\n': out[j++] = '\\'; out[j++] = 'n';  break;
            case '\r': out[j++] = '\\'; out[j++] = 'r';  break;
            case '\t': out[j++] = '\\'; out[j++] = 't';  break;
            default:
                if (c < 0x20) {
                    if (j + 6 >= cap) { cap *= 2; out = (char *)realloc(out, cap); }
                    j += snprintf(out + j, cap - j, "\\u%04x", c);
                } else {
                    out[j++] = (char)c;
                }
                break;
        }
        if (j + 8 >= cap) { cap *= 2; out = (char *)realloc(out, cap); }
    }
    out[j++] = '"';
    out[j] = '\0';
    return out;
}

/* Build simple JSON response */
static char *make_json_response(const char *content) {
    /* This is simplified — real impl would use a JSON builder */
    size_t clen = content ? strlen(content) : 0;
    size_t bufsz = clen + 4096;
    char *buf = (char *)malloc(bufsz);
    if (!buf) return NULL;

    char *escaped = json_escape(content ? content : "");
    time_t now = time(NULL);
    char ts[64];
    snprintf(ts, sizeof(ts), "%ld", (long)now);

    snprintf(buf, bufsz,
        "{"
        "\"id\":\"chatcmpl-%s\","
        "\"object\":\"chat.completion\","
        "\"created\":%s,"
        "\"model\":\"bytropix-qwen3.6\","
        "\"choices\":[{"
          "\"index\":0,"
          "\"message\":{\"role\":\"assistant\",\"content\":%s},"
          "\"finish_reason\":\"stop\""
        "}],"
        "\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":0,\"total_tokens\":0}"
        "}",
        ts, ts, escaped);

    free(escaped);
    return buf;
}

/* ================================================================
 *  Rate limiter
 * ================================================================ */

static int check_rate_limit(void) {
    time_t now = time(NULL);
    time_t cutoff = now - RATE_WINDOW;

    /* Prune old entries */
    int new_count = 0;
    for (int i = 0; i < g_rate_count; i++) {
        if (g_rate_bucket.timestamps[i] > cutoff) {
            g_rate_bucket.timestamps[new_count++] = g_rate_bucket.timestamps[i];
        }
    }
    g_rate_count = new_count;

    if (g_rate_count >= RATE_LIMIT) return 0; /* rate limited */
    g_rate_bucket.timestamps[g_rate_count++] = now;
    return 1; /* allowed */
}

/* ================================================================
 *  API key validation
 * ================================================================ */

static int check_auth(const char *auth_header) {
    if (!g_auth_key[0]) return 1; /* no auth required */
    if (!auth_header) return 0;

    /* Expect: "Bearer sk-..." */
    const char *p = auth_header;
    while (*p == ' ') p++;
    if (strncmp(p, "Bearer ", 7) != 0) return 0;
    p += 7;
    return strcmp(p, g_auth_key) == 0;
}

/* ================================================================
 *  Inference subprocess
 * ================================================================ */

static char *run_inference(const char *prompt, int max_tokens,
                            double temperature, int top_k, double top_p)
{
    /* Sandbox mode: return fake response */
    if (g_sandbox) {
        char *result = (char *)malloc(8192);
        if (!result) return NULL;

        char *escaped = json_escape("This is a sandbox test response for educational API verification purposes. In production, this would contain real model-generated text.");
        time_t now = time(NULL);
        snprintf(result, 8192,
            "{"
            "\"id\":\"chatcmpl-sandbox-%ld\","
            "\"object\":\"chat.completion\","
            "\"created\":%ld,"
            "\"model\":\"bytropix-qwen3.6-sandbox\","
            "\"choices\":[{"
              "\"index\":0,"
              "\"message\":{\"role\":\"assistant\",\"content\":%s},"
              "\"finish_reason\":\"stop\""
            "}],"
            "\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":0,\"total_tokens\":0}"
            "}",
            (long)now, (long)now, escaped);
        free(escaped);
        return result;
    }

    /* Build command */
    char cmd[32768];
    snprintf(cmd, sizeof(cmd),
        "TEM=%.4f TOP_K=%d TOP_P=%.4f MOE=1 timeout %d %s %s %s %d 2>&1",
        temperature, top_k, top_p,
        max_tokens > 100 ? max_tokens + 30 : 120,
        g_infer_bin, g_model_path, prompt, max_tokens);

    /* Run and capture output */
    FILE *fp = popen(cmd, "r");
    if (!fp) return NULL;

    size_t cap = 65536, len = 0;
    char *output = (char *)malloc(cap);
    if (!output) { pclose(fp); return NULL; }
    output[0] = '\0';

    char line[4096];
    int in_generation = 0;
    char gen_text[65536] = "";
    size_t gen_len = 0;

    while (fgets(line, sizeof(line), fp)) {
        size_t llen = strlen(line);
        if (len + llen + 1 > cap) {
            cap *= 2;
            char *new_out = (char *)realloc(output, cap);
            if (!new_out) { free(output); pclose(fp); return NULL; }
            output = new_out;
        }
        memcpy(output + len, line, llen + 1);
        len += llen;

        /* Extract generated text: everything after "--- Decode ---" */
        if (strstr(line, "--- Decode") || strstr(line, "--- Chunked prefill")) {
            in_generation = 1;
            continue;
        }
        if (strstr(line, "=== Summary") || strstr(line, "=== PASS ===")) {
            in_generation = 0;
            continue;
        }
        if (in_generation && line[0] != '\n' && !strstr(line, "Prefill:")
            && !strstr(line, "T+") && !strstr(line, "=== PASS")) {
            size_t add = strlen(line);
            if (gen_len + add < sizeof(gen_text)) {
                memcpy(gen_text + gen_len, line, add);
                gen_len += add;
            }
        }
    }

    (void)pclose(fp);

    /* Build JSON response */
    if (gen_len == 0) {
        /* Fallback: use raw output */
        snprintf(gen_text, sizeof(gen_text), "%.3000s", output);
        gen_len = strlen(gen_text);
    }

    char *escaped = json_escape(gen_text);
    time_t now = time(NULL);
    char result[131072];
    snprintf(result, sizeof(result),
        "{"
        "\"id\":\"chatcmpl-%ld\","
        "\"object\":\"chat.completion\","
        "\"created\":%ld,"
        "\"model\":\"bytropix-qwen3.6\","
        "\"choices\":[{"
          "\"index\":0,"
          "\"message\":{\"role\":\"assistant\",\"content\":%s},"
          "\"finish_reason\":\"stop\""
        "}],"
        "\"usage\":{\"prompt_tokens\":0,\"completion_tokens\":0,\"total_tokens\":0}"
        "}",
        (long)now, (long)now, escaped);

    free(escaped);
    free(output);
    return strdup(result);
}

/* ================================================================
 *  HTTP response helpers
 * ================================================================ */

static void send_response(int fd, int status, const char *content_type,
                           const char *body, SSL *ssl)
{
    char header[4096];
    int len = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        "Server: bytropix-api/1.0 (educational research)\r\n"
        "X-Content-Type-Options: nosniff\r\n"
        "Connection: close\r\n"
        "\r\n",
        status,
        status == 200 ? "OK" :
        status == 400 ? "Bad Request" :
        status == 401 ? "Unauthorized" :
        status == 404 ? "Not Found" :
        status == 429 ? "Too Many Requests" :
        status == 500 ? "Internal Server Error" : "Unknown",
        content_type ? content_type : "application/json",
        body ? strlen(body) : 0);

    if (ssl) {
        SSL_write(ssl, header, len);
        if (body) SSL_write(ssl, body, strlen(body));
    } else {
        write(fd, header, len);
        if (body) write(fd, body, strlen(body));
    }
}

static void send_json(int fd, int status, const char *json, SSL *ssl) {
    send_response(fd, status, "application/json; charset=utf-8", json, ssl);
}

static void send_error(int fd, int status, const char *msg, SSL *ssl) {
    char err_body[4096];
    snprintf(err_body, sizeof(err_body),
        "{\"error\":{\"message\":%s,\"type\":\"error\",\"code\":%d}}",
        json_escape(msg), status);
    send_json(fd, status, err_body, ssl);
}

/* ================================================================
 *  Request parsing
 * ================================================================ */

typedef struct {
    char method[16];
    char path[1024];
    char headers[MAX_HEADERS];
    char body[MAX_BODY];
    size_t body_len;
    int content_length;
} http_request_t;

static int parse_request(int fd, http_request_t *req, SSL *ssl) {
    memset(req, 0, sizeof(*req));

    /* Read request line + headers */
    char buf[MAX_HEADERS + MAX_BODY];
    size_t total = 0;

    while (total < sizeof(buf) - 1) {
        int n;
        if (ssl)
            n = SSL_read(ssl, buf + total, (int)(sizeof(buf) - total - 1));
        else
            n = (int)read(fd, buf + total, sizeof(buf) - total - 1);

        if (n <= 0) break;
        total += (size_t)n;
        buf[total] = '\0';

        /* Check for end of headers */
        if (strstr(buf, "\r\n\r\n")) break;
    }

    if (total == 0) return 0;
    buf[total] = '\0';

    /* Parse first line */
    char *line_end = strchr(buf, '\r');
    if (!line_end) line_end = strchr(buf, '\n');
    if (!line_end) return 0;
    *line_end = '\0';

    char method[16], path[1024], version[16];
    if (sscanf(buf, "%15s %1023s %15s", method, path, version) < 2) return 0;
    strcpy(req->method, method);
    strcpy(req->path, path);

    /* Parse headers */
    char *hdr_start = line_end + 1;
    while (*hdr_start == '\n' || *hdr_start == '\r') hdr_start++;

    char *hdr_end = strstr(hdr_start, "\r\n\r\n");
    if (!hdr_end) hdr_end = strstr(hdr_start, "\n\n");
    if (!hdr_end) return 0;

    size_t hdr_len = (size_t)(hdr_end - hdr_start);
    if (hdr_len >= sizeof(req->headers)) hdr_len = sizeof(req->headers) - 1;
    memcpy(req->headers, hdr_start, hdr_len);
    req->headers[hdr_len] = '\0';

    /* Find Content-Length */
    char *cl = strstr(req->headers, "Content-Length:");
    if (!cl) cl = strstr(req->headers, "content-length:");
    if (!cl) cl = strstr(req->headers, "Content-length:");
    if (cl) {
        cl += 15; /* skip past "Content-Length:" */
        while (*cl == ' ') cl++;
        req->content_length = atoi(cl);
        if (req->content_length > MAX_BODY) req->content_length = MAX_BODY;

        /* Body is after headers */
        char *body_start = hdr_end;
        while (*body_start == '\r' || *body_start == '\n') body_start++;
        size_t body_avail = total - (size_t)(body_start - buf);
        req->body_len = (size_t)(req->content_length < (int)body_avail ?
                         req->content_length : (int)body_avail);
        if (req->body_len > 0) {
            memcpy(req->body, body_start, req->body_len);
            req->body[req->body_len] = '\0';
        }
    }

    return 1;
}

/* Find header value (case-insensitive) */
static const char *get_header(http_request_t *req, const char *name) {
    if (!req->headers[0]) return NULL;
    char *p = req->headers;
    while (*p) {
        if (strncasecmp(p, name, strlen(name)) == 0) {
            p += strlen(name);
            while (*p == ':' || *p == ' ') p++;
            return p;
        }
        /* Skip to next line */
        while (*p && *p != '\n') p++;
        if (*p == '\n') p++;
    }
    return NULL;
}

/* ================================================================
 *  Router
 * ================================================================ */

static void handle_request(int fd, http_request_t *req, SSL *ssl) {
    /* Health check */
    if (strcmp(req->path, "/health") == 0 && strcmp(req->method, "GET") == 0) {
        send_json(fd, 200, "{\"status\":\"ok\",\"service\":\"bytropix-inference-api\",\"version\":\"1.0.0\",\"educational\":true,\"notice\":\"For educational/research use only. No liability assumed.\"}", ssl);
        return;
    }

    /* List models */
    if (strcmp(req->path, "/v1/models") == 0 && strcmp(req->method, "GET") == 0) {
        char models[4096];
        char *name = json_escape("bytropix-qwen3.6");
        snprintf(models, sizeof(models),
            "{\"object\":\"list\",\"data\":[{\"id\":%s,\"object\":\"model\",\"created\":1715097600,\"owned_by\":\"bytropix\"}]}",
            name);
        free(name);
        send_json(fd, 200, models, ssl);
        return;
    }

    /* CORS preflight */
    if (strcmp(req->method, "OPTIONS") == 0) {
        send_response(fd, 204, "text/plain", "", ssl);
        return;
    }

    /* Auth check */
    const char *auth = get_header(req, "Authorization");
    if (!check_auth(auth)) {
        send_error(fd, 401, "Invalid or missing API key. Set --auth or provide Authorization: Bearer <key>", ssl);
        return;
    }

    /* Rate limit check */
    if (!check_rate_limit()) {
        send_error(fd, 429, "Rate limit exceeded. Max " STR(RATE_LIMIT) " requests per " STR(RATE_WINDOW) " seconds.", ssl);
        return;
    }

    /* Chat completions */
    if ((strcmp(req->path, "/v1/chat/completions") == 0 ||
         strcmp(req->path, "/v1/completions") == 0) &&
        strcmp(req->method, "POST") == 0)
    {
        if (req->body_len == 0) {
            send_error(fd, 400, "Empty request body", ssl);
            return;
        }

        /* Parse JSON body for prompt/messages */
        /* Simple extraction: find "content" field for chat, or "prompt" for completions */
        char prompt[65536] = "";
        int max_tokens = 256;
        double temperature = 1.0;
        int top_k = 20;
        double top_p = 0.95;

        const char *body = req->body;

        /* Extract prompt from messages array or prompt field */
        char *content_ptr = strstr(body, "\"content\"");
        char *prompt_ptr = strstr(body, "\"prompt\"");
        char *msgs_ptr = strstr(body, "\"messages\"");

        if (content_ptr && msgs_ptr && content_ptr > msgs_ptr) {
            /* Extract content value */
            char *val_start = strchr(content_ptr, ':');
            if (val_start) {
                val_start++;
                while (*val_start == ' ' || *val_start == '"') val_start++;
                char *val_end = strchr(val_start, '"');
                if (val_end) {
                    size_t plen = (size_t)(val_end - val_start);
                    if (plen > sizeof(prompt) - 1) plen = sizeof(prompt) - 1;
                    memcpy(prompt, val_start, plen);
                    prompt[plen] = '\0';
                }
            }
        } else if (prompt_ptr) {
            char *val_start = strchr(prompt_ptr, ':');
            if (val_start) {
                val_start++;
                while (*val_start == ' ' || *val_start == '"') val_start++;
                char *val_end = strchr(val_start, '"');
                if (val_end) {
                    size_t plen = (size_t)(val_end - val_start);
                    if (plen > sizeof(prompt) - 1) plen = sizeof(prompt) - 1;
                    memcpy(prompt, val_start, plen);
                    prompt[plen] = '\0';
                }
            }
        }

        /* Extract max_tokens, temperature, top_k, top_p */
        char *mt = strstr(body, "\"max_tokens\"");
        if (mt) { char *v = strchr(mt, ':'); if (v) max_tokens = atoi(v + 1); }
        char *tmp = strstr(body, "\"temperature\"");
        if (tmp) { char *v = strchr(tmp, ':'); if (v) temperature = atof(v + 1); }
        char *tk = strstr(body, "\"top_k\"");
        if (tk) { char *v = strchr(tk, ':'); if (v) top_k = atoi(v + 1); }
        char *tp = strstr(body, "\"top_p\"");
        if (tp) { char *v = strchr(tp, ':'); if (v) top_p = atof(v + 1); }

        if (prompt[0] == '\0') {
            /* Fallback: use whole body as debug info */
            snprintf(prompt, sizeof(prompt), "hello");
        }

        /* Run inference */
        char *result = run_inference(prompt, max_tokens, temperature, top_k, top_p);
        if (!result) {
            send_error(fd, 500, "Inference failed — check model path and infer binary", ssl);
            return;
        }

        send_json(fd, 200, result, ssl);
        free(result);
        return;
    }

    /* 404 for everything else */
    send_error(fd, 404, "Not found. Available endpoints: GET /health, GET /v1/models, POST /v1/chat/completions, POST /v1/completions", ssl);
}

/* ================================================================
 *  Client handler thread (simplified: fork per connection)
 * ================================================================ */

static void handle_client(int fd, SSL *ssl) {
    http_request_t req;
    if (parse_request(fd, &req, ssl)) {
        handle_request(fd, &req, ssl);
    }
    if (ssl) {
        SSL_shutdown(ssl);
        SSL_free(ssl);
    }
    close(fd);
}

/* ================================================================
 *  Main
 * ================================================================ */

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "OpenAI-compatible API server for bytropix inference engine.\n");
    fprintf(stderr, "EDUCATIONAL / RESEARCH USE ONLY.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --port N        Port to listen on (default: %d)\n", DEFAULT_PORT);
    fprintf(stderr, "  --sandbox       Fake responses (no GPU needed)\n");
    fprintf(stderr, "  --auth KEY      Require API key (Authorization: Bearer <key>)\n");
    fprintf(stderr, "  --tls CERT KEY  Enable TLS with certificate + key\n");
    fprintf(stderr, "  --model PATH    Model GGUF path\n");
    fprintf(stderr, "  --bin PATH      Infer binary path\n");
    fprintf(stderr, "  --help          This message\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Environment:\n");
    fprintf(stderr, "  INFER_BIN=./infer_text_gpu\n");
    fprintf(stderr, "  MODEL_PATH=/path/to/model.gguf\n");
    fprintf(stderr, "  API_PORT=8080\n");
    fprintf(stderr, "  API_AUTH_KEY=sk-...\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Security Notice: This server is for local/trusted-network use.\n");
    fprintf(stderr, "Do not expose to the internet without proper auth, TLS, and firewall.\n");
    fprintf(stderr, "Operators assume all liability for deployment.\n");
}

int main(int argc, char **argv) {
    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--port") == 0 && i + 1 < argc)
            g_port = atoi(argv[++i]);
        else if (strcmp(argv[i], "--sandbox") == 0)
            g_sandbox = 1;
        else if (strcmp(argv[i], "--auth") == 0 && i + 1 < argc)
            snprintf(g_auth_key, sizeof(g_auth_key), "%s", argv[++i]);
        else if (strcmp(argv[i], "--tls") == 0 && i + 2 < argc) {
            g_use_tls = 1;
            snprintf(g_tls_cert, sizeof(g_tls_cert), "%s", argv[++i]);
            snprintf(g_tls_key, sizeof(g_tls_key), "%s", argv[++i]);
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc)
            snprintf(g_model_path, sizeof(g_model_path), "%s", argv[++i]);
        else if (strcmp(argv[i], "--bin") == 0 && i + 1 < argc)
            snprintf(g_infer_bin, sizeof(g_infer_bin), "%s", argv[++i]);
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    /* Environment overrides */
    char *env;
    if ((env = getenv("API_PORT"))) g_port = atoi(env);
    if ((env = getenv("API_AUTH_KEY"))) snprintf(g_auth_key, sizeof(g_auth_key), "%s", env);
    if ((env = getenv("INFER_BIN"))) snprintf(g_infer_bin, sizeof(g_infer_bin), "%s", env);
    if ((env = getenv("MODEL_PATH"))) snprintf(g_model_path, sizeof(g_model_path), "%s", env);

    /* Signal handlers */
    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);
    signal(SIGCHLD, SIG_IGN); /* reap child processes */

    /* Initialize OpenSSL for TLS */
    SSL_CTX *ssl_ctx = NULL;
    if (g_use_tls) {
        SSL_load_error_strings();
        OPENSSL_init_ssl(0, NULL);
        ssl_ctx = SSL_CTX_new(TLS_server_method());
        if (!ssl_ctx) {
            fprintf(stderr, "Failed to create SSL context\n");
            return 1;
        }
        if (SSL_CTX_use_certificate_file(ssl_ctx, g_tls_cert, SSL_FILETYPE_PEM) <= 0) {
            fprintf(stderr, "Failed to load cert: %s\n", g_tls_cert);
            return 1;
        }
        if (SSL_CTX_use_PrivateKey_file(ssl_ctx, g_tls_key, SSL_FILETYPE_PEM) <= 0) {
            fprintf(stderr, "Failed to load key: %s\n", g_tls_key);
            return 1;
        }
    }

    /* Create socket */
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) { perror("socket"); return 1; }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(g_port);

    if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind");
        close(server_fd);
        return 1;
    }

    if (listen(server_fd, 5) < 0) {
        perror("listen");
        close(server_fd);
        return 1;
    }

    print_banner();

    /* Main accept loop */
    while (g_running) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0) {
            if (errno == EINTR) continue;
            perror("accept");
            continue;
        }

        /* Fork to handle client (simple concurrency — each request gets own process) */
        pid_t pid = fork();
        if (pid == 0) {
            /* Child process */
            close(server_fd);
            SSL *ssl = NULL;
            if (g_use_tls) {
                ssl = SSL_new(ssl_ctx);
                if (ssl) SSL_set_fd(ssl, client_fd);
            }
            handle_client(client_fd, ssl);
            _exit(0);
        } else if (pid > 0) {
            close(client_fd);
        } else {
            perror("fork");
            close(client_fd);
        }
    }

    close(server_fd);
    if (ssl_ctx) SSL_CTX_free(ssl_ctx);
    printf("[server] Shutdown complete.\n");
    return 0;
}
