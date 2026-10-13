#ifndef WUBU_MAIL_H
#define WUBU_MAIL_H

/* wubu_mail.h -- Win98/XP Email client.
 *
 * A mailbox with folders (inbox / sent / archive / trash) and messages. Each
 * message carries from, to, subject, a body, a read flag, and a timestamp.
 * Messages live in exactly one folder; they can be moved between folders and
 * deleted (moved to trash). Tracks per-folder counts and unread counts.
 * Deterministic, no stubs. */

#include <stddef.h>
#include <stdint.h>

#define WUBU_MAIL_MAX_MSGS 256
#define WUBU_MAIL_FIELD_MAX 128

typedef enum {
    WUBU_MAIL_INBOX = 0,
    WUBU_MAIL_SENT,
    WUBU_MAIL_ARCHIVE,
    WUBU_MAIL_TRASH,
    WUBU_MAIL_FOLDER_COUNT
} wubu_mail_folder_t;

typedef struct {
    int id;
    wubu_mail_folder_t folder;
    char from[WUBU_MAIL_FIELD_MAX];
    char to[WUBU_MAIL_FIELD_MAX];
    char subject[WUBU_MAIL_FIELD_MAX];
    char body[WUBU_MAIL_FIELD_MAX];
    int read;              /* 1 = read, 0 = unread */
    uint64_t ts;           /* timestamp (monotonic seconds) */
} wubu_mail_msg_t;

typedef struct {
    wubu_mail_msg_t msgs[WUBU_MAIL_MAX_MSGS];
    int n_msgs;
    int next_id;
    uint64_t clock;        /* message timestamp source */
} wubu_mailbox_t;

/* Initialise (empty). */
int wubu_ma_init(wubu_mailbox_t *mb);

/* Receive a message into a folder (returns id, -1 on full). */
int wubu_ma_receive(wubu_mailbox_t *mb, wubu_mail_folder_t folder,
                    const char *from, const char *to, const char *subject,
                    const char *body);

/* Mark a message read/unread by id. Returns 0/-1. */
int wubu_ma_set_read(wubu_mailbox_t *mb, int id, int read);

/* Move a message to another folder. Returns 0/-1. */
int wubu_ma_move(wubu_mailbox_t *mb, int id, wubu_mail_folder_t folder);

/* Permanently delete a message by id (remove from the store). Returns 0/-1. */
int wubu_ma_delete(wubu_mailbox_t *mb, int id);

/* Counts. */
int wubu_ma_count(const wubu_mailbox_t *mb, wubu_mail_folder_t folder);
int wubu_ma_unread(const wubu_mailbox_t *mb, wubu_mail_folder_t folder);

/* Find a message by id. Returns a read-only pointer or NULL. */
const wubu_mail_msg_t *wubu_ma_get(const wubu_mailbox_t *mb, int id);

/* Folder name for display (NULL if invalid). */
const char *wubu_ma_folder_name(wubu_mail_folder_t folder);

#endif
