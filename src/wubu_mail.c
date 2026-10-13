/* wubu_mail.c -- Win98/XP Email client.
 * See wubu_mail.h. Real, deterministic mailbox model. */

#include "wubu_mail.h"
#include <string.h>

static const char *g_folder_names[WUBU_MAIL_FOLDER_COUNT] = {
    "Inbox", "Sent", "Archive", "Trash"
};

const char *wubu_ma_folder_name(wubu_mail_folder_t folder) {
    if (folder < 0 || folder >= WUBU_MAIL_FOLDER_COUNT) return NULL;
    return g_folder_names[folder];
}

int wubu_ma_init(wubu_mailbox_t *mb) {
    if (!mb) return -1;
    memset(mb, 0, sizeof(*mb));
    mb->next_id = 1;
    mb->clock = 1000;
    return 0;
}

int wubu_ma_receive(wubu_mailbox_t *mb, wubu_mail_folder_t folder,
                    const char *from, const char *to, const char *subject,
                    const char *body) {
    if (!mb || !from || !to || !subject || !body) return -1;
    if (folder < 0 || folder >= WUBU_MAIL_FOLDER_COUNT) return -1;
    if (mb->n_msgs >= WUBU_MAIL_MAX_MSGS) return -1;
    wubu_mail_msg_t *m = &mb->msgs[mb->n_msgs];
    memset(m, 0, sizeof(*m));
    m->id = mb->next_id++;
    m->folder = folder;
    strncpy(m->from, from, WUBU_MAIL_FIELD_MAX - 1);
    strncpy(m->to, to, WUBU_MAIL_FIELD_MAX - 1);
    strncpy(m->subject, subject, WUBU_MAIL_FIELD_MAX - 1);
    strncpy(m->body, body, WUBU_MAIL_FIELD_MAX - 1);
    m->read = 0;              /* new messages arrive unread */
    m->ts = mb->clock++;
    mb->n_msgs++;
    return m->id;
}

static int find(const wubu_mailbox_t *mb, int id) {
    for (int i = 0; i < mb->n_msgs; i++)
        if (mb->msgs[i].id == id) return i;
    return -1;
}

int wubu_ma_set_read(wubu_mailbox_t *mb, int id, int read) {
    if (!mb) return -1;
    int i = find(mb, id);
    if (i < 0) return -1;
    mb->msgs[i].read = read ? 1 : 0;
    return 0;
}

int wubu_ma_move(wubu_mailbox_t *mb, int id, wubu_mail_folder_t folder) {
    if (!mb || folder < 0 || folder >= WUBU_MAIL_FOLDER_COUNT) return -1;
    int i = find(mb, id);
    if (i < 0) return -1;
    mb->msgs[i].folder = folder;
    return 0;
}

int wubu_ma_delete(wubu_mailbox_t *mb, int id) {
    if (!mb) return -1;
    int i = find(mb, id);
    if (i < 0) return -1;
    mb->msgs[i] = mb->msgs[mb->n_msgs - 1];
    mb->n_msgs--;
    return 0;
}

int wubu_ma_count(const wubu_mailbox_t *mb, wubu_mail_folder_t folder) {
    if (!mb || folder < 0 || folder >= WUBU_MAIL_FOLDER_COUNT) return -1;
    int n = 0;
    for (int i = 0; i < mb->n_msgs; i++)
        if (mb->msgs[i].folder == folder) n++;
    return n;
}

int wubu_ma_unread(const wubu_mailbox_t *mb, wubu_mail_folder_t folder) {
    if (!mb || folder < 0 || folder >= WUBU_MAIL_FOLDER_COUNT) return -1;
    int n = 0;
    for (int i = 0; i < mb->n_msgs; i++)
        if (mb->msgs[i].folder == folder && !mb->msgs[i].read) n++;
    return n;
}

const wubu_mail_msg_t *wubu_ma_get(const wubu_mailbox_t *mb, int id) {
    if (!mb) return NULL;
    int i = find(mb, id);
    if (i < 0) return NULL;
    return &mb->msgs[i];
}
