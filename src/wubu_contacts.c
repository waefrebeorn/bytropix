/* wubu_contacts.c -- Win98/XP Address Book.
 * See wubu_contacts.h. Real, deterministic contact store. */

#include "wubu_contacts.h"
#include <string.h>
#include <ctype.h>

static int eq_ci(const char *a, const char *b) {
    if (!a || !b) return 0;
    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) return 0;
        a++; b++;
    }
    return *a == '\0' && *b == '\0';
}

static int contains_ci(const char *hay, const char *needle) {
    if (!hay || !needle) return 0;
    if (*needle == '\0') return 1;   /* empty query matches all */
    size_t nl = strlen(needle);
    size_t hl = strlen(hay);
    if (nl > hl) return 0;
    for (size_t i = 0; i + nl <= hl; i++) {
        size_t j = 0;
        for (; j < nl; j++)
            if (tolower((unsigned char)hay[i + j]) != tolower((unsigned char)needle[j]))
                break;
        if (j == nl) return 1;
    }
    return 0;
}

int wubu_ct_init(wubu_contacts_t *ab) {
    if (!ab) return -1;
    memset(ab, 0, sizeof(*ab));
    return 0;
}

int wubu_ct_add(wubu_contacts_t *ab, const char *name, const char *phone,
                const char *email) {
    if (!ab || !name || !phone || !email) return -1;
    if (wubu_ct_find(ab, name) >= 0) return -1;   /* dup (case-insensitive) */
    if (ab->n >= WUBU_CT_MAX_CONTACTS) return -1;
    wubu_ct_contact_t *c = &ab->contacts[ab->n];
    memset(c, 0, sizeof(*c));
    strncpy(c->name, name, WUBU_CT_FIELD_MAX - 1);
    strncpy(c->phone, phone, WUBU_CT_FIELD_MAX - 1);
    strncpy(c->email, email, WUBU_CT_FIELD_MAX - 1);
    ab->n++;
    return 0;
}

int wubu_ct_find(const wubu_contacts_t *ab, const char *name) {
    if (!ab || !name) return -1;
    for (int i = 0; i < ab->n; i++)
        if (eq_ci(ab->contacts[i].name, name)) return i;
    return -1;
}

int wubu_ct_update(wubu_contacts_t *ab, const char *name, const char *phone,
                   const char *email) {
    if (!ab || !name) return -1;
    int i = wubu_ct_find(ab, name);
    if (i < 0) return -1;
    if (phone) strncpy(ab->contacts[i].phone, phone, WUBU_CT_FIELD_MAX - 1);
    if (email) strncpy(ab->contacts[i].email, email, WUBU_CT_FIELD_MAX - 1);
    return 0;
}

int wubu_ct_remove(wubu_contacts_t *ab, const char *name) {
    if (!ab || !name) return -1;
    int i = wubu_ct_find(ab, name);
    if (i < 0) return -1;
    ab->contacts[i] = ab->contacts[ab->n - 1];
    ab->n--;
    return 0;
}

int wubu_ct_search(const wubu_contacts_t *ab, const char *query,
                   int *out, int max) {
    if (!ab || !out || max < 0) return -1;
    int n = 0;
    for (int i = 0; i < ab->n; i++) {
        if (contains_ci(ab->contacts[i].name, query)) {
            if (n < max) out[n] = i;
            n++;
        }
    }
    return n;
}

int wubu_ct_count(const wubu_contacts_t *ab) {
    if (!ab) return -1;
    return ab->n;
}

const char *wubu_ct_name(const wubu_contacts_t *ab, int idx) {
    if (!ab || idx < 0 || idx >= ab->n) return NULL;
    return ab->contacts[idx].name;
}

const char *wubu_ct_phone(const wubu_contacts_t *ab, int idx) {
    if (!ab || idx < 0 || idx >= ab->n) return NULL;
    return ab->contacts[idx].phone;
}

const char *wubu_ct_email(const wubu_contacts_t *ab, int idx) {
    if (!ab || idx < 0 || idx >= ab->n) return NULL;
    return ab->contacts[idx].email;
}
