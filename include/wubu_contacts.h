#ifndef WUBU_CONTACTS_H
#define WUBU_CONTACTS_H

/* wubu_contacts.h -- Win98/XP Address Book.
 *
 * A contact store: each contact has a display name, phone, and email. Supports
 * add (name-unique), update, remove, case-insensitive substring search by
 * name, and lookup. Deterministic, no stubs. */

#include <stddef.h>

#define WUBU_CT_MAX_CONTACTS 256
#define WUBU_CT_FIELD_MAX 64

typedef struct {
    char name[WUBU_CT_FIELD_MAX];
    char phone[WUBU_CT_FIELD_MAX];
    char email[WUBU_CT_FIELD_MAX];
} wubu_ct_contact_t;

typedef struct {
    wubu_ct_contact_t contacts[WUBU_CT_MAX_CONTACTS];
    int n;
} wubu_contacts_t;

/* Initialise (empty). */
int wubu_ct_init(wubu_contacts_t *ab);

/* Add a contact (name must be unique, case-insensitively). Returns 0/-1. */
int wubu_ct_add(wubu_contacts_t *ab, const char *name, const char *phone,
                const char *email);

/* Update a contact's phone and/or email by name (NULL leaves a field
 * unchanged). Returns 0/-1. */
int wubu_ct_update(wubu_contacts_t *ab, const char *name, const char *phone,
                   const char *email);

/* Remove a contact by name. Returns 0/-1. */
int wubu_ct_remove(wubu_contacts_t *ab, const char *name);

/* Find a contact by exact (case-insensitive) name. Returns index or -1. */
int wubu_ct_find(const wubu_contacts_t *ab, const char *name);

/* Case-insensitive substring search by name; fills `out` with matching
 * indices up to `max`. Returns total matches (may exceed max), -1 bad args. */
int wubu_ct_search(const wubu_contacts_t *ab, const char *query,
                   int *out, int max);

/* Query. */
int wubu_ct_count(const wubu_contacts_t *ab);
const char *wubu_ct_name(const wubu_contacts_t *ab, int idx);
const char *wubu_ct_phone(const wubu_contacts_t *ab, int idx);
const char *wubu_ct_email(const wubu_contacts_t *ab, int idx);

#endif
