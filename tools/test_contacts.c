/* test_contacts.c -- Win98/XP Address Book tests. */
#include "wubu_contacts.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

int main(void)
{
    wubu_contacts_t ab;

    CHECK(wubu_ct_init(&ab) == 0);
    CHECK(wubu_ct_count(&ab) == 0);
    CHECK(wubu_ct_init(NULL) == -1);

    /* add contacts */
    CHECK(wubu_ct_add(&ab, "Alice Smith", "555-0100", "alice@example.com") == 0);
    CHECK(wubu_ct_add(&ab, "Bob Jones", "555-0101", "bob@example.com") == 0);
    CHECK(wubu_ct_add(&ab, "Alicia Keys", "555-0102", "alicia@example.com") == 0);
    CHECK(wubu_ct_count(&ab) == 3);

    /* dup name (case-insensitive) rejected */
    CHECK(wubu_ct_add(&ab, "alice smith", "999", "x@y.z") == -1);
    CHECK(wubu_ct_add(&ab, NULL, "p", "e") == -1);
    CHECK(wubu_ct_add(&ab, "x", NULL, "e") == -1);
    CHECK(wubu_ct_add(NULL, "x", "p", "e") == -1);

    /* find (case-insensitive) */
    CHECK(wubu_ct_find(&ab, "Alice Smith") == 0);
    CHECK(wubu_ct_find(&ab, "ALICE SMITH") == 0);
    CHECK(wubu_ct_find(&ab, "alice smith") == 0);
    CHECK(wubu_ct_find(&ab, "Nobody") == -1);
    CHECK(wubu_ct_find(&ab, NULL) == -1);
    CHECK(wubu_ct_find(NULL, "x") == -1);

    /* field access */
    CHECK(strcmp(wubu_ct_name(&ab, 0), "Alice Smith") == 0);
    CHECK(strcmp(wubu_ct_phone(&ab, 1), "555-0101") == 0);
    CHECK(strcmp(wubu_ct_email(&ab, 2), "alicia@example.com") == 0);
    CHECK(wubu_ct_name(&ab, 999) == NULL);
    CHECK(wubu_ct_phone(&ab, -1) == NULL);
    CHECK(wubu_ct_name(NULL, 0) == NULL);

    /* update: change phone, leave email (NULL) */
    CHECK(wubu_ct_update(&ab, "Alice Smith", "555-9999", NULL) == 0);
    CHECK(strcmp(wubu_ct_phone(&ab, 0), "555-9999") == 0);
    CHECK(strcmp(wubu_ct_email(&ab, 0), "alice@example.com") == 0);
    /* update email only */
    CHECK(wubu_ct_update(&ab, "Alice Smith", NULL, "new@example.com") == 0);
    CHECK(strcmp(wubu_ct_phone(&ab, 0), "555-9999") == 0);   /* unchanged */
    CHECK(strcmp(wubu_ct_email(&ab, 0), "new@example.com") == 0);
    CHECK(wubu_ct_update(&ab, "Nobody", "p", "e") == -1);
    CHECK(wubu_ct_update(&ab, NULL, "p", "e") == -1);
    CHECK(wubu_ct_update(NULL, "Alice Smith", "p", "e") == -1);

    /* search: case-insensitive substring by name */
    int out[8];
    /* "ali" matches Alice Smith + Alicia Keys = 2 */
    CHECK(wubu_ct_search(&ab, "ali", out, 8) == 2);
    /* "ALI" also matches (case-insensitive) */
    CHECK(wubu_ct_search(&ab, "ALI", out, 8) == 2);
    /* "jones" matches Bob Jones = 1 */
    CHECK(wubu_ct_search(&ab, "jones", out, 8) == 1);
    /* empty query matches all = 3 */
    CHECK(wubu_ct_search(&ab, "", out, 8) == 3);
    /* no match */
    CHECK(wubu_ct_search(&ab, "zzz", out, 8) == 0);
    /* truncation: max 1 returns count 2 but fills only 1 */
    CHECK(wubu_ct_search(&ab, "ali", out, 1) == 2);
    CHECK(wubu_ct_search(&ab, "ali", out, 0) == 2);
    CHECK(wubu_ct_search(&ab, "ali", NULL, 8) == -1);
    CHECK(wubu_ct_search(NULL, "ali", out, 8) == -1);

    /* remove */
    CHECK(wubu_ct_remove(&ab, "Bob Jones") == 0);
    CHECK(wubu_ct_count(&ab) == 2);
    CHECK(wubu_ct_find(&ab, "Bob Jones") == -1);
    /* remaining contacts intact */
    CHECK(wubu_ct_find(&ab, "Alice Smith") >= 0);
    CHECK(wubu_ct_find(&ab, "Alicia Keys") >= 0);
    CHECK(wubu_ct_remove(&ab, "Bob Jones") == -1);   /* already gone */
    CHECK(wubu_ct_remove(&ab, NULL) == -1);
    CHECK(wubu_ct_remove(NULL, "x") == -1);

    /* add after remove reuses the slot */
    CHECK(wubu_ct_add(&ab, "Carol White", "555-0103", "carol@example.com") == 0);
    CHECK(wubu_ct_count(&ab) == 3);

    if (fails == 0) printf("ALL CONTACTS TESTS PASSED\n");
    else printf("%d CONTACTS TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
