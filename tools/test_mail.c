/* test_mail.c -- Win98/XP Email client tests. */
#include "wubu_mail.h"
#include <stdio.h>
#include <string.h>

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d)\n", #c, __LINE__); fails++; } } while (0)

int main(void)
{
    wubu_mailbox_t mb;

    CHECK(wubu_ma_init(&mb) == 0);
    CHECK(mb.n_msgs == 0);
    CHECK(wubu_ma_init(NULL) == -1);

    /* folder names */
    CHECK(strcmp(wubu_ma_folder_name(WUBU_MAIL_INBOX), "Inbox") == 0);
    CHECK(strcmp(wubu_ma_folder_name(WUBU_MAIL_SENT), "Sent") == 0);
    CHECK(strcmp(wubu_ma_folder_name(WUBU_MAIL_ARCHIVE), "Archive") == 0);
    CHECK(strcmp(wubu_ma_folder_name(WUBU_MAIL_TRASH), "Trash") == 0);
    CHECK(wubu_ma_folder_name((wubu_mail_folder_t)99) == NULL);

    /* receive into inbox */
    int m1 = wubu_ma_receive(&mb, WUBU_MAIL_INBOX, "alice@x.com", "me@x.com",
                             "Hello", "Hi there");
    CHECK(m1 == 1);
    int m2 = wubu_ma_receive(&mb, WUBU_MAIL_INBOX, "bob@x.com", "me@x.com",
                             "Meeting", "Tomorrow?");
    CHECK(m2 == 2);
    CHECK(wubu_ma_count(&mb, WUBU_MAIL_INBOX) == 2);
    CHECK(wubu_ma_unread(&mb, WUBU_MAIL_INBOX) == 2);   /* both unread */

    /* new messages arrive unread */
    const wubu_mail_msg_t *g = wubu_ma_get(&mb, m1);
    CHECK(g != NULL);
    CHECK(g->read == 0);
    CHECK(strcmp(g->from, "alice@x.com") == 0);
    CHECK(strcmp(g->subject, "Hello") == 0);
    CHECK(strcmp(g->body, "Hi there") == 0);
    CHECK(g->ts > 0);

    /* mark read -> unread count drops */
    CHECK(wubu_ma_set_read(&mb, m1, 1) == 0);
    CHECK(wubu_ma_unread(&mb, WUBU_MAIL_INBOX) == 1);
    CHECK(wubu_ma_count(&mb, WUBU_MAIL_INBOX) == 2);   /* count unchanged */
    CHECK(wubu_ma_set_read(&mb, m1, 0) == 0);          /* mark unread again */
    CHECK(wubu_ma_unread(&mb, WUBU_MAIL_INBOX) == 2);

    /* bad receive */
    CHECK(wubu_ma_receive(&mb, WUBU_MAIL_INBOX, NULL, "me", "s", "b") == -1);
    CHECK(wubu_ma_receive(&mb, WUBU_MAIL_INBOX, "f", NULL, "s", "b") == -1);
    CHECK(wubu_ma_receive(&mb, (wubu_mail_folder_t)99, "f", "t", "s", "b") == -1);
    CHECK(wubu_ma_receive(NULL, WUBU_MAIL_INBOX, "f", "t", "s", "b") == -1);

    /* move between folders */
    CHECK(wubu_ma_move(&mb, m1, WUBU_MAIL_ARCHIVE) == 0);
    CHECK(wubu_ma_count(&mb, WUBU_MAIL_INBOX) == 1);
    CHECK(wubu_ma_count(&mb, WUBU_MAIL_ARCHIVE) == 1);
    CHECK(wubu_ma_unread(&mb, WUBU_MAIL_INBOX) == 1);   /* m2 still unread */
    CHECK(wubu_ma_unread(&mb, WUBU_MAIL_ARCHIVE) == 1); /* m1 unread in archive */
    CHECK(wubu_ma_move(&mb, 999, WUBU_MAIL_TRASH) == -1);
    CHECK(wubu_ma_move(&mb, m2, (wubu_mail_folder_t)99) == -1);
    CHECK(wubu_ma_move(NULL, m2, WUBU_MAIL_TRASH) == -1);

    /* send a message */
    int s1 = wubu_ma_receive(&mb, WUBU_MAIL_SENT, "me@x.com", "carol@x.com",
                             "Re: Hello", "On my way");
    CHECK(s1 == 3);
    CHECK(wubu_ma_count(&mb, WUBU_MAIL_SENT) == 1);

    /* delete (trash) removes from store */
    CHECK(wubu_ma_move(&mb, m2, WUBU_MAIL_TRASH) == 0);
    CHECK(wubu_ma_count(&mb, WUBU_MAIL_TRASH) == 1);
    CHECK(wubu_ma_delete(&mb, m2) == 0);
    CHECK(wubu_ma_count(&mb, WUBU_MAIL_TRASH) == 0);
    CHECK(wubu_ma_get(&mb, m2) == NULL);   /* gone */
    CHECK(wubu_ma_delete(&mb, 999) == -1);
    CHECK(wubu_ma_delete(NULL, 1) == -1);

    /* remaining messages intact */
    CHECK(wubu_ma_count(&mb, WUBU_MAIL_ARCHIVE) == 1);
    CHECK(wubu_ma_count(&mb, WUBU_MAIL_SENT) == 1);
    CHECK(wubu_ma_count(&mb, WUBU_MAIL_INBOX) == 0);

    /* set_read / get bad */
    CHECK(wubu_ma_set_read(&mb, 999, 1) == -1);
    CHECK(wubu_ma_set_read(NULL, 1, 1) == -1);
    CHECK(wubu_ma_get(NULL, 1) == NULL);
    CHECK(wubu_ma_get(&mb, 999) == NULL);

    /* count/unread NULL + bad folder */
    CHECK(wubu_ma_count(NULL, WUBU_MAIL_INBOX) == -1);
    CHECK(wubu_ma_unread(NULL, WUBU_MAIL_INBOX) == -1);
    CHECK(wubu_ma_count(&mb, (wubu_mail_folder_t)99) == -1);
    CHECK(wubu_ma_unread(&mb, (wubu_mail_folder_t)99) == -1);

    if (fails == 0) printf("ALL MAIL TESTS PASSED\n");
    else printf("%d MAIL TEST(S) FAILED\n", fails);
    return fails ? 1 : 0;
}
