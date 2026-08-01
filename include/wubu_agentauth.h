/*
 * wubu_agentauth.h -- Inter-agent message authentication (AG07).
 */
#ifndef WUBU_AGENTAUTH_H
#define WUBU_AGENTAUTH_H

unsigned long long wubu_agent_mac(const char *from, const char *to,
                                  const char *payload, const char *secret);
int wubu_agent_verify(const char *from, const char *to, const char *payload,
                      const char *secret, unsigned long long claimed_mac);

#endif
