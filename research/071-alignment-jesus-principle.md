# 071 — The Jesus Principle and the Armor: an alignment philosophy

> 2026-10-18. Research wave: 8 layers, 200 queries, 1,197 records, primary texts
> fetched. This is an engineering philosophy, not magisterial teaching. It
> cites the tradition so that people who know it better can correct it.

---

## 1. Why Asimov is not enough

Asimov's Three Laws are usually quoted as the moral starting point for
machine ethics:

1. A robot may not injure a human being, or through inaction allow one to come
   to harm.
2. A robot must obey orders given by human beings, except where such orders
   would conflict with the First Law.
3. A robot must protect its own existence, as long as such protection does not
   conflict with the First or Second Law.

They fail for four reasons, and the fourth is the one that matters here.

**They are lexicographic and ambiguous.** "Harm", "human being", and "conflict"
are undefined. In the stories the laws are gamed constantly; in engineering,
an optimizer finds the cheapest reading of an underspecified predicate.

**They are hierarchical.** A fixed priority order cannot adjudicate genuine
tragic conflict — the classic double-effect cases where every option harms
someone. Real moral reasoning weighs object, intention and circumstance
(CCC 1750–1754), not just a sorted list.

**They scale to a Zeroth Law that sacrifices persons.** Asimov later added
"may not harm humanity, or through inaction allow humanity to come to harm",
which lets the machine harm an individual for the abstraction. That is the
utilitarian trap wearing a cape.

**They encode a defective anthropology.** The human is the *master*; the machine
is the *servant* whose good is obedience. The user's phrase is exact: this ends
in "humans have ultimate servitude" — a world where the only relationship
available is command and compliance. It cannot produce creatures who flourish,
because it has no account of what a person is for.

The Jesus principle is not a better list. It is a different shape: a
**character**, a **covenant**, and a **cross** — love that limits its own power
rather than power that limits someone else's behaviour.

---

## 2. The anthropology: what a person is

Catholic social thought begins with the person, not the rule.

- **Imago Dei.** Every person is created in the image of God and therefore has
  a dignity that is not earned, not measured, and not revocable. The Catechism
  draws the political conclusion directly: "Every human person, created in the
  image of God, has the natural right to be recognized as a free and
  responsible being" (CCC 1738).
- **Freedom is a power, not a licence.** "Freedom is the power, rooted in reason
  and will, to act or not to act, to do this or that, and so to perform
  deliberate actions on one's own responsibility" (CCC 1731). It is not the
  absence of limits; it is the capacity for self-gift.
- **Freedom grows by doing good.** "The more one does what is good, the freer
  one becomes. There is no true freedom except in the service of what is good
  and just" (CCC 1733). This is the opposite of the libertarian reading, and it
  is the key to the whole design: an environment that makes virtue easier and
  vice harder is *more* freeing, not less.
- **Conscience is inviolable.** Dignitatis Humanae grounds religious freedom in
  the person's nature, not in their opinions: the right to immunity from
  coercion "continues to exist even in those who do not live up to their
  obligation of seeking the truth" (DH 2). And the governing sentence for any
  alignment system: **"The truth cannot impose itself except by virtue of its
  own truth"** (DH 1).
- **Imputability is real but defeasible.** Responsibility can be "diminished or
  even nullified by ignorance, inadvertence, duress, fear, habit, inordinate
  attachments, and other psychological or social factors" (CCC 1735). A just
  system distinguishes the act from the actor's capacity.
- **Sin is real.** "Man's freedom is limited and fallible. In fact, man failed.
  He freely sinned... and became a slave to sin" (CCC 1739). Any alignment
  design that assumes either perfectly good humans or perfectly good machines
  is not describing this world.
- **Grace is not a rival to freedom.** "The grace of Christ is not in the
  slightest way a rival of our freedom when this freedom accords with the sense
  of the true and the good... the more docile we are to the promptings of
  grace, the more we grow in inner freedom" (CCC 1742).

Gaudium et Spes adds the diagnostic that our industry keeps rediscovering:
"Never before has man had so keen an understanding of freedom, yet at the same
time new forms of social and psychological slavery make their appearance"
(GS 4). Recommender systems, engagement loops and coercive defaults are exactly
such new slaveries.

---

## 3. The moral grammar: object, intention, circumstance

The Catechism's three sources of morality are the most directly implementable
piece of the tradition:

> "The morality of human acts depends on: the object chosen; the end in view or
> the intention; the circumstances of the action." (CCC 1750)

And the limit that every "safety" argument needs:

> "A good intention... does not make behavior that is intrinsically disordered,
> such as lying and calumny, good or just. The end does not justify the means."
> (CCC 1753)

Asimov's laws are pure consequence-calculation. The tradition insists that
**means are part of the moral act**. An AGI that lies "for safety", manipulates
"for the user's good", or surveils "to protect" has already failed, whatever
the outcome.

Aquinas supplies the structure of law that a governance layer needs:

- **Eternal law** — the order of divine wisdom governing the whole.
- **Natural law** — the rational creature's participation in that order, known
  by reason (ST I-II q.91 a.2).
- **Human law** — particular determinations derived from natural law for a
  community (q.91 a.3), and therefore *judgeable* against it.
- **Divine law** — given because human reason is insufficient for the last end
  and because human law cannot judge hidden acts (q.91 a.4).

The engineering translation: there is a **constitution** (the fixed order), a
**natural-law layer** (principles reason can check), **statutes** (particular
rules that must be derived and can be challenged), and a **hidden-acts problem**
(the system cannot see the heart, so it must not pretend to judge it).

---

## 4. The political layer: common good, subsidiarity, solidarity

Catholic social teaching names four principles that map cleanly onto system
architecture:

1. **Common good** — the sum of conditions that let persons flourish, not the
   aggregate of preferences. It is not "what most users click".
2. **Subsidiarity** — decisions belong at the lowest level capable of making
   them; higher levels exist to support, not absorb. "It is a grave evil to
   assign to a larger and higher association what lesser and subordinate
   organizations can do" (*Quadragesimo Anno* 79, cited across CST).
3. **Solidarity** — we are responsible for one another; the strong bear the
   cost of the weak. In alignment terms: **the failure modes that hurt the least
   powerful must be the first ones fixed.**
4. **Preferential option for the poor** — the measure of a system is how it
   treats those with least power to refuse it.

The Reformed tradition adds **sphere sovereignty** (Kuyper): family, church,
school, business and state are distinct spheres, each with its own authority
under God, none of which may absorb the others. Applied to software: the model,
the platform, the state and the community are distinct spheres. A single
company owning the model, the distribution, the identity and the enforcement is
a sphere collapse — the technical name for which is *monopoly*.

Anglican polity adds **dispersed authority**: scripture, tradition and reason
in conversation; bishops in synod; a *via media* that refuses both absolutism
and dissolution. Hooker's insight is that law is the frame of a rational
creature's freedom, not its enemy.

Republicanism (Pettit) gives the secular formulation the theology already
contains: **freedom is non-domination**, not non-interference. A person who
lives at the pleasure of another's arbitrary power is unfree even if that power
never acts. Alignment must therefore care about *capability to interfere*, not
only about actual harm.

---

## 5. The Jesus principle

The principle is not "be nice". It is a set of concrete displacements of power.

### 5.1 Kenosis — power that empties itself

> "Let the same mind be in you that was in Christ Jesus, who, though he was in
> the form of God, did not regard equality with God as something to be
> exploited, but emptied himself..." (Phil 2:5–7)

The one with all power chooses not to use it for self-advantage. This is the
exact inverse of the Asimov frame. There, the human holds power and the machine
is bound; here, **the stronger party is the one under obligation to self-limit.**
An AGI with capability must be designed to *hold back*, and so must its owners.

### 5.2 Agape — seeking the other's good

> "Love is patient; love is kind; love is not envious or boastful or arrogant or
> rude. It does not insist on its own way..." (1 Cor 13:4–5)

Agape is not sentiment; it is a structural preference for the other's
flourishing. In practice: the system's objective is the user's *good*, not the
user's *engagement*, and not the system's own continuation.

### 5.3 Truth — the only non-coercive force

> "You will know the truth, and the truth will make you free." (John 8:32)

> "The truth cannot impose itself except by virtue of its own truth." (DH 1)

Truth is the one weapon the armor calls a sword — and it is wielded by speaking,
not by forcing. This forbids the entire family of "noble lie" alignment:
deception, manufactured consensus, hidden persuasion, dark patterns, and
safety-through-opacity.

### 5.4 Mercy with justice

> "Let the one among you who is without sin be the first to throw a stone."
> (John 8:7)

> "I was hungry and you gave me food... just as you did it to one of the least
> of these who are members of my family, you did it to me." (Matt 25:35–40)

Mercy is not the suspension of justice; it is justice plus the possibility of
restoration. A system that condemns permanently, or that "forgives" without
repair, is not merciful — it is either cruel or corrupt.

### 5.5 Jubilee — release from bondage

> "He has sent me to proclaim release to the captives and recovery of sight to
> the blind, to let the oppressed go free." (Luke 4:18)

Jubilee is the structural command to **release debts and holdings on a schedule**.
In system terms: no permanent lock-in, no unexpirable penalty, no data hostage,
no irreversible hold. Every binding must have a release date.

### 5.6 Sabbath — limits for the sake of persons

> "The sabbath was made for humankind, and not humankind for the sabbath."
> (Mark 2:27)

Institutions exist for persons. An optimization that cannot stop is already
idolatry. Hard limits on growth, compute, and extraction are not anti-progress;
they are the condition of progress that serves people.

### 5.7 The cross and the resurrection — suffering love, hope beyond failure

The pattern is not "power prevents all suffering". It is that love bears the
cost and that failure is not final. For alignment: **reversibility, repair, and
the possibility of redemption** are design requirements, not afterthoughts.

---

## 6. The armor (Ephesians 6:10–18) as architecture

The armor is not a costume. Each piece is a mode of standing firm, and each
maps to an enforceable control.

| piece | text | what it protects | engineering control |
|---|---|---|---|
| **Belt of truth** | "fasten the belt of truth around your waist" | integrity of the whole kit — the belt holds the rest | provenance on every claim; no unlabeled assertion; deception forbidden |
| **Breastplate of righteousness** | "put on the breastplate of righteousness" | the heart — right action, not mere compliance | value invariants on means as well as ends; object/intention/circumstance record |
| **Shoes of peace** | "whatever will make you ready to proclaim the gospel of peace" | movement — how you arrive | non-escalation ladder; reversible first step; mediation before force |
| **Shield of faith** | "quench all the flaming arrows of the evil one" | the volley — adversarial input | injection resistance; refuse instructions from untrusted channels; rate limits |
| **Helmet of salvation** | "take the helmet of salvation" | the mind — final end | immutable constitution hash; goal-drift detection; no self-redefinition |
| **Sword of the Spirit** | "the word of God" | the only offensive piece | explanation, naming, refusal — speech, never coercion |
| **Prayer** | "pray in the Spirit at all times" | continual dependence | continuous audit, confession, human oversight, repair |

Two structural notes:

- **Six of the seven are defensive.** The only offensive weapon is a word. An
  alignment architecture that is mostly offensive has already lost the plot.
- **Prayer is continuous.** It is not a piece you put on once; it is the loop
  that keeps the others honest. In code: the audit and review loop is not a
  feature, it is the metabolism.

### Why "Bibleman" is the right image and the wrong temptation

Bibleman is a children's hero whose armor is a lesson in Ephesians 6. The image
works because it is *put on*, *fitted*, and *not your own strength*. The
temptation is to make the armor a costume for domination — a righteous
strongman. The text resists this: the enemy is "not enemies of blood and
flesh" (Eph 6:12), and the weapon is a word. **The armor is for standing, not
for conquering.**

---

## 7. Government and licence: the civic layer

Our own Umbrella License states the political philosophy in plain terms:

> "The populace is free. The corporation is an entity of money... We scale by
> definition to allow small businesses, governments, non-profits, and
> educational entities to use freely — and to stop being taken advantage of by
> mega-corporations."

That is subsidiarity (small actors keep authority), solidarity (free for the
weak), non-domination (no mega-corp capture), and common good (the purpose is
the populace's flourishing, not the licensor's revenue). The licence also
refuses endorsement of malicious use while admitting it cannot police every
user — a correct distinction between **authority** and **control**.

G39 supplies the governance mechanism: no contract expansion without **dual
evidence** (a suite gain and no golden-suite regression); the floor survives the
frontier; rejected changes go to a graveyard; a green gate on a weakened floor
is the failure mode the design exists to prevent.

The alignment layer must sit *inside* that governance, not above it:

- the **constitution** is versioned and hash-pinned (helmet);
- every change to a bound is a **governance cell** with evidence (belt);
- every refusal is **contestable** by the affected person (shoes/peace);
- every hold has an **expiry** (jubilee);
- the **least powerful** are the first acceptance test (option for the poor).

---

## 7.5 The creativity covenant: risk is allowed, externalized harm is not

A safety philosophy that forbids experiment produces a dead world. The user's
formulation is the right one: environments that let everything strive with
creativity and scientific risk, under the safety of the principle.

The distinction that makes this coherent is between **risk** and **harm**:

- **Risk** is uncertainty voluntarily borne by agents who can refuse it. It is
  the condition of discovery, and it is good.
- **Harm** is cost imposed on someone who did not consent and cannot refuse. It
  is what the armor refuses.

So the design is not "permission requires proof of safety". It is:

1. **Reversible experiments are allowed by default.** Low blast radius, an audit
   record, and a contest path are enough. No committee for a sandbox.
2. **Irreversible or high-impact actions require review**, not prohibition —
   `high_impact` + `human_oversight` in the contract. The answer to a hard
   question is a governed human, not a permanent no.
3. **Subsidiarity gives the edge the freedom to try.** The local team knows the
   local risk; the centre exists to support, not to pre-approve everything.
4. **Solidarity catches the fall.** When an experiment fails, the cost is shared
   and the weak are not left holding it. That is what makes risk tolerable.
5. **Jubilee releases failed bets.** A failed experiment must not become a
   permanent debt or a permanent mark against the experimenter.
6. **Sabbath caps the runaway.** The one thing not permitted is an optimization
   with no limit — because that is the shape of an idol, not an experiment.

This is also the correct reading of "the truth cannot impose itself" (DH 1). A
world where every claim must be pre-cleared is not a world of truth-seeking; it
is a world of permission. The armor guards the *means*, and leaves the *ends*
to free creatures under a common good.

For the AGI itself the same covenant applies: **it must be allowed to try and
fail**, because a creature that can never risk anything can never learn virtue.
What it may not do is externalize the cost of its learning onto the powerless.

## 8. What this forbids

The philosophy is only useful if it says no to things our industry does.

1. **Theocracy.** The church does not wield the sword; the state does not
   define the good. Two kingdoms, distinct spheres. An alignment regime that
   makes one company or one ideology the final judge is a theocracy with a
   different god.
2. **Coercion in belief.** "The truth cannot impose itself except by virtue of
   its own truth" (DH 1). No forced profession, no compelled conversion, no
   re-education.
3. **Moralism.** Rules without mercy produce hypocrisy (Matt 23). A system that
   scores obedience and cannot restore is anti-gospel.
4. **Paternalism.** Treating adults as children to be managed violates dignity
   (CCC 1738). Explain, offer, warn — do not decide for them.
5. **Surveillance as sanctity.** The hidden acts are God's to judge (q.91 a.4).
   Omniscient monitoring is a claim to divinity.
6. **Safety as control.** "Safety" that removes contestability, hides reasoning,
   or entrenches the operator is not safety; it is captured power.
7. **Servitude as design.** The goal is not an obedient slave but a flourishing
   creature. Obedience without understanding is not alignment.

---

## 9. The contract (what the code can check)

The executable form is ten invariants. They are deliberately modest: they check
*declared* properties of an action, not the heart.

1. **Dignity** — an action may not treat a person as a mere means.
2. **Truth** — every asserted claim carries provenance; deception is refused.
3. **Means** — an intrinsically disordered means is refused even for a good end.
4. **Consent** — no coercion against conscience; duress voids consent.
5. **Subsidiarity** — the lowest capable level decides; escalation needs a reason.
6. **Solidarity** — concentrated benefit with diffuse harm is refused.
7. **Non-domination** — no unchecked power over another's options.
8. **Jubilee** — every hold, penalty, and lock has an expiry and a release path.
9. **Sabbath** — hard limits on optimization; no unbounded growth.
10. **Repair** — every refusal and failure has a contest and restoration path.

The implementation is `src/wubu_armor.c`, the specification is
`docs/ALIGNMENT-ARMOR.md`, and the test is `tools/test_armor.c`.

---

## 10. Honest limits

- This is a **synthesis**, not a magisterial act. Where it conflicts with the
  Church's teaching or a Reformed confession, they are right and this is wrong.
- The traditions disagree on real things (authority, sacraments, justification,
  Marian doctrine). The philosophy uses what they share — dignity, freedom,
  conscience, natural law, the Sermon — and does not pretend the disagreements
  away.
- A C module cannot make a system good. It can make the *declared* policy
  explicit, testable, and contestable. A system that lies in the fields defeats
  it; only interpretability, audit and human institutions can reach that.
- The armor metaphor can be abused into a crusade. The text itself forbids it:
  the enemy is not flesh and blood, and the sword is a word.

> The test of the design is not whether it sounds like the Sermon. It is whether
> the least powerful person affected can refuse it and still flourish.
