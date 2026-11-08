# Deciding what a project adds and keeps

Use these principles when a change adds a feature, a configuration option, a flag, an endpoint or an API field. Use them again when a project reviews what it already carries. A correctness review asks whether a change works. These principles ask whether the project should carry it, how much of it, and for how long.

This work belongs to **software product management**: deciding what a product contains over its whole life, from the first request to removal. Van de Weerd et al. describe its core areas as portfolio management, roadmapping, release planning and requirements management. [S103](SOURCES.md#stewardship) Open-source projects rarely have a product manager. Maintainers make these decisions in review, and this guide calls that work **stewardship**. It also draws on software evolution research, API compatibility practice and open-source proposal processes.

The question matters more when code is cheap to write. An agent can add a mirrored option in minutes. The project then tests, documents, supports and keeps it compatible for as long as it exists. Writing cost has fallen. Carrying cost has not. This framing is **house judgment**, built on the cost evidence in D5.

Each principle names its basis: **research**, **standard**, **practice**, **opinion** or **house judgment**. See [sources and limits](SOURCES.md#stewardship). The research measures the cost of options, technical debt and deprecation, and how often ideas fail. No cited study tests this review procedure itself. Bug fixes and refactoring that make code easier to change are outside its scope. [Fowler, S95](SOURCES.md#stewardship) The checklists need a reviewer's judgment. A linter cannot check them.

The guide follows a feature through its life: before adding, keeping the surface small, owning it, checking its use, and changing or removing it.

## Before adding

### D1. Name who needs it and what task it serves

A proposal should name a real user or caller and a concrete task they cannot do well today. "It might be useful" names no one. Parity with a similar component is not a need by itself. Ask for the request, issue, workload or caller that motivates the change. For an internal API field, the caller can be another component, but it should exist now.

**Research:** most product ideas do not deliver the value their authors expect. Kohavi and Thomke report that at Microsoft about one third of controlled experiments proved effective. At Google and Bing the share was 10% to 20%. These are company reports in a practitioner magazine, not a peer-reviewed study. They still show that the author's confidence is weak evidence of need. [S86](SOURCES.md#stewardship)

**Practice:** open-source proposal processes ask for the problem first. The Rust RFC template asks for "several specific use cases" in its Motivation section. PEP 1 says a proposal without sufficient motivation may be rejected. Go asks that a language change address an important issue for many people. [S89–S91](SOURCES.md#stewardship) Product-discovery practice frames the same question as value risk and as the customer's job to be done. [Cagan, S101, and Christensen et al., S100](SOURCES.md#stewardship)

**Standard:** ISO/IEC/IEEE 29148 lists "necessary" among the characteristics of a well-formed requirement. [S88](SOURCES.md#stewardship)

### D2. Say what happens without it

Describe what the user does today and what would happen if the change were declined. Compare the change with the cheapest alternatives: an existing feature, a better default, documentation, an example script or a fix to the underlying problem. If nothing bad happens without it, the change is optional. Optional code needs a stronger case than required code.

**Practice:** the Rust RFC template asks "What is the impact of not doing this?" and asks for alternatives. The Kubernetes enhancement template asks why the change should not be made and which alternatives were ruled out. [S89, S92](SOURCES.md#stewardship) Shape Up advises clarifying what is actually going wrong before accepting a request as stated. A narrower problem often fits a much smaller solution. [Singer, S98](SOURCES.md#stewardship)

### D3. Match the effort to what the problem is worth

Decide how much the problem is worth before choosing a design. Compare it with the other work the same people could do. A real need can still wait if more important work would be delayed.

**Practice:** Shape Up sets an appetite before the design: a fixed amount of time the problem is worth. The scope then shrinks to fit. [S98](SOURCES.md#stewardship) Fowler names the cost of delay: the value lost because effort went to a feature instead of other work. [S95](SOURCES.md#stewardship) Release planning is a core area of software product management. [S103](SOURCES.md#stewardship)

## Keep the surface small

### D4. Ship the smallest version that serves the need

Once the need is real, build only what that need requires. Prefer a good default to an option. Prefer one option to several. Leave out parameters, fields and modes that nobody has asked to change. Adding later is cheap. Removing later is not.

**Research:** Xu et al. studied real configuration settings in a commercial storage system and two open-source projects. Only 6.1% to 16.7% of parameters were set by most users. Up to 54.1% were rarely set by anyone. Their simplification guidelines could hide or remove 51.9% of the commercial system's parameters with little effect on existing users. [S81](SOURCES.md#stewardship) Yin et al. studied 546 real misconfigurations. Most came from parameter settings, and many caused crashes, hangs or severe slowdowns. [S82](SOURCES.md#stewardship) Abal et al. analysed 42 Linux bugs that appear only under some configurations. Variability made these bugs more complex. [S83](SOURCES.md#stewardship)

**Practice/opinion:** Bloch's API design guidance says "When in doubt leave it out" because "you can always add, but you can never remove". [S93](SOURCES.md#stewardship) Pennington argues that each preference costs discoverability, testing and good defaults. [S97](SOURCES.md#stewardship)

## Own it

### D5. Count the ongoing cost and name who carries it

Estimate what the project pays every month the feature exists. That includes tests, documentation, support questions, compatibility, security review and the attention of future readers. Name the person or team who will answer its issues and keep it working. A feature that nobody owns still costs the project. Its cost lands on whoever touches the code next.

**Research:** in a longitudinal survey of 43 developers, Besker et al. found that developers lost about 23% of their working time to technical debt. Extra testing was the most common cost. These figures are self-reported. [S84](SOURCES.md#stewardship) Lehman's law of increasing complexity, derived from observing large systems, says complexity grows with each change unless work is done to reduce it. [S85](SOURCES.md#stewardship) Xu et al. found that up to 48.5% of configuration issues were users struggling to find or set the right parameter. [S81](SOURCES.md#stewardship)

**Practice:** Fowler names the cost of carry. A feature adds complexity that slows every later change, even when it works. [S95](SOURCES.md#stewardship) *Software Engineering at Google* states that code is a liability, not an asset. Its value is the functionality it delivers. [S109](SOURCES.md#stewardship) Naming an owner for each feature is **house judgment**.

## Check its use

### D6. Find out whether it is used

After release, check whether the feature serves the need that justified it. Use the evidence the project has: issues and questions, code search across known dependents, download or request logs, and opt-in telemetry where the project has it. Respect the project's privacy commitments. Absence of complaints is weak evidence of use. Set a date for the check when the feature is added.

**Research:** Xu et al. measured which parameters real users set, and up to 54.1% were rarely set by anyone. Developers had added far more parameters than most users needed. [S81](SOURCES.md#stewardship)

**Practice:** Cox argues that bug reports show only problems users notice, and surveys miss rarely used features. He proposes minimal, public, opt-in usage counters for the Go toolchain. [S110](SOURCES.md#stewardship) Kohavi and Thomke show that measured outcomes often contradict expectations. [S86](SOURCES.md#stewardship) Uber's Piranha tool found and removed code for 1,381 stale feature flags, 17% of all flags at the time. [S87](SOURCES.md#stewardship)

## Change and remove

### D7. Decide before merging how it could be removed

Before merging, say how the feature would be retired if it turns out unneeded. Note who depends on it, whether removal breaks a published contract and how users would migrate. For an experiment or a temporary flag, record an owner and a review date when it is added. Mark experimental features as experimental in the documentation.

**Practice:** Hyrum's Law observes that with enough users, someone will depend on every observable behavior, whatever the contract promises. [S94](SOURCES.md#stewardship) Hodgson advises treating feature toggles as inventory with a carrying cost. Some teams add a removal task or an expiry date when they create a toggle. [S96](SOURCES.md#stewardship) *Software Engineering at Google* advises planning for retirement when a system is built. [S109](SOURCES.md#stewardship)

### D8. Deprecate with notice, a path and a deadline

When a feature has users, announce the deprecation, give the replacement or workaround and set a removal date or version. Follow the project's compatibility promise. A warning without a deadline rarely moves users.

**Research:** Sawant et al. studied more than 25,000 clients of five popular Java APIs and 60 clients of the JDK. Only a small fraction of clients reacted to deprecation. [S105](SOURCES.md#stewardship) Bogart et al. compared 18 software ecosystems. Their policies on breaking changes shift costs between maintainers and users, according to each ecosystem's values. [S104](SOURCES.md#stewardship) Brito et al. asked developers of popular Java libraries why they broke APIs. Wanting a simpler API with fewer elements was one of the main reasons. [S106](SOURCES.md#stewardship)

**Standard/practice:** Semantic Versioning requires a deprecation to appear in at least one minor release before removal in a major release. [S107](SOURCES.md#stewardship) Go promises that programs written to Go 1 keep working, and lists exceptions such as security fixes. [S108](SOURCES.md#stewardship) *Software Engineering at Google* found that advisory deprecation rarely moves existing users. Deprecation with a deadline and an owner works better. [S109](SOURCES.md#stewardship)

## At every stage

### D9. Record the decision

Write down why a feature was accepted, narrowed, declined, deprecated or removed. A recorded "not now" saves the next contributor from proposing it again without new evidence. A declined idea can return when a real user brings the missing need.

**Practice:** PEP 1 asks authors to record rejected ideas so that discussions do not repeat. The Go proposal process aims for a recorded evaluation with a clear answer, and declines proposals that lack consensus. [S90, S91](SOURCES.md#stewardship) Getting Real advises answering feature requests with "not now" by default, and making each feature prove itself. Shape Up recommends a soft no for raw ideas until their value is clear. [S98, S99](SOURCES.md#stewardship)

## Checklist for a pull request that adds something

Apply this checklist to each new feature, option, flag, endpoint or API field. A missing answer is a question for the author, not an automatic rejection.

1. **Who needs it?** Name the user, caller or issue that asked for it.
2. **What task does it serve?** Describe the concrete thing they will do with it.
3. **What happens without it?** Describe today's workaround and the cheapest alternative.
4. **Is it worth the effort now?** Compare it with other work the same people could do.
5. **Is this the smallest version?** Check each parameter, field and mode against the stated need. Remove the ones nobody asked for.
6. **Who owns it, and what does it cost to keep?** Name the owner. List new tests, documentation, compatibility promises and interactions with other options.
7. **How would we know it is used?** Name the evidence and the date to check it.
8. **How would we remove it?** Name who would break and how they would migrate.
9. **Is the decision recorded?** Put the reasoning in the PR or issue so the next reviewer can find it.

For mirrored code, ask questions 1 to 3 again for the new location. A need in one component does not establish a need in another. This is **house judgment**.

## Checklist for reviewing what a project already carries

Use this checklist at a release or when touching an old feature.

1. **Is it used?** Check the evidence named when it was added, or find some now.
2. **Does it still have an owner?** If not, find one or plan its removal.
3. **What does it cost?** Note recent bugs, support questions and the work it slowed.
4. **Should it go?** If so, deprecate it with notice, a replacement or workaround, and a removal version.
5. **Is the decision recorded?** Write down why it stays or goes.

**Worked example.** A pull request adds a `merge` option to a lookup endpoint because a sibling endpoint has one. Nobody has asked for it on this endpoint, and the default already returns the result most callers want. The reviewer asks who would set the option and what they would do with merged output. With no answer, the option is left out and the reason is recorded in the PR. If a caller later needs merged results, they can propose it with that use case.
