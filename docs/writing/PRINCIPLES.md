# Writing that helps people act

Use these principles for English documentation and interface text. Choose the reader and task first. A research explanation, a command reference and a confirmation dialog need different amounts of detail. This guide is opinionated about clarity and accuracy; exact word limits and punctuation bans are optional house styles. For text whose goal is understanding, such as algorithm articles and conceptual explanations, also read [TEACHING.md](TEACHING.md).

Each principle names its basis: **research**, **standard**, **practice**, or **house judgment**. See [sources and limitations](SOURCES.md). A cited source supports the stated principle, not every possible implementation of it. These principles require contextual review; the linter does not certify them.

## P1. Give the reader the information needed for the task

Start with the result, question or action that matters in this context. Retain prerequisites, consequences and qualifications that affect a decision. Remove repetition before removing substance. **Research:** Morkes and Nielsen's web-reading experiments and Loranger and Moran's study of domain experts support concise, scannable, factual content. They do not establish a universal word budget. [S1, S2](SOURCES.md#research)

For example, a short diagnostic can say “Output could not be saved. Choose a writable directory with `--output`.” A tutorial also needs a working command and an explanation of the expected output. A proof can need several paragraphs of connected reasoning.

Weinreich et al.'s longitudinal web-use study also supports scan-friendly presentation while documenting variation between readers and sites. Do not infer a universal reading percentage. [S19](SOURCES.md#research)

## P2. Organize documentation around the reader's need

Distinguish learning a workflow, completing a task, looking up a contract, and understanding a concept. Link these forms instead of making every page do all four jobs. **Practice:** Diátaxis provides this tutorial/how-to/reference/explanation framework. It is a design framework, not a controlled comparison proving one mandatory site layout. [S3](SOURCES.md#documentation-and-language)

**Research:** Carroll et al.'s *The Minimal Manual* supports task-oriented learning and recovery support in its word-processing experiments. It motivates actionable tutorials, not removing conceptual explanations or complete reference material. [S18](SOURCES.md#research)

## P3. Use precise, familiar language

Prefer direct sentences and a clear actor when responsibility matters. Define unfamiliar terms at the point of need. Keep established technical language for expert audiences and reproduce actual API names accurately. Passive voice is useful when the outcome matters more than the actor. **Practice:** Google's developer guide and Microsoft's global writing guidance support accessible language and consistent terminology. [S4, S5](SOURCES.md#documentation-and-language)

Prefer “The server rejected the expired token” when diagnosis matters; “Nothing was saved” is appropriate when the immediate concern is whether data changed. Do not replace correct domain terminology merely because it contains an underscore.

## P4. Preserve facts and uncertainty

Verify defaults, limits, units and promises against their owning source. Distinguish an observed outcome from an inference or a prediction. Never turn uncertainty into certainty to satisfy a style check. **Research/practice:** factual, verifiable content matters to expert readers [S2](SOURCES.md#research). Requiring a code/config reference for implementation claims is our **house verification practice**, not a result measured by that study.

“The request timed out. Its completion status is unknown. Check the job history before retrying” preserves a meaningful uncertainty. “The operation failed” would invent an outcome. Words such as “may” and “might” remain valid when they express real uncertainty.

## P5. Make errors useful and truthful

Identify the problem and give a feasible next step when known. Include the cause only when known and useful. Keep relevant input, field or file information close to the message, subject to security and privacy constraints. **Standard:** WCAG 3.3.3 requires known input-error suggestions, with a security/purpose exception. **Practice:** GOV.UK and NN/g support specific, constructive error messages. [S6–S8](SOURCES.md#errors-and-accessibility)

Avoid a bare “Oops.” Do not invent a cause, promise recovery, blame the reader or mechanically demand three sentences. A diagnostic identifier can be visible and copyable when readers need it; a fold is an optional presentation choice.

## P6. Give structure and controls meaningful names

Use descriptive headings, labels and link text. Keep terminology consistent for the same function. Choose lists for parallel items, numbered steps for ordered actions and tables for comparisons. **Standard:** WCAG 2.4.6 covers descriptive headings/labels; 2.4.4 permits link purpose from text or its programmatically determined context. **Practice:** Google supports descriptive links and structured documentation. [S4, S9, S10](SOURCES.md)

Prefer “Download the report” to an isolated “Click here.” A question can be the clearest heading on a question page. Neither a two-word opening nor a fixed number of list-item lines is required.

## P7. Keep decision-critical information visible

Show prerequisites and consequences before the action they govern. Offer optional background separately with a descriptive disclosure label. **Practice:** progressive disclosure and GOV.UK's Details guidance support deferring secondary material, while retaining information most readers need. [S11, S12](SOURCES.md#interface-presentation)

For a destructive action, name the object and effect. Whether confirmation or undo is appropriate depends on the product's risk and recovery model. This writing guide does not change repository approval or security policy.

### Multi-step help and consequential actions

When readers can take different routes, name each route where they choose it. Put an exception beside the step it changes. Separate doing the task from checking its result. Someone returning after an interruption should be able to recognize their current state, identify who acts next, and continue without repeating a consequential action. [W3C cognitive accessibility patterns, S28](SOURCES.md#cognitive-accessibility)

Use emphasis for the control someone needs to find or the consequence that changes their decision. Put that consequence before the action. Keep supporting details in nearby ordinary text. A warning that competes with several other highlighted passages can be hard to find. Inspect the rendered page at phone and desktop widths. Warning length is an editorial choice. [GOV.UK warning text](https://design-system.service.gov.uk/components/warning-text/)

**Worked example.** A booking service accepts new requests and lets people resume saved drafts. Its initial help says, “Start a request or open your draft, select a date, and send it to the coordinator, who will review it. Dates cannot be changed after sending. Pending requests are under My bookings.”

A reader who returns midway can use these instructions:

> **Choose a date**
>
> For a new booking, select **New request**. To continue a saved one, open its draft under **My bookings**. Choose the date.
>
> **Send for review**
>
> You cannot change the date after sending. Check it, then select **Send**.
>
> **Check your request**
>
> Open **My bookings**. If your request says *Pending*, the coordinator has it for review.

When reviewing a page, ask someone to find their route, explain the consequence in their own words, and pick up the task after a pause. Observe where they look or repeat a step. Use those observations to revise the page. [W3C guidance on testing focus, S28](SOURCES.md#cognitive-accessibility)

## P8. Describe states accurately and make feedback perceivable

Distinguish loading, empty results, filtered results, errors and completed work. Explain an empty state when its cause or next step is not apparent. Expose qualifying dynamic status messages to assistive technology without unnecessarily moving focus. **Practice/standard:** Carbon's empty-state guidance and WCAG 4.1.3. [S13, S14](SOURCES.md#interface-presentation)

Do not show a made-up percentage or an ETA merely to meet a time threshold. “None” can be correct in a compact table cell with an unambiguous column heading; a new-user screen usually needs more help.

## P9. Evaluate the rendered experience

Check text in context at relevant viewport sizes, zoom, text-spacing overrides and with long/localized content. Keep essential labels and actions discoverable. **Standard:** WCAG reflow and text-spacing criteria have specific scope and exceptions. Its AAA visual-presentation criterion is distinct from AA. **House implementation choice:** choose comfortable prose widths and typography by testing the actual fonts and layout. [S15–S17](SOURCES.md#interface-presentation)

No font-independent `34em` cap, single phone screenshot or word-count proxy establishes usability or WCAG conformance. Automated accessibility checks complement manual keyboard, reading-order and task review.

### Diagrams in help pages

When a page includes a figure, edit the figure and its surrounding text together. Start with the question the reader should be able to answer by tracing the drawing. Show the relationship that supplies the answer. For a dependency, draw the connections and show whether every input is required or any one is enough. For a sequence, show its order. Preserve any condition that changes what the reader can conclude. [Google on technical illustrations, S29](SOURCES.md#diagrams)

Use short labels for the inputs, conditions and outcome. Move explanations, examples and troubleshooting into nearby prose. Keep a consequence visible before the action it governs. Give a complex figure a short identifying text alternative and make the relationships and essential details available in text. [W3C WAI on complex images, S30](SOURCES.md#diagrams)

**Worked example.** To explain when a library hold becomes ready for pickup, draw two inputs, “Book returned” and “Hold is first in queue”, converging at **Both required**. Draw an arrow from there to “Staff shelve the book”, then to “Ready for pickup”. Both inputs can be true while the book is still at the desk. Put the explanation of queue positions in the page text.

**House review practice.** Trace the finished figure at desktop and phone widths. Check that each connector meets its target, short labels stay readable, and the final box names the outcome once. If a label wraps with a single dangling word, try shorter wording or a different box width. Keep the facts intact. If you cannot inspect the rendered figure, say which checks remain open.

## P10. Review meaning before mechanical style

Before finishing, read the text as its intended reader: can they find the point, understand the state and take the right action? Check factual claims and examples, then run the repository's selected lint rules. This sequence is our **house workflow**, motivated by P1–P9. A clean lint result is not evidence of correctness, accessibility or successful comprehension.

Opinions are welcome when labeled and scoped. Teams can choose punctuation, capitalization, tone and tighter UI budgets. Keep those choices in a named local profile; evaluate false positives and user outcomes before treating them as universal rules.
