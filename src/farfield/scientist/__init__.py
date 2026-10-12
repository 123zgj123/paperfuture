"""A research programme whose experiments are real programs and whose goal is the question.

A project starts from a research question and ends with an answer to it, sub-question by
sub-question, or with the recorded reason it could not get one. In between:

* the question is **decomposed** into sub-questions, each with the evidence it needs and an
  honest statement of whether this lab can produce it;
* each reachable sub-question gets a **line of inquiry** that first **explores** — a sweep on
  seeds reserved for looking, tabulated by the runtime and never counted as evidence — then
  proposes hypotheses grounded in what it saw, each with its author's credence;
* a hypothesis is **confirmed** under pre-registration: the claim, metric, contrast, decision
  rule and code are frozen, and the seeds that decide are derived from the frozen digest;
  pilot seeds from the same range must first show the program runs, that its seeds are
  replicates, that it is not at a ceiling, and that the registered seeds could detect the
  registered effect;
* a decisive result answers nothing until it is **replicated** on changed conditions; a line
  whose hypotheses fail goes back to explore, and stops only after its last look, saying why;
* the budget of model calls is metered call by call, and a project does not stop early while
  a sub-question is unsettled and half the budget remains;
* every number comes from the stdout of an isolated process (no network, no view of the host,
  read-only datasets) and every verdict from an exact test with the threshold divided by the
  number of protocols the project froze;
* a review panel may judge each line's result — with a second adjudicator to show how much the
  judgment depends on who makes it — but it is advisory: it decides nothing, and nobody who
  proposes or designs research reads it;
* the project scores its own forecasts against its results, and a known-answer benchmark
  (``benchmarks/``, ``scripts/run_benchmark.py``) measures whether it concludes correctly and
  whether it ever asserts something wrong.

`notebook` is the only record, `lab` runs programs, `protocol` is what gets frozen,
`experiment` explores, develops and confirms, `stats` decides, `lines` says where each line of
inquiry stands, `minds` holds every model role, `loop` drives the programme and `report`
writes down what happened — including everything that did not work.
"""
