# HPML Final Project

## Submission Instructions

High Performance Machine Learning

*Spring 2026 · Dr. Kaoutar El Maghraoui · Columbia University*

> **Submit on CourseWorks**
>
> Final project assignment link:
>
> https://courseworks2.columbia.edu/courses/237605/assignments/1654215

## 1. What to Submit

Four artifacts both submitted to CoursWorks and Github:

1. **Public GitHub repository** with code, README, configs, scripts, and a deliverables/ folder containing the final report PDF and the presentation file.
2. **Public experiment-tracking dashboard** (Wandb recommended; MLflow, TensorBoard, Comet, Neptune, or equivalent).
3. **Final report PDF** (IEEE format, max 6 main pages) — committed to deliverables/ AND uploaded to CourseWorks.
4. **Final presentation deck** (PPT or PDF) — committed to deliverables/ AND uploaded to CourseWorks.

## 2. GitHub Repository

Public at submission time. All team members must commit under their own accounts. Commit history is graded. Empty commits, end-of-project bulk commits, and rename-only commits do not count toward individual contribution.

**Required at the root:**

- README.md — use the provided HPML README template, with all sections filled in.
- requirements.txt or environment.yml pinned to the versions used for reported results.
- configs/, scripts/, src/ — everything needed to reproduce reported figures and tables.
- **deliverables/** — final report PDF and presentation file. The same files must be uploaded to CourseWorks.
- LICENSE (MIT, Apache-2.0, or BSD recommended).

Do not commit datasets or checkpoints over 100 MB — use external storage and link to it from the README.

## 3. Experiment Tracking

Log all runs to a public dashboard. Wandb is the default; MLflow, TensorBoard (incl. TensorBoard.dev), Comet, Neptune, ClearML, Aim, or vendor-managed services (Vertex AI Experiments, SageMaker Experiments) are all acceptable. If your platform cannot produce a public link, commit a static export under results/dashboard/ instead.

Log model config, training/validation curves, system metrics (GPU utilization, memory, throughput), hyperparameter sweeps, and tag runs so baseline vs. optimized are easy to compare. Include the public link in the README and the report. Verify it opens in an incognito browser before submitting.

## 4. Final Report

- **Format:** IEEE conference LaTeX template (ieee.org/conferences/publishing/templates), written in Overleaf. Two-column, default IEEE fonts, no layout customization.
- **Length:** max 6 pages of main content. References and appendices may extend it.
- **File:** TeamName_HPML_Final_Report.pdf — in deliverables/ and on CourseWorks.

**Recommended structure (subsections welcome):**

- I. Introduction — background, problem statement, scope.
- II. Literature Review.
- III. Methodology — data, model, optimization scope (training / inference / both), profiling tools, evaluation metrics.
- IV. Experimental Results — setup, before/after comparison, analysis.
- V. Discussion — interpretation, limitations, future work.
- VI. Conclusion.
- VII. References.

## 5. Presentation (10 min + Q&A)

It is recommended to use the the provided HPML class template. Place the deck in deliverables/ and upload to CourseWorks. Include the GitHub URL on the title and final slide. Cameras ON for all presenters. For teams of 3–4, split into consecutive segments. Do not switch presenters mid-section.

| Section | Time | Content |
|---|---:|---|
| 1. Title & Team | 0:30 | Project title, team, GitHub URL. |
| 2. Motivation & Problem | 1:00 | Why this matters; what the workload is. |
| 3. Technical Approach | 3:00 | Model, dataset, optimizations, profiling methodology. |
| 4. Results & Evaluation | 2:00 | Before/after metrics, key plots from your tracking dashboard. |
| 5. Demo or Visualization | 1:00 | Live demo, profiler trace, or animated comparison. |
| 6. Lessons Learned | 1:00 | What surprised you; what didn't work. |
| 7. Next Steps | 0:30 | Concrete future work. |
| 8. Teamwork | 0:30 | Who did what; how you collaborated. |

## 6. Grading

The final project counts as 30% of the overall course grade. Within the project:

| Component | Weight | Notes |
|---|---:|---|
| Project Proposal/Project Mid-point (Best grade out of the two will be picked) | 5% | Scope, feasibility, team roles. |
| Code, Profiling, Documentation, Reproducibility | 25% | GitHub repo. Documented code 80%, README 20%. |
| Experiment Tracking & Dashboard | 20% | Public dashboard with baseline vs. optimized comparisons. |
| Technical Contributions, Methodology, Analysis | 25% | Depth of optimization study, soundness, insight. |
| Presentation & Q&A | 10% | Clarity, timing, response to questions. |
| Final Report Quality | 15% | IEEE formatting, completeness, technical writing. |
| Bonus (optional) | up to +10% | See Section 7. |

All four artifacts are graded at the team level. Individual grades may be adjusted up or down based on commit history, the teamwork slide, and end-of-semester peer evaluation. Significantly unequal contributions documented in commit history will result in different individual grades.

## 7. Bonus Opportunities

Up to 10 percentage points of bonus credit on your final project grade, with evidence:

- **Blog post or article:** (Medium, Substack, or technical-blog). Provide the published URL.
- **Open-source contribution:** A merged or actively reviewed PR to PyTorch, vLLM, TGI, AIHWKIT, FMS, or other notable open-source projects.
- **Publishable novel results:** Meeting the bar for a workshop or conference paper.
- **Re-usable Public artifact:** Library, dataset, or benchmark released under a permissive license.

A blog post that is a verbatim copy of the report does not qualify.

## 8. AI Use Policy Reminder

**The HPML AI Use Policy** applies to the final project — including code, profiling analysis, the report, and the presentation. Read it in full on CourseWorks before submitting. Key points:

- **Permitted:** AI tools (Claude, ChatGPT, Copilot, etc.) as a learning aid for clarification, debugging, and polishing prose you have already drafted yourself.
- **Not permitted:** using AI to generate your profiling interpretations, performance reasoning, or written analysis.
- **Mandatory disclosure:** include the disclosure block from the AI Use Policy in your README under "AI Tool Use" AND as a brief appendix or footnote in the final report.
- **Detection & oral verification:** submissions may be processed through GPTZero, Copyleaks, Turnitin AI, Codequiry, and MOSS. Course staff may request a 10–15 minute oral check-in. Inability to explain your own work results in a zero on the project, independent of the written score.
- **Consequences:** undisclosed AI use is a zero on the project (first instance) or course failure (repeat). Fabricated disclosure goes directly to the Academic Integrity Committee.

If you used AI substantively, disclose it. If AI did the thinking the project is supposed to build, that is not allowed.

## 9. Submission Checklist

- [ ] Public GitHub repo URL submitted on CourseWorks; opens from a logged-out browser.
- [ ] README follows the HPML template; quickstart command reproduces the headline result.
- [ ] requirements.txt pinned to the versions used.
- [ ] Experiment-tracking dashboard is public and linked in the README.
- [ ] deliverables/ folder contains the final report PDF and the presentation file.
- [ ] Same report PDF and presentation file uploaded as CourseWorks attachments (matching content).
- [ ] Final report is IEEE two-column, ≤ 6 main pages.
- [ ] Presentation deck has GitHub URL on title and final slide.
- [ ] Teamwork slide names who did what.
- [ ] AI use disclosure block included in the README and the final report (per the AI Use Policy).
- [ ] All team members have committed under their own GitHub accounts.

## 10. Late Policy & Contact

Late submissions are penalized 10 percentage points per 24 hours, up to 72 hours; not accepted afterward. Presentation slots cannot be rescheduled. If you cannot make any of the times we have allocated for presentation, contact course staff immediately.

Use the course Ed Discussion board for technical questions visible to all teams. Email instructor and TAs only for confidential or grading-related issues. Office hours are scheduled weekly.

*Build something you would want to read about and be proud of. Report what you learned, including what didn't work.*

*— Dr. Kaoutar El Maghraoui, HPML Spring 2026*
