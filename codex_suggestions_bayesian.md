# Codex Suggestions: Bayesian Inference Chapter

## Scope and Framing
- Keep the current scope (Normal, Bernoulli/Beta, ridge regression); it is appropriate for an undergraduate ML course.
- Add a short roadmap at the chapter start: prior -> likelihood -> posterior -> posterior summaries (MAP/mean) -> prediction/interpretation.
- Add a notation table near the beginning for recurring symbols (`t_*`, `\sigma^2`, `\mu_0`, `\tau^2`, `h`, `N`, `s`).

## Presentation Improvements
- End each major subsection with a brief "What did we learn?" summary including:
  - Posterior distribution form
  - MAP estimator
  - Posterior mean
  - Effect of prior strength
- Make transitions explicit when changing assumptions (especially in the Normal section from general setup to fixed variance case).
- Keep terminology and capitalization consistent (e.g., use "ridge regression" consistently in prose).

## Topic-Level Additions (Within Current Scope)
- In the Bernoulli/Beta section, include the general conjugate update once:
  - `\mathrm{Beta}(\alpha,\beta) + h\text{ heads in }N \Rightarrow \mathrm{Beta}(\alpha+h,\beta+N-h)`
  - Then present uniform prior and `\mathrm{Beta}(5,5)` as concrete special cases.
- In the ridge section, explicitly connect Bayesian MAP to regularized optimization:
  - `\min_M \|Y-XM\|^2 + \lambda\|M\|^2`, with `\lambda = \sigma^2/\tau^2`.

## Concrete Cleanup Targets in Current Draft
- `chapters/03-1-Bayesian-Inference.md:35`
  - If described as a likelihood, use conditional notation (e.g., `P(\mathbf{t}\mid t_*,\sigma^2)`).
- `chapters/03-1-Bayesian-Inference.md:73`
  - Use consistent expectation variable (`E[t_*]` if `t_*` is the random quantity).
- `chapters/03-1-Bayesian-Inference.md:93`
  - Use `\mathbf{t}_0` on the RHS to match conditioning on observed data.
- `chapters/03-1-Bayesian-Inference.md:370`
  - Normalize wording/capitalization for ridge regression.
