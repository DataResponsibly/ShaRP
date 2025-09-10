# Changelog

## [Unreleased](https://github.com/DataResponsibly/ShaRP/tree/HEAD)

[Full Changelog](https://github.com/DataResponsibly/ShaRP/compare/v0.1.1...HEAD)

**Fixed bugs:**

- Top-k QoI - can't change k [\#68](https://github.com/DataResponsibly/ShaRP/issues/68)

## [v0.1.1](https://github.com/DataResponsibly/ShaRP/tree/v0.1.1) (2025-09-10)

[Full Changelog](https://github.com/DataResponsibly/ShaRP/compare/v0.1.0...v0.1.1)

**Implemented features and enhancements:**

- FEAT: Add optional feature\_names argument to ShaRP.fit\(\) [\#62](https://github.com/DataResponsibly/ShaRP/issues/62)
- Add Hash table to lookup inference results [\#58](https://github.com/DataResponsibly/ShaRP/issues/58)

**Fixed bugs:**

- DOCS: Broken link to issues in bug report issue template [\#65](https://github.com/DataResponsibly/ShaRP/issues/65)
- Fix bugs raised by tests [\#59](https://github.com/DataResponsibly/ShaRP/issues/59)

**Documentation updates:**

- DOCS: Broken link to Contributing Guidelines in issue template [\#64](https://github.com/DataResponsibly/ShaRP/issues/64)
- Update README with correct installation instructions [\#57](https://github.com/DataResponsibly/ShaRP/issues/57)

## [v0.1.0](https://github.com/DataResponsibly/ShaRP/tree/v0.1.0) (2025-03-26)

[Full Changelog](https://github.com/DataResponsibly/ShaRP/compare/v0.1.a1...v0.1.0)

**Implemented features and enhancements:**

- Implement ShaRP wrapper for document retrieval [\#56](https://github.com/DataResponsibly/ShaRP/issues/56)
- Get test coverage to \>75% [\#49](https://github.com/DataResponsibly/ShaRP/issues/49)
- Raise warning when qoi is meant for labels \(in classification\) and there are a lot of unique values in the target [\#33](https://github.com/DataResponsibly/ShaRP/issues/33)

**Fixed bugs:**

- Defining `ShaRP` without an explicit `QoI` raises an `AttributeError` [\#44](https://github.com/DataResponsibly/ShaRP/issues/44)

**Documentation updates:**

- Improve content in documentation pages [\#13](https://github.com/DataResponsibly/ShaRP/issues/13)
- Improve docstrings [\#3](https://github.com/DataResponsibly/ShaRP/issues/3)

**Closed issues:**

- Add python 3.13 support [\#60](https://github.com/DataResponsibly/ShaRP/issues/60)

## [v0.1.a1](https://github.com/DataResponsibly/ShaRP/tree/v0.1.a1) (2024-11-14)

[Full Changelog](https://github.com/DataResponsibly/ShaRP/compare/v0.1.a0...v0.1.a1)

**Implemented features and enhancements:**

- Add additional grouping methods \(other than strata\) for boxplot visualizations [\#39](https://github.com/DataResponsibly/ShaRP/issues/39)

## [v0.1.a0](https://github.com/DataResponsibly/ShaRP/tree/v0.1.a0) (2024-11-12)

[Full Changelog](https://github.com/DataResponsibly/ShaRP/compare/14353cb9e444a5a2b1112adbd629028914a2c9c5...v0.1.a0)

**Implemented features and enhancements:**

- Add additional grouping methods \(other than strata\) for boxplot visualizations [\#38](https://github.com/DataResponsibly/ShaRP/issues/38)
- Add visualization for aggregate contributions per stratum [\#30](https://github.com/DataResponsibly/ShaRP/issues/30)
- Add `sharp.show_versions()` function [\#28](https://github.com/DataResponsibly/ShaRP/issues/28)
- Add `get_qoi` and `get_qoi_names` functions [\#27](https://github.com/DataResponsibly/ShaRP/issues/27)
- Add possibility for parallelizing calculations and introduce `n_jobs` parameter [\#16](https://github.com/DataResponsibly/ShaRP/issues/16)
- Sample without replacement [\#9](https://github.com/DataResponsibly/ShaRP/issues/9)
- Implement visualizations from paper [\#8](https://github.com/DataResponsibly/ShaRP/issues/8)

**Fixed bugs:**

- Waterfall plots displaying undesired information [\#32](https://github.com/DataResponsibly/ShaRP/issues/32)
- Shapley values ranges are not in the correct range [\#18](https://github.com/DataResponsibly/ShaRP/issues/18)
- Flip QoI does not work when passing parameter `qoi` as string [\#17](https://github.com/DataResponsibly/ShaRP/issues/17)
- `rank` method from ranking QoIs not working as expected [\#11](https://github.com/DataResponsibly/ShaRP/issues/11)
- Check Rank-based QoI's [\#4](https://github.com/DataResponsibly/ShaRP/issues/4)

**Documentation updates:**

- broken documentation website [\#42](https://github.com/DataResponsibly/ShaRP/issues/42)
- Documentation deployment [\#23](https://github.com/DataResponsibly/ShaRP/issues/23)
- Add example for classification [\#22](https://github.com/DataResponsibly/ShaRP/issues/22)
- Add CSRankings example [\#21](https://github.com/DataResponsibly/ShaRP/issues/21)
- Set up a documentation page [\#5](https://github.com/DataResponsibly/ShaRP/issues/5)
- Add documentation for QOI classes [\#2](https://github.com/DataResponsibly/ShaRP/issues/2)

**Closed issues:**

- Rename QoI's with `ranking` in the name to `rank` [\#25](https://github.com/DataResponsibly/ShaRP/issues/25)
- Rename `scores_to_rank` to `scores_to_ordering` [\#24](https://github.com/DataResponsibly/ShaRP/issues/24)
- Add docstring tests [\#1](https://github.com/DataResponsibly/ShaRP/issues/1)



\* *This Changelog was automatically generated by [github_changelog_generator](https://github.com/github-changelog-generator/github-changelog-generator)*
