# How to release

A short primer in the steps needed to release a new version of the `weldx` package

## create release PR

- [ ] create a PR that finalizes the code for the next version
  - [ ] name the PR according to the version `vX.Y.Z` and add the `release`
    tag ([example here](https://github.com/BAMWelDX/weldx/pull/419))
  - [ ] make sure `CHANGELOG.md` is up-to-date and enter current date to the release version
  - [ ] add summarized release highlights where appropriate
  - [ ] update the `CITATION.cff` version number and date
  - [ ] search the project for `deprecated` and remove deprecated code
- [ ] wait for review and the CI jobs to finish
- [ ] check the readthedocs PR build

## Merge the Pull Request

- [ ] merge normally and wait for all CI actions to finish on the main branch

## add Git(hub) tag

- [ ] tag and release the current master version on GitHub using the **Releases** feature
  - [ ] name the release **git tag** according to the version released (e.g. **v0.3.3**)
  - [ ] name the GitHub release accordingly, omitting the **v** prefix (this can be change later so don't worry, in
    doubt use **vX.Y.Z** everywhere)
  - [ ] copy the changes/release notes of the current version into the description and change the GitHub PR links to GitHub markdown
- [ ] wait for all Github Actions to finish

## ReadTheDocs updates

- [ ] check the build processes for `latest`, `stable` and `vX.Y.Z` get triggered on RTD (the tag build can get
  triggered twice, resulting in a failed/duplicated build, no need to worry)

## pypi release

- [ ] check the automatic release to pypi after the `build` action completes [here](https://pypi.org/project/weldx/)

## conda-forge release

- pypi release should get picked up by the conda-forge bot and create the new
  pull-request [here](https://github.com/conda-forge/weldx-feedstock/pulls)
- [ ] carefully check the `meta.yaml` in the pull request, manually update all changes in the build and run dependencies
- [ ] merge with 2 or more approved reviews
