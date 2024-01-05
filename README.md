# Kindling

> *Kindling*: A starting point, for lighting a torch.

Cookiecutter template repository for managing semi-complex machine learning research projects (as standalone Python
packages) built with [PyTorch](https://pytorch.org/), with sane quality defaults (`black`, `ruff`, `pre-commit`).

Template created by 🔥 Sidd Karamcheti 🔥; if you find this useful, but are looking for a more opinionated
[Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) setup, definitely check out the
[`mjolnir`](https://github.com/siddk/mjolnir) template!

---

## Setup

The preferred setup is via Github Templates (Green Button above --> "Use as Template") or upon new repository creation
(borrowed with gratitude from
[Stefan Buck's instructions](https://stefanbuck.com/blog/repository-templates-meets-github-actions)). Manually edit the
`cookiecutter.json` file (in browser!), then commit, and let Github Actions take care of the rest.

*Note*: Prior to editing the `cookiecutter.json` file, navigate to the newly created repository's "Secrets" page and add
a token `REPO_SETUP_TOKEN` (following
[Github's Personal Access Token rules](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)).
This will automatically get removed once the repository is properly set up.

---

You can also load this repository via the default `cookiecutter` tool:

```bash
# Create a new directory with Cookiecutter templates (prompts you for config values)
cookiecutter gh:siddk/kindling

# If you've already initialized a github repo with same name, and want to replace contents (run from root of github repo)
cookiecutter gh:siddk/kindling -o ../ -f
```
