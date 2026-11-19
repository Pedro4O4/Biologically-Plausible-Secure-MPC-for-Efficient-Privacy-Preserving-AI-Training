# Publishing the GitHub profile (manual, ~2 minutes)

Git cannot do these steps from this machine (no GitHub credentials in WSL), so they are done in the browser.

## 1. Profile README
1. github.com/new, repository name **exactly** `Pedro4O4` (same as your username), **Public**,
   tick "Add a README file".
2. Replace its README.md with the contents of `docs/github-profile/README.md`, commit.
3. Before committing, edit freely. Things only you know and that were left out on purpose:
   LinkedIn link, e-mail, university name, any CTF handles (HackTheBox / TryHackMe), real projects you want listed.

## 2. Pin the thesis repository
Profile page -> "Customize your pins" -> select the thesis repo (plus your best web/security projects).

## 3. Repository "About" box (thesis repo -> gear icon next to About)
**Description**

    SNN with local STDP learning on MNIST, built for secure MPC training (MP-SPDZ). Plaintext stage done; MPC in progress.

**Topics** (space separated)

    spiking-neural-networks stdp neuromorphic secure-multi-party-computation mpc mp-spdz privacy-preserving-machine-learning pytorch mnist bachelor-thesis

Untick "Releases" and "Packages" in the sidebar if unused.

## 4. Push this repository's changes
    git add -A && git commit -m "Professional README, lean requirements, MNIST script, MP-SPDZ submodule metadata"
    git push

## 5. Still open (your decision)
- **License.** None is set. Choose one (e.g. MIT) or leave all rights reserved; ask your supervisor if the thesis has IP rules.
- **Dashboard screenshot.** Take one of the Phase 4 tab and save it as `docs/img/dashboard.png`, then add
  `![Dashboard](docs/img/dashboard.png)` under the "Using the dashboard" heading in README.md.
