# MVA-Project

This repository uses ROOT samples for the analysis workflow.

## File Assignment

The files listed below each name are the samples that person will be working with.

### Evie

- Diboson
- SingleH
- tops
- tt_bar

### Esteban

- Vgamma
- VVV
- Wjets
- Zjets

## ROOT File Meaning

| Sample | Meaning |
| --- | --- |
| Diboson | Diboson background processes, for example WW/WZ/ZZ. |
| signal_ggF | Signal sample produced via gluon-gluon fusion (ggF). |
| signal_VBF | Signal sample produced via vector boson fusion (VBF). |
| SingleH | Single-Higgs background sample. |
| tops | Single-top and related top backgrounds. |
| tt_bar / ttbar | Top pair background sample. |
| Vgamma | Vector boson plus photon background. |
| VVV | Triboson background processes. |
| Wjets | W plus jets background. |
| Zjets | Z plus jets background. |

## Notes

- In data folders, these are ROOT files (for example diboson.root, singleH.root, ttbar.root).
- ROOT datasets are large binary inputs and are ignored by Git to keep the repository lightweight.

## Cuts Needed

### 1 Lepton 2 Taus

- n_b_jet == 0
- n_jet >= 2

### 2 Lepton 2 Taus

- n_b_jet == 0
- 21_charge * 12_charge < 0 or 21_charge != 12_charge
- mZ_cut > 0

