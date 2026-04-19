## Ulaby_Long_2014_Ch11.pdf

**Source:** Ulaby, F.T. and Long, D.G. (2014). *Microwave Radar and Radiometric Remote Sensing*, University of Michigan Press / Artech House. Chapter 11: Volume-Scattering Models and Land Observations.

**Obtained:** April 2026 via Open University Library.

**Used for:**
- Canonical S²RT/R model closed-form equations (§11-6, §11-7, eqs. 11.76, 11.77, 11.85)
- Scattering mechanism decomposition structure (Fig. 11-13, eq. 11.74–11.75)
- Secondary mechanism-ratio anchor for Set C (Table 11-1, p.484)
- Primary-source compiled description of MIMICS (§11-12.3)

**Note on coverage:** Chapter 11 does not contain finite-cylinder scattering amplitude derivations (no Bessel-J₁ radial or sinc axial form factor material). Those sit in Chapter 8 §8-5 of the same textbook (not staged) or in external sources (Ulaby & Elachi 1990, Ulaby et al. 1990 IJRS — not staged). Finite-cylinder correctness in PyTorch MIMICS is established via (a) Rayleigh-limit analytic regression from Ch 11 eqs 11.76/11.77/11.85, and (b) the G2 published_table arm against T94 Fig. 2 (primary external anchor).
