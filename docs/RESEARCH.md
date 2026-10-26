# Research log: one card per paper

**Rule (course GenAI policy):** only cite a paper after someone in the group has actually read it.
The arXiv IDs below are starting points: open each one, confirm the details, then fill in its card.
Each card maps 1:1 to a "Research" sticky on the Miro board.

## Reading list (split by owner area)

| # | Paper | arXiv / link | Why it matters to us | Reader | Read? |
|---|---|---|---|---|---|
| R1 | Ito & Johnson (2017) *The LJ Speech Dataset* | keithito.com/LJ-Speech-Dataset | Our data: recording conditions, normalised column | M1 | ☐ |
| R2 | Wang et al. (2017) *Tacotron* | 1703.10135 | First seq2seq TTS; attention alignment and its failure modes | M2 | ☐ |
| R3 | Shen et al. (2018) *Tacotron 2* | 1712.05884 | Classic baseline: mel + vocoder; skipping/repeat failure modes | M2 | ☐ |
| R4 | van den Oord et al. (2016) *WaveNet* | 1609.03499 | Original neural vocoder; why it is slow (RTF) | M3 | ☐ |
| R5 | Prenger et al. (2019) *WaveGlow* | 1811.00002 | Flow vocoder; trade-off vs HiFi-GAN | M3 | ☐ |
| R6 | Kong et al. (2020) *HiFi-GAN* | 2010.05646 | Fast high-quality vocoder, LJSpeech results | M2 | ☐ |
| R7 | Ren et al. (2021) *FastSpeech 2* | 2006.04558 | Non-autoregressive, duration/pitch/energy; speed and robustness | M2 | ☐ |
| R8 | Kim et al. (2020) *Glow-TTS* | 2005.11129 | Monotonic alignment search, which fixes attention failures | M2 | ☐ |
| R9 | Kim et al. (2021) *VITS* | 2106.06103 | End-to-end, strong LJSpeech MOS; candidate method | M2 | ☐ |
| R10 | Popov et al. (2021) *Grad-TTS* | 2105.06337 | Diffusion TTS: quality vs number of steps (a natural controlled variable) | M3 | ☐ |
| R11 | Mehta et al. (2024) *Matcha-TTS* | 2309.03199 | Flow-matching, fast and small, LJSpeech; candidate method | M3 | ☐ |
| R12 | Li et al. (2023) *StyleTTS 2* | 2306.07691 | Recent state of the art on LJSpeech; for related work/future work | M1 | ☐ |
| R13 | Radford et al. (2022) *Whisper* | 2212.04356 | ASR used for our WER/CER intelligibility metric | M3 | ☐ |
| R14 | Saeki et al. (2022) *UTMOS* | 2204.02152 | Automatic MOS prediction (naturalness without a listening panel) | M3 | ☐ |
| R15 | Kubichek (1993) *Mel-cepstral distance* | IEEE PACRIM | MCD metric definition | M3 | ☐ |
| R16 | Ravanelli et al. (2021) *SpeechBrain* | 2106.04624 | Toolkit option (pretrained Tacotron2/HiFi-GAN on LJSpeech) | M1 | ☐ |
| R17 | Coqui TTS (idiap fork `coqui-tts`) | github.com/idiap/coqui-ai-TTS | Toolkit option (VITS/Tacotron2/Glow-TTS recipes for LJSpeech) | M1 | ☐ |

---

## Card template (copy one per paper)

### R? — <Title> (<Authors>, <Year>)
- **Link:**
- **Problem it solves:**
- **Key idea (2 lines):**
- **Architecture stage:** front-end / acoustic / vocoder / end-to-end / evaluation
- **Reported results (LJSpeech if available):** MOS __, RTF __, hardware __
- **Strengths for our constraints (laptop, 1 week, 1–3k utts):**
- **Weaknesses / failure modes it reports:**
- **What we take from it:** method choice / experiment idea / metric / error-analysis argument
- **Report section(s) it supports:** 2 Background · 3 System design · 4 Experiments · 6 Error analysis
- **Harvard reference (checked):**
