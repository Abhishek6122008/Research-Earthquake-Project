




<p align="center">
  <img src="https://img.shields.io/badge/release-v0.0.1-blue" />
  <img src="https://img.shields.io/badge/license-MIT-green" />
  <img src="https://img.shields.io/github/issues/Abhishek6122008/Reserch-Earthquake-Project" />
  <img src="https://img.shields.io/github/stars/Abhishek6122008/Reserch-Earthquake-Project?style=social" />
  <img src="https://img.shields.io/github/forks/Abhishek6122008/Reserch-Earthquake-Project?style=social" />
  <img src="https://img.shields.io/github/watchers/Abhishek6122008/Reserch-Earthquake-Project?style=social" />
</p>

<p align="center">
  <a href="https://github.com/Abhishek6122008/Reserch-Earthquake-Project">
    <img 
      src="https://i.postimg.cc/yNFB1dPd/project-imagwe.png"
      alt="Quantum Leap"
      width="900"
      style="border-radius: 20px;"
    />
  </a>
</p>

# CNN Based P and S waves arrival detection

### Project Overview

<p><i>"The earthquake starts softly and then it hits hard."</i></p>

<p>
This project presents a complete end-to-end system for detecting
<strong>P-wave (Primary)</strong> and <strong>S-wave (Secondary)</strong> arrival times
from raw seismic signals using signal processing, baseline machine learning,
and an upgraded CNN-based deep learning model.
</p>

---

# 📌 Project Overview

<p>
Accurate identification of P and S wave arrivals is critical for:
</p>

<ul>
<li>Earthquake early warning systems</li>
<li>Epicenter localization</li>
<li>Magnitude estimation</li>
<li>Structural safety analysis</li>
<li>Disaster mitigation</li>
</ul>

<p>
Manual picking of arrivals is slow, subjective, and sensitive to noise.
This system automates the entire workflow.
</p>

---

# 🌊 Seismic Wave Fundamentals

## 🌊 P Wave (Primary Wave)

<p><strong>Definition:</strong></p>

<ul>
<li>Arrives first</li>
<li>Longitudinal (compressional)</li>
<li>Ground moves back–forth in direction of wave travel</li>
</ul>

<p><i>Think of pushing and pulling a slinky.</i></p>

<table>
<tr><th>Property</th><th>P Wave</th></tr>
<tr><td>Speed</td><td>Fastest</td></tr>
<tr><td>Medium</td><td>Solid, Liquid, Gas</td></tr>
<tr><td>Motion</td><td>Compression & Expansion</td></tr>
<tr><td>Arrival</td><td>First</td></tr>
<tr><td>Damage</td><td>Very low</td></tr>
</table>

<p>
On a seismogram: small oscillations, weak amplitude.
</p>

---

## 🌊 S Wave (Secondary Wave)

<p><strong>Definition:</strong></p>

<ul>
<li>Arrives after P wave</li>
<li>Transverse (shear)</li>
<li>Ground moves perpendicular to direction of travel</li>
</ul>

<p><i>Think of shaking a rope.</i></p>

<table>
<tr><th>Property</th><th>S Wave</th></tr>
<tr><td>Speed</td><td>Slower</td></tr>
<tr><td>Medium</td><td>Solids only</td></tr>
<tr><td>Motion</td><td>Shearing</td></tr>
<tr><td>Arrival</td><td>Second</td></tr>
<tr><td>Damage</td><td>High</td></tr>
</table>

<p>
S waves have much larger amplitude and cause structural damage.
</p>

---

# 🔥 Why P Arrives First but S is Dangerous

<p>
P waves compress → Earth handles compression well.<br>
S waves shear → Structures do not handle shear well.
</p>

<ul>
<li>P wave → Knock on the door</li>
<li>S wave → Door gets kicked in</li>
</ul>

<p><strong>One-line takeaway:</strong> P waves arrive first and are weak; S waves arrive later and are stronger and destructive.</p>

---

# 🧠 System Architecture

<pre>
Raw Seismic Signal
        ↓
Signal Conditioning
        ↓
Butterworth Bandpass Filtering
        ↓
Feature Extraction
        ↓
Baseline ML Model
        ↓
CNN Model
        ↓
P & S Arrival Detection
</pre>

---

# 🎛 Signal Processing Foundation

## Nyquist Frequency

<p>
Nyquist = Sampling Rate / 2
</p>

<p>
It defines the highest frequency that can be captured without aliasing
when converting continuous seismic signals to discrete form.
</p>

---

## Butterworth Bandpass Filtering

<p>
A Butterworth bandpass filter retains only the relevant seismic frequency band.
Filter order (N=4) controls how sharply unwanted frequencies are removed.
</p>

<ul>
<li>Low order → smooth filtering</li>
<li>High order → sharp but risky (ringing, instability)</li>
</ul>

<p>
Each order adds approximately 20 dB/decade roll-off.
</p>

<table>
<tr><th>Order</th><th>Roll-off</th></tr>
<tr><td>1</td><td>20 dB/decade</td></tr>
<tr><td>2</td><td>40 dB/decade</td></tr>
<tr><td>4</td><td>80 dB/decade</td></tr>
<tr><td>8</td><td>160 dB/decade</td></tr>
</table>

<p>
N=4 provides a balanced tradeoff between noise removal and signal preservation.
When used with zero-phase filtering (filtfilt), the effective order doubles.
</p>

---

# 🤖 Baseline Machine Learning Model

<p>
A traditional ML model was first implemented to establish a performance benchmark.
</p>

<p><strong>Features used:</strong></p>

<ul>
<li>Signal energy</li>
<li>Rolling variance</li>
<li>STA/LTA ratio</li>
<li>Frequency band energy</li>
<li>Amplitude envelope</li>
<li>Zero-crossing rate</li>
</ul>

<p>
The baseline model provided initial detection capability but struggled in low SNR and complex noise scenarios.
</p>

---

# 🧠 CNN-Based Deep Learning Model

<p>
To improve robustness, the system was upgraded to a Convolutional Neural Network (CNN).
</p>

<p><strong>Why CNN?</strong></p>

<ul>
<li>Learns waveform patterns automatically</li>
<li>Captures local temporal features</li>
<li>Handles noisy signals better</li>
<li>Reduces reliance on manual feature engineering</li>
</ul>

<p>
The CNN significantly improved detection accuracy and generalization across seismic events.
</p>

---

# 📊 Evaluation Metrics

<ul>
<li>Precision</li>
<li>Recall</li>
<li>F1 Score</li>
<li>Mean Absolute Error</li>
<li>Detection latency</li>
</ul>

<p>
High recall for P-wave detection is critical for early warning systems.
</p>

---

# 🚀 Real-World Impact

<p>
Even detecting a P wave seconds earlier can enable early warnings before the destructive S wave arrives.
</p>

<p>
This project integrates physics-based signal processing with deep learning for reliable, automated seismic phase detection.
</p>
