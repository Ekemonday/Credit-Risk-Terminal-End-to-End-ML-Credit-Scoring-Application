(function () {
  "use strict";

  const form = document.getElementById("risk-form");
  const runBtn = document.getElementById("run-btn");
  const needleGroup = document.getElementById("needle-group");
  const pdValue = document.getElementById("pd-value");
  const gradeLetter = document.getElementById("grade-letter");
  const gradeLabel = document.getElementById("grade-label");
  const gradeBadge = document.getElementById("grade-badge");
  const readoutStatus = document.getElementById("readout-status");
  const factorsList = document.getElementById("factors-list");
  const readoutPanel = document.querySelector(".readout");
  const ticksGroup = document.getElementById("ticks");

  // ---- Build gauge tick marks (0% to 100%, every 10%) ----
  function buildTicks() {
    const cx = 120, cy = 130, rOuter = 100, rInner = 88;
    let svg = "";
    for (let i = 0; i <= 10; i++) {
      const angleDeg = -180 + i * 18; // -180 (0%) to 0 (100%)
      const rad = (angleDeg * Math.PI) / 180;
      const x1 = cx + rOuter * Math.cos(rad);
      const y1 = cy + rOuter * Math.sin(rad);
      const x2 = cx + rInner * Math.cos(rad);
      const y2 = cy + rInner * Math.sin(rad);
      svg += `<line x1="${x1.toFixed(2)}" y1="${y1.toFixed(2)}" x2="${x2.toFixed(2)}" y2="${y2.toFixed(2)}" stroke="#3A414D" stroke-width="2"/>`;
    }
    ticksGroup.innerHTML = svg;
  }
  buildTicks();

  // ---- Live readouts for range sliders ----
  document.querySelectorAll('input[type="range"]').forEach((slider) => {
    const readout = document.querySelector(`.range-readout[data-for="${slider.name}"]`);
    const update = () => {
      readout.textContent = Math.round(parseFloat(slider.value) * 100) + "%";
    };
    slider.addEventListener("input", update);
    update();
  });

  // ---- Needle rotation: probability 0..1 maps to -90deg .. +90deg ----
  function setNeedle(probability) {
    const angle = -90 + probability * 180;
    needleGroup.style.transform = `rotate(${angle}deg)`;
    needleGroup.setAttribute("transform", `rotate(${angle} 120 130)`);
  }

  function renderFactors(factors) {
    factorsList.innerHTML = "";
    factors.forEach((f, idx) => {
      const li = document.createElement("li");
      li.className = "factor-item pulse-in";
      li.style.animationDelay = `${idx * 60}ms`;

      const niceName = f.feature
        .replace(/_/g, " ")
        .replace(/\b\w/g, (c) => c.toUpperCase());

      li.innerHTML = `
        <div class="factor-top-row">
          <span class="factor-name">${niceName}</span>
          <span class="factor-flag ${f.flagged ? "on" : "off"}">${f.flagged ? "elevated" : "normal"}</span>
        </div>
        <div class="factor-bar-track"><div class="factor-bar-fill" data-width="${90 - idx * 14}"></div></div>
        <div class="factor-explanation">${f.explanation}</div>
      `;
      factorsList.appendChild(li);
    });

    requestAnimationFrame(() => {
      factorsList.querySelectorAll(".factor-bar-fill").forEach((bar) => {
        bar.style.width = bar.dataset.width + "%";
      });
    });
  }

  function setGradeClass(grade) {
    readoutPanel.classList.remove("risk-A", "risk-B", "risk-C", "risk-D", "risk-E");
    readoutPanel.classList.add(`risk-${grade}`);
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    runBtn.classList.add("loading");
    runBtn.querySelector(".run-btn-label").textContent = "Scoring…";

    const formData = new FormData(form);
    const payload = Object.fromEntries(formData.entries());

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.error || "Prediction failed");
      }

      const pct = (data.probability_default * 100).toFixed(1);
      setNeedle(data.probability_default);
      pdValue.textContent = pct + "%";
      gradeLetter.textContent = data.risk_grade;
      gradeLabel.textContent = data.risk_label;
      setGradeClass(data.risk_grade);
      renderFactors(data.top_factors);

      readoutStatus.classList.remove("pulse-in");
      void readoutStatus.offsetWidth;
      readoutStatus.classList.add("pulse-in");
    } catch (err) {
      pdValue.textContent = "Error";
      gradeLabel.textContent = err.message || "Something went wrong";
      factorsList.innerHTML = `<li class="factor-placeholder">${err.message || "Could not reach the scoring service."}</li>`;
    } finally {
      runBtn.classList.remove("loading");
      runBtn.querySelector(".run-btn-label").textContent = "Run assessment";
    }
  });
})();
