/* Power dashboard — frontend.
 *
 * Every chart is fed by one endpoint in app.py; nothing is computed here beyond
 * pivoting rows into Chart.js datasets. Tabs load lazily and cache their data.
 *
 * Colour: the categorical slots below are assigned in fixed order and never
 * cycled — a 9th series folds into "Khác". Years use a one-hue ordinal ramp.
 * Both sets were checked with the data-viz validator against this page's own
 * surfaces (#ffffff / #17171b): all gates pass. Three light-mode slots sit under
 * 3:1 on white, which obliges the relief rule — hence the legend on every
 * multi-series chart and the "Bảng" table view on every chart.
 */

// ---------------------------------------------------------------- palette
const CAT = {
  light: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
  dark:  ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
};
// 5 steps is the most a one-hue blue ramp fits while keeping every adjacent
// ΔL ≥ 0.06, so year overlays show five years at a time.
const YEAR_RAMP = {
  light: ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"],
  dark:  ["#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
};
const MAX_YEARS = YEAR_RAMP.light.length;
const MAX_SERIES = CAT.light.length;
const OTHER = "Khác";
// ENSO phases are a polarity, not an identity — warm/cool poles with a neutral
// middle, exactly as README_mongodb.md prescribes.
const PHASE_COLORS = {
  light: { el_nino: "#e34948", la_nina: "#2a78d6", neutral: "#898781" },
  dark:  { el_nino: "#e66767", la_nina: "#3987e5", neutral: "#898781" },
};
const INK = {
  light: { text: "#0b0b0b", muted: "#898781", grid: "#e1e0d9", axis: "#c3c2b7", surface: "#ffffff" },
  dark:  { text: "#ffffff", muted: "#898781", grid: "#2c2c2a", axis: "#383835", surface: "#17171b" },
};

const darkMQ = window.matchMedia("(prefers-color-scheme: dark)");
const mode = () => (darkMQ.matches ? "dark" : "light");
const cat = (i) => CAT[mode()][i % MAX_SERIES];
const ink = () => INK[mode()];

// ---------------------------------------------------------------- utils
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const nf = new Intl.NumberFormat("vi-VN", { maximumFractionDigits: 1 });
const nf0 = new Intl.NumberFormat("vi-VN", { maximumFractionDigits: 0 });
const fmt = (v, d) => (v === null || v === undefined || Number.isNaN(v) ? "—"
  : (d === 0 ? nf0 : nf).format(v));

async function api(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path} → ${res.status}`);
  return res.json();
}

function fail(host, err) {
  host.innerHTML = `<p class="err">Không tải được dữ liệu: ${esc(err.message || err)}</p>`;
  console.error(err);
}

/** Keep the 8 biggest series and fold the rest into "Khác" (never cycle hues). */
function foldSeries(names, valueOf, limit = MAX_SERIES) {
  if (names.length <= limit) return { keep: names, folded: [] };
  const ranked = [...names].sort((a, b) => valueOf(b) - valueOf(a));
  const keep = ranked.slice(0, limit - 1);
  const folded = ranked.slice(limit - 1);
  // Preserve the caller's original order among the survivors.
  return { keep: names.filter((n) => keep.includes(n)).concat([OTHER]), folded };
}

/** Sum the folded series into a single "Khác" column on every row. */
function applyFold(rows, keep, folded) {
  if (!folded.length) return rows;
  return rows.map((r) => {
    const values = {};
    for (const k of keep) {
      if (k === OTHER) {
        const total = folded.reduce((s, f) => s + (r.values[f] || 0), 0);
        values[OTHER] = (r.values[OTHER] || 0) + total;
      } else values[k] = r.values[k];
    }
    return { ...r, values };
  });
}

// ---------------------------------------------------------------- charts
const CHARTS = new Map();   // canvas id -> Chart
const TABLES = new Map();   // canvas id -> {columns, rows}

Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
Chart.defaults.font.size = 11.5;
Chart.defaults.animation.duration = 260;

function baseOptions({ stacked = false, unit = "", horizontal = false, legend = true, decimals } = {}) {
  const c = ink();
  return {
    responsive: true,
    maintainAspectRatio: false,
    indexAxis: horizontal ? "y" : "x",
    interaction: { mode: "index", intersect: false },
    layout: { padding: { top: 4 } },
    plugins: {
      legend: {
        display: legend,
        position: "bottom",
        labels: { color: c.muted, boxWidth: 9, boxHeight: 9, usePointStyle: true,
                  pointStyle: "rectRounded", padding: 12 },
      },
      tooltip: {
        backgroundColor: mode() === "dark" ? "#26262c" : "#1a1a1f",
        titleColor: "#fff", bodyColor: "#fff", padding: 9, cornerRadius: 8,
        boxWidth: 9, boxHeight: 9, usePointStyle: true,
        callbacks: {
          label: (ctx) => {
            const v = ctx.parsed[horizontal ? "x" : "y"];
            return ` ${ctx.dataset.label}: ${fmt(v, decimals)}${unit ? " " + unit : ""}`;
          },
        },
      },
    },
    scales: {
      x: {
        stacked,
        grid: { display: false, drawBorder: false },
        border: { color: c.axis },
        ticks: { color: c.muted, autoSkip: true, maxRotation: 0, maxTicksLimit: 14 },
      },
      y: {
        stacked,
        grid: { color: c.grid, drawBorder: false, lineWidth: 1 },
        border: { display: false },
        ticks: { color: c.muted, maxTicksLimit: 6,
                 callback: (v) => fmt(v, Math.abs(v) >= 1000 ? 0 : undefined) },
        title: unit ? { display: true, text: unit, color: c.muted } : undefined,
      },
    },
  };
}

/** Bars: thin, 4px rounded data-end, a 2px surface gap between stacked fills. */
function barSet(label, data, color, { stacked = false } = {}) {
  return {
    label, data, backgroundColor: color,
    borderColor: ink().surface, borderWidth: stacked ? 2 : 0,
    borderRadius: 4, borderSkipped: "start",
    maxBarThickness: 34, categoryPercentage: 0.82, barPercentage: 0.92,
  };
}

function lineSet(label, data, color, { fill = false, dashed = false } = {}) {
  return {
    label, data, borderColor: color, backgroundColor: color,
    borderWidth: 2, borderDash: dashed ? [5, 4] : undefined,
    pointRadius: 0, pointHoverRadius: 4, pointHoverBorderWidth: 2,
    pointHoverBorderColor: ink().surface, tension: 0.25, fill,
    spanGaps: true,
  };
}

/** Draw (or redraw) a chart and register its table view. */
function paint(id, config, table) {
  const canvas = document.getElementById(id);
  if (!canvas) return;
  CHARTS.get(id)?.destroy();
  CHARTS.set(id, new Chart(canvas, config));
  if (table) {
    TABLES.set(id, table);
    ensureTableToggle(id, canvas);
    // Keep an already-open table in step when the chart is redrawn (freq switch).
    const box = canvas.closest(".chartbox").nextElementSibling;
    if (box?.classList.contains("tablebox") && box.style.display !== "none") {
      box.innerHTML = tableHTML(table);
    }
  }
}

function ensureTableToggle(id, canvas) {
  const card = canvas.closest(".card");
  if (!card || card.querySelector(`[data-table="${id}"]`)) return;
  const head = card.querySelector(".head");
  let ctrl = head.querySelector(".ctrl");
  if (!ctrl) { ctrl = document.createElement("span"); ctrl.className = "ctrl"; head.appendChild(ctrl); }
  const btn = document.createElement("button");
  btn.className = "preset";
  btn.dataset.table = id;
  btn.textContent = "Bảng";
  ctrl.appendChild(btn);
  const box = document.createElement("div");
  box.className = "tablebox";
  box.style.display = "none";
  canvas.closest(".chartbox").after(box);
  btn.addEventListener("click", () => {
    const open = box.style.display === "none";
    btn.classList.toggle("active", open);
    box.style.display = open ? "block" : "none";
    if (open) box.innerHTML = tableHTML(TABLES.get(id));
  });
}

function tableHTML({ columns, rows }) {
  // A column is numeric only if its values are numbers — plant tables mix text
  // columns (loại nguồn, chủ đầu tư) in with the capacities.
  const numeric = columns.map((_, i) =>
    i > 0 && rows.some((r) => typeof r[i] === "number") &&
    rows.every((r) => r[i] === null || r[i] === undefined || typeof r[i] === "number"));
  const head = columns.map((c, i) => `<th class="${numeric[i] ? "num" : ""}">${esc(c)}</th>`).join("");
  const body = rows.map((r) =>
    `<tr>${r.map((v, i) => `<td class="${numeric[i] ? "num" : ""}">${numeric[i] ? fmt(v) : esc(v ?? "—")}</td>`).join("")}</tr>`
  ).join("");
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

/** `[{period, values{}}]` + metric names -> a table view of the same numbers. */
function tableFrom(rows, metrics, firstCol = "Kỳ") {
  return {
    columns: [firstCol, ...metrics],
    rows: rows.map((r) => [r.period ?? r.date, ...metrics.map((m) => r.values?.[m])]),
  };
}

// ---------------------------------------------------------------- controls
const FREQ_LABELS = { day: "Ngày", month: "Tháng", quarter: "Quý", year: "Năm" };

function segmented(host, options, current, onPick) {
  host.innerHTML = "";
  const seg = document.createElement("span");
  seg.className = "seg";
  for (const opt of options) {
    const b = document.createElement("button");
    b.textContent = FREQ_LABELS[opt] || opt;
    b.classList.toggle("active", opt === current);
    b.addEventListener("click", () => {
      $$("button", seg).forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      onPick(opt);
    });
    seg.appendChild(b);
  }
  host.appendChild(seg);
}

/** Split `[{period,date,value}]` into one dataset per year for a year overlay. */
function yearOverlay(rows, freq, valueOf = (r) => r.value) {
  const byYear = new Map();
  for (const r of rows) {
    const year = String(r.period).slice(0, 4);
    const x = freq === "year" ? year
      : freq === "quarter" ? String(r.period).split("-")[1]
      : freq === "month" ? String(r.period).slice(5, 7)
      : String(r.period).slice(5);
    if (!byYear.has(year)) byYear.set(year, new Map());
    byYear.get(year).set(x, valueOf(r));
  }
  const years = [...byYear.keys()].sort();
  const labels = [...new Set(rows.map((r) =>
    freq === "year" ? String(r.period).slice(0, 4)
      : freq === "quarter" ? String(r.period).split("-")[1]
      : freq === "month" ? String(r.period).slice(5, 7)
      : String(r.period).slice(5)))].sort();
  return { years, labels, byYear };
}

// ================================================================ Tab 1 — Chính phủ
const gov = {
  loaded: false,
  async load() {
    await Promise.all([this.capacity(), this.capex(), this.framework(), this.plants()]);
    this.loaded = true;
  },

  async capacity() {
    const d = await api("/api/gov/pdp8-capacity");
    const rows = d.rows.filter((r) => r.type !== d.total_type);
    const { keep, folded } = foldSeries(rows.map((r) => r.type),
      (t) => Math.max(...d.scenarios.map((s) => rows.find((r) => r.type === t).values[s] || 0)));
    const valueOf = (type, scenario) => {
      if (type !== OTHER) return rows.find((r) => r.type === type)?.values[scenario] ?? null;
      return folded.reduce((s, f) => s + (rows.find((r) => r.type === f)?.values[scenario] || 0), 0);
    };
    const datasets = keep.map((type, i) =>
      barSet(type, d.scenarios.map((s) => valueOf(type, s)), cat(i), { stacked: true }));
    const total = d.rows.find((r) => r.type === d.total_type);
    paint("c-pdp8-capacity", {
      type: "bar",
      data: { labels: d.scenarios, datasets },
      options: baseOptions({ stacked: true, unit: d.unit, decimals: 0 }),
    }, {
      columns: ["Loại nguồn", ...d.scenarios],
      rows: d.rows.map((r) => [r.type, ...d.scenarios.map((s) => r.values[s])]),
    });
    if (total) {
      $("#c-pdp8-capacity").closest(".card").querySelector(".note").innerHTML =
        `Tổng công suất: ` + d.scenarios.map((s) => `<b>${s}</b> ${fmt(total.values[s], 0)} MW`).join(" · ")
        + (folded.length ? ` · ${folded.length} loại nhỏ gộp vào “${OTHER}”.` : "");
    }
  },

  async capex() {
    const d = await api("/api/gov/pdp8-capex");
    const labels = d.rows.map((r) => `${r.phase}\n${r.investment_type === "Power Grids" ? "Lưới điện" : "Nguồn điện"}`);
    // Split phases stack State + Socialized; the 2021-2025 rows only carry a total,
    // so they get their own series instead of a fake split.
    const state = d.rows.map((r) => (d.split[r.phase] ? r.state : null));
    const social = d.rows.map((r) => (d.split[r.phase] ? r.socialized : null));
    const total = d.rows.map((r) => (d.split[r.phase] ? null : r.total));
    paint("c-pdp8-capex", {
      type: "bar",
      data: {
        labels,
        datasets: [
          barSet("Vốn nhà nước", state, cat(0), { stacked: true }),
          barSet("Vốn xã hội hoá", social, cat(1), { stacked: true }),
          barSet("Tổng vốn (chưa tách)", total, cat(6), { stacked: true }),
        ],
      },
      options: baseOptions({ stacked: true, unit: d.unit }),
    }, {
      columns: ["Giai đoạn", "Hạng mục", "Tổng", "Nhà nước", "Xã hội hoá"],
      rows: d.rows.map((r) => [`${r.phase} — ${r.investment_type}`, r.investment_type, r.total, r.state, r.socialized]),
    });
  },

  async framework() {
    const d = await api("/api/gov/price-framework");
    const sel = $("#pf-region");
    sel.innerHTML = d.regions.map((r) => `<option value="${esc(r)}">${esc(r)}</option>`).join("")
      + `<option value="">Tất cả vùng (${d.rows.length} dòng)</option>`;
    // All 36 rows at once is past the point where a bar chart reads; default to
    // the region with the most sources (~12) and keep the full set in the table.
    // "National" holds a single row, which would be a one-bar chart.
    const count = (r) => d.rows.filter((x) => x.region === r).length;
    sel.value = [...d.regions].sort((a, b) => count(b) - count(a))[0] || "";
    const draw = () => {
      const pick = sel.value;
      const rows = pick ? d.rows.filter((r) => r.region === pick) : d.rows;
      const labels = rows.map((r) => (pick ? r.source : `${r.source} · ${r.region}`));
      $("#c-price-framework").closest(".chartbox").style.height =
        Math.max(240, labels.length * 26 + 50) + "px";
      paint("c-price-framework", {
        type: "bar",
        data: { labels, datasets: [barSet("Giá trần", rows.map((r) => r.vnd), cat(0))] },
        options: {
          ...baseOptions({ unit: d.unit, horizontal: true, legend: false, decimals: 0 }),
          scales: {
            x: { grid: { color: ink().grid }, border: { display: false },
                 ticks: { color: ink().muted, callback: (v) => fmt(v, 0) } },
            y: { grid: { display: false }, border: { color: ink().axis },
                 ticks: { color: ink().muted, autoSkip: false, font: { size: 10.5 } } },
          },
        },
      });
      $("#t-price-framework").innerHTML = tableHTML({
        columns: ["Nguồn điện", "VND/kWh", "US cent/kWh", "Vùng"],
        rows: d.rows.map((r) => [r.source, r.vnd, r.cent, r.region]),
      });
    };
    sel.onchange = draw;
    draw();
  },

  async plants() {
    const d = await api("/api/gov/pdp8-plants");
    const groupSel = $("#plant-group");
    const filterSel = $("#plant-filter");
    const draw = () => {
      const by = groupSel.value;
      const buckets = by === "type" ? d.by_type : d.by_year;
      const key = by === "type" ? "type" : "year";
      const { keep, folded } = foldSeries(buckets.map((b) => b[key]),
        (n) => buckets.find((b) => b[key] === n).capacity);
      const value = (n) => n === OTHER
        ? folded.reduce((s, f) => s + buckets.find((b) => b[key] === f).capacity, 0)
        : buckets.find((b) => b[key] === n).capacity;
      paint("c-pdp8-plants", {
        type: "bar",
        data: { labels: keep, datasets: [barSet("Công suất", keep.map(value), cat(0))] },
        options: { ...baseOptions({ unit: d.unit, legend: false, decimals: 0 }),
                   scales: { ...baseOptions({ unit: d.unit }).scales,
                             x: { grid: { display: false }, border: { color: ink().axis },
                                  ticks: { color: ink().muted, autoSkip: false, maxRotation: 45, minRotation: 30,
                                           font: { size: 10 } } } } },
      }, {
        columns: [by === "type" ? "Loại nguồn" : "Năm vận hành", "Công suất (MW)", "Số dự án"],
        rows: buckets.map((b) => [b[key], b.capacity, b.count]),
      });

      const pick = filterSel.value;
      const rows = pick ? d.rows.filter((r) => (by === "type" ? r.type : r.operation).includes(pick)) : d.rows;
      $("#t-pdp8-plants").innerHTML = tableHTML({
        columns: ["Dự án", "MW", "Loại nguồn", "Vận hành", "Tiến độ", "Chủ đầu tư"],
        rows: rows.slice(0, 500).map((r) => [r.project, r.capacity, r.type, r.operation, r.progress, r.investor]),
      });
    };
    const fillFilter = () => {
      const by = groupSel.value;
      const names = (by === "type" ? d.by_type.map((b) => b.type) : d.by_year.map((b) => b.year));
      filterSel.innerHTML = `<option value="">Tất cả (${d.rows.length} dự án)</option>` +
        names.map((n) => `<option value="${esc(n)}">${esc(n)}</option>`).join("");
    };
    groupSel.onchange = () => { fillFilter(); draw(); };
    filterSel.onchange = draw;
    fillFilter();
    draw();
  },
};

// ================================================================ Tab 2 — Thời tiết
const weather = {
  loaded: false, tempFreq: "month", tempData: null,
  async load() {
    await Promise.all([this.enso(), this.oni(), this.temperature()]);
    segmented($("#temp-freq"), ["month", "quarter", "year"], this.tempFreq, (f) => {
      this.tempFreq = f; this.temperature();
    });
    this.loaded = true;
  },

  async enso() {
    const d = await api("/api/weather/enso-probability");
    const labels = d.rows.map((r) => r.date.slice(0, 7));
    const LABELS = { "La Nina prob": "La Niña", "Neutral prob": "Trung tính", "El Nino prob": "El Niño" };
    // Polarity, not identity: cool pole / neutral / warm pole.
    const colors = [PHASE_COLORS[mode()].la_nina, PHASE_COLORS[mode()].neutral, PHASE_COLORS[mode()].el_nino];
    paint("c-enso-prob", {
      type: "bar",
      data: {
        labels,
        datasets: d.metrics.map((m, i) =>
          barSet(LABELS[m] || m, d.rows.map((r) => r.values[m]), colors[i], { stacked: true })),
      },
      options: baseOptions({ stacked: true, unit: d.unit, decimals: 0 }),
    }, tableFrom(d.rows.map((r) => ({ period: r.date.slice(0, 7), values: r.values })), d.metrics, "Tháng công bố"));
  },

  async oni() {
    const d = await api("/api/weather/enso-condition");
    const sel = $("#enso-years");
    const years = [...new Set(d.rows.map((r) => r.date.slice(0, 4)))].sort().reverse();
    sel.innerHTML = `<option value="0">Toàn bộ lịch sử</option>` +
      [5, 10, 15].map((n) => `<option value="${n}">${n} năm gần nhất</option>`).join("");
    sel.value = "10";
    const draw = () => {
      const n = +sel.value;
      const keep = n ? new Set(years.slice(0, n)) : null;
      const rows = keep ? d.rows.filter((r) => keep.has(r.date.slice(0, 4))) : d.rows;
      const colors = PHASE_COLORS[mode()];
      paint("c-enso-cond", {
        type: "bar",
        data: {
          labels: rows.map((r) => r.date.slice(0, 7)),
          datasets: [{
            label: "ONI", data: rows.map((r) => r.oni),
            backgroundColor: rows.map((r) => colors[r.phase]),
            borderRadius: 3, borderSkipped: false, maxBarThickness: 14,
          }],
        },
        options: baseOptions({ unit: "ONI", legend: false }),
      }, {
        columns: ["Tháng", "ONI"],
        rows: rows.map((r) => [r.date.slice(0, 7), r.oni]),
      });
    };
    sel.onchange = draw;
    draw();
  },

  async temperature() {
    const d = this.tempData && this.tempData.freq === this.tempFreq
      ? this.tempData : (this.tempData = await api(`/api/weather/temperature?freq=${this.tempFreq}`));
    const host = $("#temp-cards");
    if (!$$(".card", host).length) {
      host.innerHTML = d.cities.map((c, i) => `
        <div class="card">
          <div class="head"><h3>${esc(c)}</h3></div>
          <div class="chartbox short"><canvas id="c-temp-${i}"></canvas></div>
        </div>`).join("");
    }
    d.cities.forEach((city, i) => {
      const rows = d.series[city] || [];
      const { labels, byYear } = yearOverlay(rows, this.tempFreq);
      const years = [...byYear.keys()].sort().slice(-MAX_YEARS);
      const ramp = YEAR_RAMP[mode()];
      paint(`c-temp-${i}`, {
        type: "line",
        data: {
          labels,
          datasets: years.map((y, k) =>
            lineSet(y, labels.map((x) => byYear.get(y).get(x) ?? null), ramp[k % ramp.length])),
        },
        options: baseOptions({ unit: d.unit }),
      }, {
        columns: ["Kỳ", ...years],
        rows: labels.map((x) => [x, ...years.map((y) => byYear.get(y).get(x))]),
      });
    });
    $("#temp-cards").closest("section").querySelectorAll(".card .head h3").forEach(() => {});
  },
};

// ================================================================ Tab 3 — Chi phí
const cost = {
  loaded: false, resFreq: "month",
  async load() {
    await Promise.all([this.coal(), this.gas(), this.reservoir()]);
    segmented($("#res-freq"), ["month", "quarter", "year"], this.resFreq, (f) => {
      this.resFreq = f; this.reservoir();
    });
    this.loaded = true;
  },

  async coal() {
    const d = await api("/api/cost/coal");
    const LABEL = { "Cost for Vinh Tan": "Vĩnh Tân", "Cost for Mong Duong": "Mông Dương" };
    paint("c-coal", {
      type: "line",
      data: {
        labels: d.rows.map((r) => r.date.slice(0, 7)),
        datasets: d.metrics.map((m, i) => lineSet(LABEL[m] || m, d.rows.map((r) => r.values[m]), cat(i))),
      },
      options: baseOptions({ unit: d.unit, decimals: 0 }),
    }, tableFrom(d.rows.map((r) => ({ period: r.date.slice(0, 7), values: r.values })), d.metrics, "Tháng"));
  },

  async gas() {
    const d = await api("/api/cost/gas");
    paint("c-gas", {
      type: "line",
      data: {
        labels: d.rows.map((r) => r.date.slice(0, 7)),
        datasets: d.metrics.map((m, i) => lineSet(m, d.rows.map((r) => r.values[m]), cat(i))),
      },
      options: baseOptions({ unit: d.unit }),
    }, tableFrom(d.rows.map((r) => ({ period: r.date.slice(0, 7), values: r.values })), d.metrics, "Tháng"));
  },

  async reservoir() {
    const d = await api(`/api/cost/reservoir-yoy?freq=${this.resFreq}`);
    paint("c-reservoir", {
      type: "bar",
      data: {
        labels: d.rows.map((r) => r.period),
        datasets: d.metrics.map((m, i) => barSet(m, d.rows.map((r) => r.values[m]), cat(i))),
      },
      options: baseOptions({ unit: d.unit }),
    }, tableFrom(d.rows, d.metrics));
    $("#res-note").textContent = d.note || "";
  },
};

// ================================================================ Tab 4 — Giá
const price = {
  loaded: false, freq: "month",
  async load() {
    await Promise.all([this.cgm(), this.can(), this.retail()]);
    segmented($("#cgm-freq"), ["day", "month", "quarter", "year"], this.freq, (f) => {
      this.freq = f; this.cgm();
    });
    this.loaded = true;
  },

  async cgm() {
    const d = await api(`/api/price/cgm?freq=${this.freq}`);
    const host = $("#cgm-cards");
    if (!$$(".card", host).length) {
      host.innerHTML = d.regions.map((r, i) => `
        <div class="card">
          <div class="head"><h3>CGM — ${esc(r.label)}</h3></div>
          <div class="chartbox"><canvas id="c-cgm-${i}"></canvas></div>
        </div>`).join("");
    }
    const ramp = YEAR_RAMP[mode()];
    d.regions.forEach((region, i) => {
      const rows = d.cgm.map((b) => ({ period: b.period, value: b.values[region.key] ?? null }));
      const { labels, byYear } = yearOverlay(rows, this.freq);
      const years = [...byYear.keys()].sort().slice(-MAX_YEARS);
      paint(`c-cgm-${i}`, {
        type: "line",
        data: {
          labels,
          datasets: years.map((y, k) =>
            lineSet(y, labels.map((x) => byYear.get(y).get(x) ?? null), ramp[k % ramp.length])),
        },
        options: baseOptions({ unit: d.unit, decimals: 0 }),
      }, {
        columns: ["Kỳ", ...years],
        rows: labels.map((x) => [x, ...years.map((y) => byYear.get(y).get(x))]),
      });
    });
  },

  async can() {
    const d = await api("/api/price/can");
    $("#can-note").textContent = d.note || "";
    paint("c-can", {
      type: "bar",
      data: {
        labels: d.rows.map((r) => r.period),
        datasets: [barSet("Giá CAN", d.rows.map((r) => r.value), cat(0))],
      },
      options: baseOptions({ unit: d.unit, legend: false, decimals: 0 }),
    }, { columns: ["Năm", "VND/kWh"], rows: d.rows.map((r) => [r.period, r.value]) });
  },

  async retail() {
    const d = await api("/api/price/retail");
    paint("c-retail", {
      type: "line",
      data: {
        labels: d.rows.map((r) => r.date),
        datasets: [{ ...lineSet("Giá bán lẻ bình quân", d.rows.map((r) => r.value), cat(0)), stepped: "after" }],
      },
      options: baseOptions({ unit: d.unit, legend: false, decimals: 0 }),
    }, { columns: ["Ngày áp dụng", "VND/kWh"], rows: d.rows.map((r) => [r.date, r.value]) });
  },
};

// ================================================================ Tab 5 — Sản lượng
const volume = {
  loaded: false, freq: "month",
  async load() {
    await this.draw();
    segmented($("#vol-freq"), ["day", "month", "quarter", "year"], this.freq, (f) => {
      this.freq = f; this.draw();
    });
    this.loaded = true;
  },

  async draw() {
    await Promise.all([this.total(), this.breakdown(), this.mismatch(), this.installed(), this.region()]);
  },

  async total() {
    const d = await api(`/api/volume/total?freq=${this.freq}`);
    const partial = d.rows.filter((r) => this.expected(r.period) && r.n < this.expected(r.period));
    paint("c-vol-total", {
      type: "bar",
      data: { labels: d.rows.map((r) => r.period),
              datasets: [barSet("Sản lượng thương phẩm", d.rows.map((r) => r.value), cat(0))] },
      options: baseOptions({ unit: d.unit, legend: false, decimals: 0 }),
    }, { columns: ["Kỳ", d.unit, "Số ngày có dữ liệu"], rows: d.rows.map((r) => [r.period, r.value, r.n]) });
    $("#vol-total-note").innerHTML = partial.length
      ? `⚠ ${partial.length} kỳ thiếu ngày (nguồn scrape có khoảng trống) nên tổng bị thấp hơn thực tế — xem cột “số ngày” trong bảng.`
      : "";
  },

  /** Days a period should contain, so a partly-scraped bucket can be flagged. */
  expected(period) {
    if (this.freq === "day") return 1;
    if (this.freq === "month") {
      const [y, m] = period.split("-").map(Number);
      return new Date(y, m, 0).getDate();
    }
    if (this.freq === "quarter") return 90;
    return 365;
  },

  async breakdown() {
    const d = await api(`/api/volume/breakdown?freq=${this.freq}`);
    const totals = (name) => d.rows.reduce((s, r) => s + (r.values[name] || 0), 0);
    const { keep, folded } = foldSeries(d.metrics, totals);
    const rows = applyFold(d.rows, keep, folded);
    paint("c-vol-breakdown", {
      type: "bar",
      data: { labels: rows.map((r) => r.period),
              datasets: keep.map((m, i) => barSet(m, rows.map((r) => r.values[m]), cat(i), { stacked: true })) },
      options: baseOptions({ stacked: true, unit: d.unit, decimals: 0 }),
    }, tableFrom(rows, keep));
  },

  async mismatch() {
    const d = await api(`/api/volume/mismatch?freq=${this.freq}`);
    paint("c-mismatch", {
      type: "bar",
      data: {
        labels: d.rows.map((r) => r.period),
        datasets: [
          barSet("Công suất khả dụng (mùa cao điểm)", d.rows.map((r) => r.supply ?? null), cat(0)),
          barSet("Phụ tải đỉnh (đầu cực)", d.rows.map((r) => r.demand ?? null), cat(1)),
        ],
      },
      options: baseOptions({ unit: d.unit, decimals: 0 }),
    }, {
      columns: ["Kỳ", "Khả dụng", "Phụ tải đỉnh", "Chênh lệch"],
      rows: d.rows.map((r) => [r.period, r.supply, r.demand, r.gap]),
    });
    const last = [...d.rows].reverse().find((r) => r.gap !== null && r.gap !== undefined);
    $("#mismatch-note").innerHTML = last
      ? `Kỳ gần nhất có đủ hai chiều (<b>${last.period}</b>): khả dụng ${fmt(last.supply, 0)} MW · đỉnh `
        + `${fmt(last.demand, 0)} MW · ${last.gap >= 0 ? "dư" : "thiếu"} <b>${fmt(Math.abs(last.gap), 0)} MW</b>. `
        + `Nguồn cung lấy dòng <code>${esc(d.supply_object)}</code>.`
      : "";
  },

  async installed() {
    const d = await api("/api/volume/capacity-installed");
    // A pie stays readable to ~6 slices; the long tail folds into "Khác" and the
    // full list lives in the table view.
    const { keep, folded } = foldSeries(d.rows.map((r) => r.type), (t) => d.rows.find((r) => r.type === t).value, 6);
    const value = (t) => t === OTHER
      ? folded.reduce((s, f) => s + d.rows.find((r) => r.type === f).value, 0)
      : d.rows.find((r) => r.type === t).value;
    const values = keep.map(value);
    const sum = values.reduce((a, b) => a + b, 0);
    paint("c-installed", {
      type: "doughnut",
      data: {
        labels: keep,
        datasets: [{
          data: values, backgroundColor: keep.map((_, i) => cat(i)),
          borderColor: ink().surface, borderWidth: 2, hoverOffset: 6,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false, cutout: "52%",
        plugins: {
          legend: { position: "right",
                    labels: { color: ink().muted, boxWidth: 9, boxHeight: 9, usePointStyle: true,
                              pointStyle: "rectRounded", padding: 9 } },
          tooltip: {
            backgroundColor: mode() === "dark" ? "#26262c" : "#1a1a1f", padding: 9, cornerRadius: 8,
            callbacks: { label: (ctx) => ` ${ctx.label}: ${fmt(ctx.parsed, 0)} MW (${fmt(ctx.parsed / sum * 100)}%)` },
          },
        },
      },
    }, {
      columns: ["Loại nguồn", "MW", "% tổng"],
      rows: d.rows.map((r) => [r.type, r.value, d.total ? r.value / d.total * 100 : null]),
    });
    $("#installed-note").innerHTML =
      `Tổng công suất đặt toàn quốc <b>${fmt(d.total, 0)} MW</b> · ngày ${esc(d.date)}`
      + (folded.length ? ` · ${folded.length} nguồn nhỏ gộp vào “${OTHER}” (bảng có đủ).` : "");
  },

  async region() {
    const d = await api(`/api/volume/capacity-region?freq=${this.freq}`);
    paint("c-region", {
      type: "bar",
      data: {
        labels: d.rows.map((r) => r.period),
        datasets: d.metrics.map((m, i) => barSet(m, d.rows.map((r) => r.values[m]), cat(i), { stacked: true })),
      },
      options: baseOptions({ stacked: true, unit: d.unit, decimals: 0 }),
    }, {
      columns: ["Kỳ", ...d.metrics, d.system_label],
      rows: d.rows.map((r) => [r.period, ...d.metrics.map((m) => r.values[m]), r.system]),
    });
    const last = d.rows[d.rows.length - 1];
    $("#volume-kpis").innerHTML = last ? d.metrics.map((m, i) => `
      <div class="kpi"><div class="k">${esc(m)} — ${esc(last.period)}</div>
        <div class="v">${fmt(last.values[m], 0)} <span class="u">MW</span></div></div>`).join("")
      + `<div class="kpi"><div class="k">${esc(d.system_label)} — ${esc(last.period)}</div>
           <div class="v">${fmt(last.system, 0)} <span class="u">MW</span></div></div>` : "";
  },
};

// ================================================================ Tab 6 — Doanh nghiệp
const company = {
  loaded: false, ticker: null, importantOnly: false,
  async load() {
    const { companies } = await api("/api/company/list");
    const sel = $("#ticker");
    sel.innerHTML = companies.map((c) =>
      `<option value="${c.ticker}">${c.ticker} — ${esc(c.name)}</option>`).join("");
    sel.onchange = () => this.select(sel.value);
    $("#company-news-imp").onclick = () => {
      this.importantOnly = !this.importantOnly;
      $("#company-news-imp").classList.toggle("active", this.importantOnly);
      this.news();
    };
    this.loaded = true;
    await this.select(companies[0].ticker);
  },

  async select(ticker) {
    this.ticker = ticker;
    $("#company-src").textContent = "";
    $("#company-charts").innerHTML = `<p class="loading">Đang tải ${esc(ticker)}…</p>`;
    $("#water-charts").innerHTML = "";
    $("#company-news").innerHTML = `<p class="loading">Đang tải tin…</p>`;
    await Promise.all([this.charts(), this.water(), this.news()]);
  },

  /** Chart list is derived from what the company's objects actually carry (§5.7). */
  specs(d) {
    const has = (objs, metric) => objs.filter((o) => d.series[o]?.[metric]);
    const out = [];
    if (d.segments.length) {
      const rev = has(d.segments, "net revenue");
      if (rev.length) out.push({ id: "seg-rev", title: "Doanh thu thuần theo mảng", type: "stacked",
                                 metric: "net revenue", objects: rev, unit: "VNDbn" });
      const gp = has(d.segments, "gross profit");
      const npat = has(d.segments, "NPAT");
      if (gp.length) out.push({ id: "seg-gp", title: "Lợi nhuận gộp theo mảng", type: "stacked",
                                metric: "gross profit", objects: gp, unit: "VNDbn" });
      else if (npat.length) out.push({ id: "seg-npat", title: "Lợi nhuận sau thuế theo mảng", type: "stacked",
                                       metric: "NPAT", objects: npat, unit: "VNDbn",
                                       note: "Nguồn chỉ có NPAT, chưa có cột lợi nhuận gộp." });
    }
    if (d.generation.length) {
      const rev = has(d.generation, "Revenue");
      if (rev.length) out.push({ id: "gen-rev", title: "Doanh thu theo nguồn phát", type: "stacked",
                                 metric: "Revenue", objects: rev, unit: "VNDbn" });
      const vol = has(d.generation, "Volume");
      if (vol.length) out.push({ id: "gen-vol", title: "Sản lượng theo nguồn phát", type: "stacked",
                                 metric: "Volume", objects: vol, unit: "triệu kWh" });
      const asp = has(d.generation, "ASP");
      if (asp.length) out.push({ id: "gen-asp", title: "Giá bán bình quân (ASP) theo nguồn", type: "line",
                                 metric: "ASP", objects: asp, unit: "VND/kWh" });
    }
    if (d.plants.length) {
      const plants = d.plants.filter((p) => p !== d.total_object);
      const rev = has(plants, "Revenue");
      if (rev.length) out.push({ id: "plant-rev", title: "Doanh thu theo nhà máy", type: "stacked",
                                 metric: "Revenue", objects: rev, unit: "VNDbn" });
      for (const [metric, title] of [["Mobilized", "Sản lượng huy động theo nhà máy"],
                                     ["Volume", "Sản lượng theo nhà máy"],
                                     ["Contracted", "Sản lượng hợp đồng theo nhà máy"]]) {
        const objs = has(plants, metric);
        if (objs.length) out.push({ id: `plant-${metric}`, title, type: "stacked",
                                    metric, objects: objs, unit: "triệu kWh" });
      }
      const asp = has(plants, "ASP");
      if (asp.length) out.push({ id: "plant-asp", title: "ASP theo nhà máy", type: "line",
                                 metric: "ASP", objects: asp, unit: "VND/kWh" });
    }
    // Single-object companies (HND / PPC / QTP): the ticker itself carries everything.
    if (d.total_object && !d.segments.length && !d.generation.length && !d.plants.length) {
      for (const [metric, title, unit, type] of [
        ["Revenue", "Doanh thu", "VNDbn", "bar"],
        ["Volume", "Sản lượng", "triệu kWh", "bar"],
        ["ASP", "Giá bán bình quân (ASP)", "VND/kWh", "line"]]) {
        if (d.series[d.total_object]?.[metric])
          out.push({ id: `solo-${metric}`, title, type, metric, objects: [d.total_object], unit });
      }
    }
    return out;
  },

  async charts() {
    const host = $("#company-charts");
    try {
      const d = await api(`/api/company/${this.ticker}`);
      $("#company-src").textContent = d.dataset;
      const specs = this.specs(d);
      host.innerHTML = specs.map((s) => `
        <div class="card">
          <div class="head"><h3>${esc(s.title)}</h3><span class="src">${esc(s.metric)}</span></div>
          <div class="chartbox"><canvas id="c-co-${s.id}"></canvas></div>
          ${s.note ? `<p class="note warn">⚠ ${esc(s.note)}</p>` : ""}
        </div>`).join("");

      const periods = d.periods;
      for (const s of specs) {
        const at = (obj, period) =>
          d.series[obj][s.metric]?.find((p) => p.period === period)?.value ?? null;
        const total = (obj) => d.series[obj][s.metric].reduce((acc, p) => acc + Math.abs(p.value), 0);
        const { keep, folded } = foldSeries(s.objects, total);
        const valueAt = (name, period) => name === OTHER
          ? folded.reduce((acc, f) => acc + (at(f, period) || 0), 0)
          : at(name, period);
        const stacked = s.type === "stacked";
        const datasets = keep.map((obj, i) => {
          const data = periods.map((p) => valueAt(obj, p));
          return s.type === "line" ? lineSet(obj, data, cat(i)) : barSet(obj, data, cat(i), { stacked });
        });
        paint(`c-co-${s.id}`, {
          type: s.type === "line" ? "line" : "bar",
          data: { labels: periods, datasets },
          options: baseOptions({ stacked, unit: s.unit, legend: keep.length > 1,
                                 decimals: s.unit === "VND/kWh" ? 0 : undefined }),
        }, {
          columns: ["Kỳ", ...keep],
          rows: periods.map((p) => [p, ...keep.map((o) => valueAt(o, p))]),
        });
      }
    } catch (e) { fail(host, e); }
  },

  async water() {
    const host = $("#water-charts");
    try {
      const d = await api(`/api/company/${this.ticker}/reservoirs`);
      const show = d.lakes.length > 0;
      $("#water-title").style.display = show ? "" : "none";
      $("#water-note").textContent = show
        ? `Cắt cùng mốc 1/1 → ${d.cutoff} mỗi năm. Tổng công suất thuỷ điện ${fmt(d.total_mw)} MW. ${d.note}`
        : "";
      if (!show) { host.innerHTML = ""; return; }
      host.innerHTML = d.lakes.map((l, i) => `
        <div class="card">
          <div class="head"><h3>${esc(l.lake)}</h3><span class="src">${esc(l.region || "")}</span></div>
          <div class="chartbox short"><canvas id="c-lake-${i}"></canvas></div>
          <p class="lakecap">${fmt(l.mw)} MW · ${fmt(l.share)}% công suất thuỷ điện của ${esc(d.ticker)}</p>
        </div>`).join("");
      d.lakes.forEach((l, i) => {
        paint(`c-lake-${i}`, {
          type: "bar",
          data: {
            labels: l.years.map((y) => y.year),
            datasets: [barSet("Mực nước TB luỹ kế", l.years.map((y) => y.value), cat(0))],
          },
          options: {
            ...baseOptions({ unit: "Mực nước (m)", legend: false }),
            scales: {
              ...baseOptions({ unit: "Mực nước (m)" }).scales,
              // Reservoir levels sit in a narrow band hundreds of metres up;
              // a zero baseline would flatten every bar to the same height.
              y: { ...baseOptions({ unit: "Mực nước (m)" }).scales.y, beginAtZero: false },
            },
          },
        }, {
          columns: ["Năm", "Mực nước TB (m)", "Số lần đo"],
          rows: l.years.map((y) => [y.year, y.value, y.readings]),
        });
      });
    } catch (e) { fail(host, e); }
  },

  async news() {
    const host = $("#company-news");
    try {
      const d = await api(`/api/company/${this.ticker}/news?important_only=${this.importantOnly}&limit=24`);
      host.innerHTML = d.items.length
        ? `<div class="newsgrid">` +
          [["document", "Công bố"], ["press", "Thông cáo"], ["news", "Tin tức"]].map(([type, label]) => {
            const items = d.items.filter((i) => i.type === type);
            return `<div><div class="sechead"><h2>${label}</h2>
                      <span class="count" style="margin-left:auto;color:var(--muted);font-size:12px">${items.length}</span></div>
                    ${items.map(newsCard).join("") || `<p class="empty">Chưa có</p>`}</div>`;
          }).join("") + `</div>`
        : `<p class="empty">Chưa có tin nào cho ${esc(this.ticker)} trong các feed hiện tại.</p>`;
    } catch (e) { fail(host, e); }
  },
};

function newsCard(it) {
  const title = it.url
    ? `<a href="${esc(it.url)}" target="_blank" rel="noopener">${esc(it.title || "(không tiêu đề)")} ↗</a>`
    : esc(it.title || "(không tiêu đề)");
  return `<article class="ncard">
    <div class="meta">${it.source ? `<span>${esc(it.source)}</span>` : ""}
      ${it.is_important ? `<span class="badge imp">Quan trọng</span>` : ""}
      <span class="date">${esc(it.date)}</span></div>
    <h4>${title}</h4>
    ${it.snippet ? `<p class="snip">${esc(it.snippet)}</p>` : ""}
    ${(it.tags || []).length ? `<div class="tags">${it.tags.map((t) => `<span>${esc(t)}</span>`).join("")}</div>` : ""}
  </article>`;
}

// ================================================================ Tab 7 — Tin tức
const news = {
  loaded: false,
  PAGE: 12,
  SECTIONS: [
    { type: "industry", label: "Tin ngành nghề" },
    { type: "company", label: "Tin doanh nghiệp" },
    { type: "document", label: "Công bố" },
    { type: "press", label: "Thông cáo" },
  ],
  filters: { q: "", tag: "", relevance: "relevant", date_from: "", date_to: "" },
  state: {},

  async load() {
    const grid = $("#newsgrid");
    grid.innerHTML = "";
    for (const s of this.SECTIONS) {
      const el = document.createElement("div");
      el.innerHTML = `<div class="sechead"><h2>${s.label}</h2>
          <span class="count" style="margin-left:auto;color:var(--muted);font-size:12px"></span></div>
        <div class="feed"></div>
        <p class="empty" style="display:none">Không có tin nào khớp bộ lọc.</p>
        <button class="more" style="display:none">Xem thêm</button>`;
      grid.appendChild(el);
      const ref = { type: s.type, skip: 0, total: 0, feed: $(".feed", el), empty: $(".empty", el),
                    more: $(".more", el), count: $(".count", el) };
      ref.more.addEventListener("click", () => this.section(ref, false));
      this.state[s.type] = ref;
    }
    this.wire();
    this.loaded = true;
    this.reload();
  },

  wire() {
    let debounce;
    $("#q").addEventListener("input", (e) => {
      clearTimeout(debounce);
      debounce = setTimeout(() => { this.filters.q = e.target.value.trim(); this.reload(); }, 300);
    });
    $("#relbtns").addEventListener("click", (e) => {
      const b = e.target.closest(".preset");
      if (!b) return;
      $$("#relbtns .preset").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      this.filters.relevance = b.dataset.rel;
      this.reload();
    });
    const iso = (d) => d.toISOString().slice(0, 10);
    $$(".dpreset").forEach((p) => p.addEventListener("click", () => {
      $$(".dpreset").forEach((x) => x.classList.remove("active"));
      p.classList.add("active");
      const days = +p.dataset.days;
      if (!days) { this.filters.date_from = this.filters.date_to = ""; $("#dfrom").value = $("#dto").value = ""; }
      else {
        const to = new Date(), from = new Date();
        from.setDate(from.getDate() - days + 1);
        this.filters.date_from = $("#dfrom").value = iso(from);
        this.filters.date_to = $("#dto").value = iso(to);
      }
      this.reload();
    }));
    [$("#dfrom"), $("#dto")].forEach((inp) => inp.addEventListener("change", () => {
      $$(".dpreset").forEach((x) => x.classList.remove("active"));
      this.filters.date_from = $("#dfrom").value;
      this.filters.date_to = $("#dto").value;
      this.reload();
    }));
    api("/api/tags").then(({ tickers, keywords }) => {
      const opt = (t) => `<option value="${esc(t.tag)}">${esc(t.tag)} (${t.count})</option>`;
      const sel = $("#tagselect");
      if (tickers?.length) sel.insertAdjacentHTML("beforeend",
        `<optgroup label="Mã cổ phiếu">${tickers.map(opt).join("")}</optgroup>`);
      if (keywords?.length) sel.insertAdjacentHTML("beforeend",
        `<optgroup label="Từ khoá">${keywords.map(opt).join("")}</optgroup>`);
      sel.onchange = () => { this.filters.tag = sel.value; this.reload(); };
    }).catch(() => {});
  },

  reload() { for (const t in this.state) this.section(this.state[t], true); },

  async section(ref, reset) {
    if (reset) { ref.skip = 0; ref.feed.innerHTML = ""; }
    ref.more.disabled = true;
    const p = new URLSearchParams({ type: ref.type, ...this.filters, skip: ref.skip, limit: this.PAGE });
    try {
      const { items, total } = await api("/api/news?" + p);
      ref.total = total;
      ref.count.textContent = total;
      ref.feed.insertAdjacentHTML("beforeend", items.map(newsCard).join(""));
      ref.skip += items.length;
      ref.empty.style.display = total === 0 ? "block" : "none";
      ref.more.style.display = ref.skip < total ? "block" : "none";
    } catch (e) { fail(ref.feed, e); }
    ref.more.disabled = false;
  },
};

// ================================================================ boot
const TABS = { gov, weather, cost, price, volume, company, news };
let activeTab = "gov";

async function show(name, { push = true } = {}) {
  if (!TABS[name]) name = "gov";
  activeTab = name;
  if (push && location.hash.slice(1) !== name) location.hash = name;
  $$("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".tabpage").forEach((p) => p.classList.toggle("active", p.id === `tab-${name}`));
  const tab = TABS[name];
  if (tab && !tab.loaded) {
    try { await tab.load(); }
    catch (e) { fail($(`#tab-${name}`), e); }
  }
}

$("#tabs").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (b) show(b.dataset.tab);
});

// Repaint on a light/dark switch — the palettes are per-mode, not a filter.
darkMQ.addEventListener("change", () => {
  for (const [name, tab] of Object.entries(TABS)) {
    if (tab.loaded && name !== "news") { tab.loaded = false; if (name === activeTab) show(name); }
  }
});

// Each tab has its own URL fragment (#volume, #company…) so a view is linkable.
window.addEventListener("hashchange", () => show(location.hash.slice(1), { push: false }));

api("/api/meta").then((m) => {
  const latest = m.datasets.map((d) => d.updated_at).filter(Boolean).sort().pop();
  $("#stamp").textContent = `${m.datasets.length} nguồn dữ liệu · cập nhật ${latest || "—"}`;
}).catch(() => {});

show(location.hash.slice(1) || "gov", { push: false });
