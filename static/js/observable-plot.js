import * as Plot from "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6/+esm";
import { csvParse } from "https://cdn.jsdelivr.net/npm/d3-dsv@3/+esm";

const plotNodes = document.querySelectorAll("[data-observable-plot]");

plotNodes.forEach((node) => {
  renderPlot(node);
});

async function renderPlot(node) {
  const mount = node.querySelector(".observable-plot__mount");
  if (!mount) return;

  const labels = (node.dataset.series || "Baseline,ASURA (x3),Variant")
    .split(",")
    .map((label) => label.trim())
    .filter(Boolean);

  const seed = Number(node.dataset.seed || 42);
  const total = Number(node.dataset.points || 72);
  const showDots = node.dataset.showDots !== "false";
  const showLastLabels = node.dataset.showLastLabels === "true";
  const lastLabelDecimals = parseOptionalNumber(node.dataset.lastLabelDecimals);
  const smoothWindow = normalizeSmoothWindow(node.dataset.smoothWindow);
  const csvUrl = node.dataset.csvUrl;
  const csvColumns = (node.dataset.columns || "")
    .split(",")
    .map((col) => col.trim())
    .filter(Boolean);
  const xColumn = node.dataset.x || "Step";
  const colors = (node.dataset.colors || "")
    .split(",")
    .map((color) => color.trim())
    .filter(Boolean);
  const xMin = parseOptionalNumber(node.dataset.xMin);
  const xMax = parseOptionalNumber(node.dataset.xMax);
  const yMin = parseOptionalNumber(node.dataset.yMin);
  const yMax = parseOptionalNumber(node.dataset.yMax);
  const xScale = sanitizeScale(node.dataset.xScale);
  const yScale = sanitizeScale(node.dataset.yScale);

  let series = null;
  if (csvUrl && csvColumns.length > 0) {
    series = await loadSeriesFromCsv(csvUrl, csvColumns, labels, xColumn);
    if (!series) {
      mount.textContent = "Chart data unavailable.";
      return;
    }
  } else {
    series = buildSyntheticSeries(labels.length, total, seed);
  }

  const rawSeries = series;
  if (smoothWindow > 1) {
    series = series.map((values) => smoothSeries(values, smoothWindow));
  }

  const defaultStrokes = ["var(--plot-line)", "var(--plot-line-2)", "var(--plot-line-3)"];
  const strokeVars = colors.length > 0 ? colors : defaultStrokes;
  const marks = [];
  if (smoothWindow > 1) {
    rawSeries.forEach((values, index) => {
      if (!values || values.length === 0) return;
      marks.push(
        Plot.line(values, {
          x: "step",
          y: "value",
          stroke: strokeVars[index] || "var(--plot-line)",
          strokeWidth: 1.4 - index * 0.1,
          strokeOpacity: 0.25
        })
      );
    });
  }
  series.forEach((values, index) => {
    if (!values || values.length === 0) return;
    marks.push(
      Plot.line(values, {
        x: "step",
        y: "value",
        stroke: strokeVars[index] || "var(--plot-line)",
        strokeWidth: 2.2 - index * 0.1
      })
    );
  });

  if (showDots && series[0] && series[0].length > 0) {
    marks.push(
      Plot.dot(series[0].filter((d) => d.step % 6 === 0), {
        x: "step",
        y: "value",
        fill: "var(--plot-dot)",
        stroke: "var(--plot-fg)",
        strokeOpacity: 0.35,
        strokeWidth: 0.8,
        r: 2.1
      })
    );

    const last = series[0][series[0].length - 1];
    marks.push(
      Plot.dot([last], {
        x: "step",
        y: "value",
        fill: "var(--plot-emphasis)",
        stroke: "var(--plot-fg)",
        strokeOpacity: 0.6,
        strokeWidth: 1,
        r: 3.4
      })
    );
  }

  if (showLastLabels) {
    const decimals = Number.isFinite(lastLabelDecimals) ? Math.max(0, Math.round(lastLabelDecimals)) : 3;
    const lastPoints = series
      .map(findLastPoint)
      .map((point, index) =>
        point
          ? {
              step: point.step,
              value: point.value,
              seriesIndex: index,
              label: formatValue(point.value, decimals)
            }
          : null
      )
      .filter(Boolean);

    if (lastPoints.length > 0) {
      marks.push(
        Plot.dot(lastPoints, {
          x: "step",
          y: "value",
          fill: (d) => strokeVars[d.seriesIndex] || "var(--plot-fg)",
          stroke: "var(--plot-bg)",
          strokeOpacity: 0.7,
          strokeWidth: 1.2,
          r: 3.6
        })
      );
      marks.push(
        Plot.text(lastPoints, {
          x: "step",
          y: "value",
          text: "label",
          dx: -6,
          dy: -6,
          textAnchor: "end",
          fontSize: 12,
          fontWeight: 600,
          fill: (d) => strokeVars[d.seriesIndex] || "var(--plot-fg)",
          stroke: "var(--plot-bg)",
          strokeWidth: 3
        })
      );
    }
  }

  const width = 720;
  const height = 320;
  const xLabel = node.dataset.xLabel || (csvUrl ? "Step" : "Iteration");
  const yLabel = node.dataset.yLabel || (csvUrl ? "Val/ppl" : "Synthetic loss");
  const xLabelOffset = parseOptionalNumber(node.dataset.xLabelOffset) ?? 28;
  const yLabelOffset = parseOptionalNumber(node.dataset.yLabelOffset) ?? 34;
  const xDomain = buildDomain(xMin, xMax);
  const yDomain = buildDomain(yMin, yMax);

  const plot = Plot.plot({
    width,
    height,
    marginLeft: 42,
    marginRight: 18,
    marginTop: 23,
    marginBottom: 38,
    x: {
      label: xLabel,
      grid: true,
      domain: xDomain,
      type: xScale,
      clamp: xDomain ? true : false,
      labelOffset: xLabelOffset
    },
    y: {
      label: yLabel,
      grid: true,
      domain: yDomain,
      type: yScale,
      clamp: yDomain ? true : false,
      labelOffset: yLabelOffset
    },
    marks
  });

  plot.classList.add("asura-plot-svg");
  plot.setAttribute("viewBox", `0 0 ${width} ${height}`);
  plot.setAttribute("preserveAspectRatio", "xMidYMid meet");
  plot.style.width = "100%";
  plot.style.height = "auto";
  plot.style.background = "var(--plot-bg)";
  plot.style.color = "var(--plot-fg)";
  plot.style.fontFamily = "'IBM Plex Sans', sans-serif";

  mount.replaceChildren(plot);

  if (colors.length > 0) {
    const swatches = node.querySelectorAll(".observable-plot__legend-swatch");
    swatches.forEach((swatch, index) => {
      if (colors[index]) {
        swatch.style.background = colors[index];
      }
    });
  }
}

function buildSyntheticSeries(count, total, seed) {
  const rng = mulberry32(seed);
  const series = [];

  for (let idx = 0; idx < Math.max(1, count); idx += 1) {
    const points = [];
    const start = 2.7 - idx * 0.12;
    const end = 1.45 + idx * 0.08;
    const phase = idx * 0.55;

    for (let step = 0; step < total; step += 1) {
      const t = step / (total - 1);
      const baseline = start * (1 - t) + end * t;
      const noise = (rng() - 0.5) * (0.14 - idx * 0.015);
      const wiggle = 0.08 * Math.sin(step / (5 + idx) + phase) + 0.05 * Math.cos(step / (9 + idx));
      const value = baseline + noise + wiggle;
      points.push({ step, value: Number(value.toFixed(3)) });
    }

    series.push(points);
  }

  return series;
}

function mulberry32(seed) {
  let t = seed >>> 0;
  return function () {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

async function loadSeriesFromCsv(url, columns, labels, xColumn) {
  try {
    const response = await fetch(url);
    if (!response.ok) return null;
    const text = await response.text();
    const rows = csvParse(text);
    const series = labels.map(() => []);

    rows.forEach((row) => {
      const step = Number(row[xColumn]);
      if (!Number.isFinite(step)) return;
      columns.forEach((colName, index) => {
        const rawValue = row[colName];
        if (rawValue === undefined || rawValue === "") return;
        const value = Number(rawValue);
        if (!Number.isFinite(value)) return;
        if (!series[index]) return;
        series[index].push({ step, value });
      });
    });

    return series;
  } catch (error) {
    return null;
  }
}

function parseOptionalNumber(value) {
  if (value === undefined || value === null || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function buildDomain(min, max) {
  if (min === null || max === null) return undefined;
  return [min, max];
}

function sanitizeScale(value) {
  if (!value) return undefined;
  const normalized = value.trim().toLowerCase();
  if (normalized === "log") return "log";
  if (normalized === "linear") return "linear";
  return undefined;
}

function findLastPoint(points) {
  if (!points || points.length === 0) return null;
  return points.reduce((latest, point) => (point.step > latest.step ? point : latest), points[0]);
}

function formatValue(value, decimals) {
  if (!Number.isFinite(value)) return "";
  return value.toFixed(decimals);
}

function normalizeSmoothWindow(value) {
  const parsed = parseOptionalNumber(value);
  if (!Number.isFinite(parsed) || parsed <= 0) return 0;
  return Math.max(1, Math.round(parsed));
}

function smoothSeries(points, windowSize) {
  if (!points || points.length === 0) return points;
  const size = Math.max(1, Math.round(windowSize));
  if (size <= 1) return points;

  const smoothed = [];
  const values = [];
  let sum = 0;
  let start = 0;

  for (let i = 0; i < points.length; i += 1) {
    const point = points[i];
    const value = point.value;
    values.push(value);
    sum += value;

    while (i - start + 1 > size) {
      sum -= values[start];
      start += 1;
    }

    const count = i - start + 1;
    const avg = sum / count;
    smoothed.push({ step: point.step, value: avg });
  }

  return smoothed;
}
