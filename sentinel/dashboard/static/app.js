/* Project Sentinel dashboard — Plotly charts, auto-refresh, toast, dark mode. */
(function () {
  "use strict";

  /* colour palette */
  var C = {
    blue: "#3056d3", navy: "#1e40af", red: "#ef4444",
    green: "#10b981", amber: "#f59e0b", purple: "#8b5cf6",
    pink: "#ec4899", cyan: "#06b6d4", slate: "#64748b",
    grid: "#e2e8f0", line: "#cbd5e1",
    seq: ["#3056d3","#10b981","#f59e0b","#ef4444","#8b5cf6","#ec4899","#06b6d4","#64748b"],
  };

  /* chart definitions */
  var charts = {

    /* Drift timeline */
    "drift-timeline": {
      url: "/api/drift/timeseries",
      build: function (p) {
        return [{
          x: p.timestamps || [], y: p.statistics || [],
          mode: "lines+markers", type: "scatter",
          line: { color: C.blue, width: 2 }, marker: { color: C.navy, size: 5 },
          name: "drift statistic",
        }];
      },
      layout: function (p) {
        var shapes = [];
        if (p.threshold != null && (p.timestamps || []).length) {
          shapes.push({ type: "line", xref: "paper", x0: 0, x1: 1,
            y0: p.threshold, y1: p.threshold,
            line: { color: C.red, width: 1, dash: "dash" } });
        }
        return baseLayout({ shapes: shapes,
          yaxis: { title: "statistic", gridcolor: C.grid, linecolor: C.line } });
      },
    },

    /* Token daily cost with budget threshold */
    "tokens-daily": {
      url: "/api/tokens/daily?days=14",
      build: function (p) {
        var daily = p.daily || [];
        return [{
          x: daily.map(function (d) { return d.day; }),
          y: daily.map(function (d) { return d.cost_usd; }),
          type: "bar", marker: { color: C.blue, opacity: 0.85 },
          name: "cost (USD)",
        }];
      },
      layout: function (p) {
        var shapes = [];
        var budget = p.budgets && p.budgets.daily_max_cost;
        if (budget && (p.daily || []).length) {
          shapes.push({ type: "line", xref: "paper", x0: 0, x1: 1,
            y0: budget, y1: budget,
            line: { color: C.red, width: 1.5, dash: "dash" } });
        }
        return baseLayout({ shapes: shapes,
          yaxis: { title: "USD", gridcolor: C.grid, linecolor: C.line } });
      },
    },

    /* Feature importance horizontal bar */
    "feature-importance": {
      url: "/api/features/chart",
      build: function (p) {
        var feats = (p.features || []).slice(0, 15);
        var names = feats.map(function (f) { return f.name; }).reverse();
        var imp = feats.map(function (f) { return f.importance; }).reverse();
        var drift = feats.map(function (f) { return f.drift_score; }).reverse();
        var colors = feats.map(function (f) {
          return f.is_drifted ? C.red : C.blue;
        }).reverse();
        return [
          { y: names, x: imp, type: "bar", orientation: "h", name: "importance",
            marker: { color: colors, opacity: 0.8 } },
          { y: names, x: drift, type: "bar", orientation: "h", name: "drift score",
            marker: { color: C.amber, opacity: 0.6 } },
        ];
      },
      layout: function () {
        return baseLayout({
          barmode: "group",
          margin: { l: 120, r: 16, t: 16, b: 36 },
          xaxis: { title: "score", gridcolor: C.grid, linecolor: C.line },
          legend: { x: 0.7, y: 1.05, orientation: "h", font: { size: 10 } },
        });
      },
    },

    /* Drift detail per-feature bar */
    "drift-feature-scores": {
      url: null,
      build: function (p) {
        var sorted = (p.feature_scores_sorted || []);
        var names = sorted.map(function (f) { return f[0]; }).reverse();
        var scores = sorted.map(function (f) { return f[1]; }).reverse();
        var colors = scores.map(function (s) {
          return s > (p.threshold || 0.2) ? C.red : C.blue;
        });
        return [{ y: names, x: scores, type: "bar", orientation: "h",
          marker: { color: colors, opacity: 0.85 }, name: "drift score" }];
      },
      layout: function (p) {
        var shapes = [];
        if (p.threshold) {
          shapes.push({ type: "line", yref: "paper", y0: 0, y1: 1,
            x0: p.threshold, x1: p.threshold,
            line: { color: C.red, width: 1, dash: "dash" } });
        }
        return baseLayout({ shapes: shapes, margin: { l: 120, r: 16, t: 16, b: 36 },
          xaxis: { title: "score", gridcolor: C.grid, linecolor: C.line } });
      },
    },

    /* Compliance events donut */
    "compliance-events": {
      url: "/api/compliance/chart",
      build: function (p) {
        var labels = p.labels || [];
        var values = p.values || [];
        if (!labels.length) return [];
        return [{
          labels: labels, values: values, type: "pie",
          hole: 0.45, textinfo: "label+percent",
          marker: { colors: C.seq.slice(0, labels.length) },
          textfont: { size: 10 },
        }];
      },
      layout: function () {
        return baseLayout({ showlegend: false, margin: { l: 20, r: 20, t: 10, b: 10 } });
      },
    },

    /* Agent trace waterfall */
    "agent-trace-waterfall": {
      url: null,
      build: function (p) {
        var spans = p.spans || [];
        if (!spans.length) return [];
        var base = spans[0].offset_ms || 0;
        var names = spans.map(function (s) { return s.name; }).reverse();
        var starts = spans.map(function (s) { return (s.offset_ms || 0) - base; }).reverse();
        var durations = spans.map(function (s) { return s.duration_ms || 0; }).reverse();
        var colors = spans.map(function (s) {
          if (s.kind === "tool_call") return C.amber;
          if (s.status === "error") return C.red;
          return C.blue;
        }).reverse();
        return [{ y: names, x: durations, base: starts, type: "bar", orientation: "h",
          marker: { color: colors, opacity: 0.85 },
          text: durations.map(function (d) { return d + "ms"; }),
          textposition: "outside", name: "duration" }];
      },
      layout: function () {
        return baseLayout({
          margin: { l: 140, r: 40, t: 16, b: 36 },
          xaxis: { title: "time (ms)", gridcolor: C.grid, linecolor: C.line },
          bargap: 0.3,
        });
      },
    },

    /* Tool success/failure grouped bar */
    "tool-success-rates": {
      url: "/api/agentops/tools/chart",
      build: function (p) {
        var tools = p.tools || [];
        if (!tools.length) return [];
        var names = tools.map(function (t) { return t.name; });
        return [
          { x: names, y: tools.map(function (t) { return t.successes; }),
            type: "bar", name: "successes", marker: { color: C.green, opacity: 0.85 } },
          { x: names, y: tools.map(function (t) { return t.failures; }),
            type: "bar", name: "failures", marker: { color: C.red, opacity: 0.85 } },
        ];
      },
      layout: function () {
        return baseLayout({
          barmode: "group",
          xaxis: { tickangle: -30, gridcolor: C.grid, linecolor: C.line },
          yaxis: { title: "count", gridcolor: C.grid, linecolor: C.line },
          legend: { x: 0.8, y: 1.05, orientation: "h", font: { size: 10 } },
        });
      },
    },

    /* Guardrail violations trend */
    "guardrail-violations-trend": {
      url: "/api/llmops/guardrails/trend",
      build: function (p) {
        var days = p.days || [];
        if (!days.length) return [];
        return [{
          x: days.map(function (d) { return d.day; }),
          y: days.map(function (d) { return d.count; }),
          mode: "lines+markers", type: "scatter",
          line: { color: C.red, width: 2 },
          marker: { color: C.red, size: 5 },
          fill: "tozeroy", fillcolor: "rgba(239,68,68,0.08)",
          name: "violations",
        }];
      },
      layout: function () {
        return baseLayout({ yaxis: { title: "violations", gridcolor: C.grid, linecolor: C.line } });
      },
    },

    /* Token cost by model donut */
    "tokens-by-model": {
      url: "/api/tokens/by-model",
      build: function (p) {
        var models = p.models || [];
        if (!models.length) return [];
        return [{
          labels: models.map(function (m) { return m.model; }),
          values: models.map(function (m) { return m.cost_usd; }),
          type: "pie", hole: 0.45, textinfo: "label+percent",
          marker: { colors: C.seq.slice(0, models.length) },
          textfont: { size: 10 },
        }];
      },
      layout: function () {
        return baseLayout({ showlegend: false, margin: { l: 20, r: 20, t: 10, b: 10 } });
      },
    },

    /* Cohort comparison bar chart */
    "cohortComparison": {
      url: "/api/cohorts/chart",
      build: function (p) {
        var cohorts = p.cohorts || [];
        if (!cohorts.length) return [];
        var flags = p.disparity_flags || [];
        var names = cohorts.map(function (c) { return c.cohort_id; });
        var accs = cohorts.map(function (c) { return c.accuracy != null ? c.accuracy * 100 : 0; });
        var colors = cohorts.map(function (c) {
          return flags.indexOf(c.cohort_id) !== -1 ? C.red : C.blue;
        });
        return [{
          x: names, y: accs, type: "bar",
          marker: { color: colors, opacity: 0.85 },
          name: "accuracy %",
        }];
      },
      layout: function () {
        return baseLayout({
          yaxis: { title: "accuracy %", gridcolor: C.grid, linecolor: C.line, range: [0, 105] },
        });
      },
    },

    /* Global feature importance horizontal bar */
    "globalImportance": {
      url: "/api/explanations/global",
      build: function (p) {
        var feats = (p.features || []).slice(0, 15);
        if (!feats.length) return [];
        var names = feats.map(function (f) { return f.name; }).reverse();
        var vals = feats.map(function (f) { return f.importance; }).reverse();
        return [{
          y: names, x: vals, type: "bar", orientation: "h",
          marker: { color: C.green, opacity: 0.85 },
          name: "mean |attribution|",
        }];
      },
      layout: function () {
        return baseLayout({
          margin: { l: 120, r: 16, t: 16, b: 36 },
          xaxis: { title: "mean |attribution|", gridcolor: C.grid, linecolor: C.line },
        });
      },
    },
  };

  /* base layout helper */
  function baseLayout(extra) {
    var layout = {
      margin: { l: 40, r: 16, t: 16, b: 36 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: "Inter, system-ui, sans-serif", size: 11, color: "#475569" },
      xaxis: { gridcolor: C.grid, linecolor: C.line },
      yaxis: { gridcolor: C.grid, linecolor: C.line },
    };
    return Object.assign(layout, extra || {});
  }

  /* chart render */
  function renderChart(el) {
    var key = el.getAttribute("data-chart");
    var cfg = charts[key];
    if (!cfg || !window.Plotly) return;
    var url = el.getAttribute("data-url") || cfg.url;
    if (!url) return;
    el.classList.add("chart-loading");
    fetch(url)
      .then(function (r) { return r.json(); })
      .then(function (payload) {
        el.classList.remove("chart-loading");
        var traces = cfg.build(payload);
        if (!traces || !traces.length) {
          el.innerHTML = '<div class="text-xs text-slate-400 text-center py-10">no data to chart</div>';
          return;
        }
        window.Plotly.newPlot(el, traces, cfg.layout(payload), {
          displayModeBar: false, responsive: true,
        });
      })
      .catch(function () {
        el.classList.remove("chart-loading");
        el.innerHTML = '<div class="text-xs text-slate-400 text-center py-10">chart unavailable</div>';
      });
  }

  /* auto-refresh */
  function startAutoRefresh() {
    var el = document.querySelector("[data-refresh-interval]");
    var secs = el ? parseInt(el.getAttribute("data-refresh-interval"), 10) : 0;
    if (!secs || secs < 5) return;
    setInterval(function () {
      var chartEls = document.querySelectorAll("[data-chart]");
      for (var i = 0; i < chartEls.length; i++) renderChart(chartEls[i]);
    }, secs * 1000);
  }

  /* toast notifications */
  window.sentinelToast = function (msg, type) {
    type = type || "info";
    var container = document.getElementById("toast-container");
    if (!container) return;
    var colors = {
      success: "bg-emerald-600", error: "bg-red-600",
      warning: "bg-amber-500", info: "bg-slate-700",
    };
    var icons = { success: "\u2713", error: "\u2717", warning: "\u26A0", info: "\u2139" };
    var div = document.createElement("div");
    div.className = "toast-item " + (colors[type] || colors.info) +
      " text-white text-sm px-4 py-3 rounded-lg shadow-lg flex items-center gap-2";
    div.innerHTML = "<span class='font-medium'>" + (icons[type] || icons.info) + "</span> " + msg;
    container.appendChild(div);
    setTimeout(function () {
      div.style.opacity = "0";
      div.style.transform = "translateX(100%)";
      setTimeout(function () { div.remove(); }, 300);
    }, 4000);
  };

  /* HTMX toast on POST actions */
  document.addEventListener("htmx:afterRequest", function (evt) {
    var xhr = evt.detail.xhr;
    if (!xhr || evt.detail.requestConfig.verb !== "post") return;
    if (xhr.status >= 200 && xhr.status < 300) {
      window.sentinelToast("Action completed successfully", "success");
    } else {
      var msg = "Request failed";
      try { msg = JSON.parse(xhr.responseText).detail || msg; } catch (e) { /* ignore */ }
      window.sentinelToast(msg, "error");
    }
  });

  /* dark mode toggle */
  window.toggleDarkMode = function () {
    document.documentElement.classList.toggle("dark");
    var isDark = document.documentElement.classList.contains("dark");
    localStorage.setItem("sentinel-theme", isDark ? "dark" : "light");
  };
  (function () {
    if (localStorage.getItem("sentinel-theme") === "dark") {
      document.documentElement.classList.add("dark");
    }
  })();

  /* active nav highlighting */
  function highlightActiveNav() {
    var path = window.location.pathname;
    var links = document.querySelectorAll("header nav a");
    for (var i = 0; i < links.length; i++) {
      var href = links[i].getAttribute("href");
      var isActive = (href === "/" && path === "/") ||
                     (href !== "/" && path.startsWith(href));
      if (isActive) {
        links[i].classList.add("bg-slate-700", "text-white");
      }
    }
  }

  /* init */
  function init() {
    var els = document.querySelectorAll("[data-chart]");
    for (var i = 0; i < els.length; i++) renderChart(els[i]);
    highlightActiveNav();
    startAutoRefresh();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
