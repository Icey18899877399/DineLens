const VISUAL_MODELS = ["传统基线", "联合 BERT"];
const MATRIX_PALETTE = [
  "#fdf390", "#cdec9e", "#a5deb6", "#8ecfc1",
  "#91c0c8", "#99aec6", "#a099bf", "#9e7fa9",
];
const VISUAL_COUNTS = [
  ["zero", "无候选"],
  ["one", "单候选"],
  ["multiple", "多候选"],
];
const scoreText = (value) => value.toFixed(3);
const percentText = (value) => value === null ? "—" : (100 * value).toFixed(1) + "%";

function evaluationSection(title, description) {
  const section = element("section", undefined, "metrics-section");
  section.append(element("h3", title));
  if (description) section.append(element("p", description, "state"));
  return section;
}

function plotLegend() {
  const legend = element("div", undefined, "plot-legend");
  for (const [id, name] of [["baseline", VISUAL_MODELS[0]], ["bert", VISUAL_MODELS[1]]]) {
    const key = element("span", name, "plot-key");
    key.prepend(element("i", undefined, "plot-mark " + id));
    key.firstChild.setAttribute("aria-hidden", "true");
    legend.append(key);
  }
  return legend;
}

function renderModelSummary(data) {
  const result = element("div");
  const summary = evaluationSection(
    "模型测试结果",
    "同一批 " + integer.format(data.test_reviews) + " 条 ASAP 测试评论。情感 Macro-F1 依据人工标记为提及的方面计算。",
  );
  summary.append(table(
    ["模型", "方面 Macro-F1", "情感 Macro-F1", "端到端负面 Micro-F1"],
    data.models.map(model => [
      model.name, scoreText(model.aspect_macro_f1), scoreText(model.sentiment_macro_f1),
      scoreText(model.end_to_end_negative_f1),
    ]),
  ));
  result.append(summary);
  if (state.overview) {
    const comparison = evaluationSection("负面评论数对照", "按预测条数和人工标注条数的差距排序。");
    const aspects = [...state.overview.aspects].sort((a, b) =>
      Math.abs(b.negative - b.gold_negative) - Math.abs(a.negative - a.gold_negative));
    comparison.append(table(["方面", "人工标注", "模型预测", "漏报", "误报"], aspects.map(aspect => [
      aspect.name, integer.format(aspect.gold_negative), integer.format(aspect.negative),
      integer.format(aspect.false_negative), integer.format(aspect.false_positive),
    ])));
    result.append(comparison);
  }
  return result;
}

function renderAspectPlot(data) {
  const section = evaluationSection(
    "18 个方面的提及识别",
    "横轴是每个方面的 F1。点击一行查看精确率、召回率和人工提及数。",
  );
  const chart = element("div", undefined, "aspect-plot");
  chart.append(plotLegend());
  const axis = element("div", undefined, "aspect-axis");
  const ticks = element("div", undefined, "aspect-ticks");
  for (const value of ["0", "0.25", "0.50", "0.75", "1.00"]) ticks.append(element("span", value));
  axis.append(element("span", "方面"), ticks, element("span", "F1"));
  chart.append(axis);
  const readout = element("div", undefined, "evaluation-readout");
  readout.id = "aspect-statistics";
  readout.setAttribute("role", "status");
  const controls = [];
  let lastGroup;
  for (const row of data.aspects) {
    if (row.group !== lastGroup) {
      chart.append(element("p", row.group, "plot-group"));
      lastGroup = row.group;
    }
    const track = element("span", undefined, "score-track");
    track.setAttribute("aria-hidden", "true");
    const from = 100 * row.baseline.f1;
    const to = 100 * row.bert.f1;
    const baseline = element("i", undefined, "plot-mark baseline");
    baseline.style.width = from + "%";
    const bert = element("i", undefined, "plot-mark bert");
    bert.style.width = to + "%";
    track.append(baseline, bert);
    const control = button(undefined, "aspect-plot-row", () => {
      select(row, control);
      readout.scrollIntoView({block: "nearest"});
    });
    control.setAttribute("aria-controls", readout.id);
    control.setAttribute("aria-label", row.name + "，基线 F1 " +
      scoreText(row.baseline.f1) + "，BERT F1 " + scoreText(row.bert.f1));
    control.append(element("span", row.name), track,
      element("span", scoreText(row.bert.f1), "score-value"));
    controls.push(control);
    chart.append(control);
  }

  function select(row, active) {
    for (const control of controls) control.setAttribute("aria-pressed", String(control === active));
    active.after(readout);
    readout.replaceChildren(
      element("strong", row.name + " · 人工提及 " + integer.format(row.support) + " 次"),
      element("span", rowSummary(VISUAL_MODELS[0], row.baseline)),
      element("span", rowSummary(VISUAL_MODELS[1], row.bert)),
    );
  }

  function rowSummary(name, values) {
    return name + "　精确率 " + scoreText(values.precision) +
      "　召回率 " + scoreText(values.recall) + "　F1 " + scoreText(values.f1);
  }

  const example = Math.max(0, data.aspects.findIndex(row => row.aspect === "Location#Downtown"));
  if (controls.length) select(data.aspects[example], controls[example]);
  section.append(chart);
  return section;
}

function matrixColor(rate) {
  if (rate === null) return "#f1f1f1";
  const position = rate * (MATRIX_PALETTE.length - 1);
  const index = Math.min(Math.floor(position), MATRIX_PALETTE.length - 2);
  const weight = (position - index) * 100;
  return "color-mix(in srgb, " + MATRIX_PALETTE[index] + ", " + MATRIX_PALETTE[index + 1] + " " + weight + "%)";
}

function renderSentimentPlots(data) {
  const section = evaluationSection(
    "情感混淆矩阵",
    "切换模型可对照同一格。行是人工情感，列是模型预测。色阶表示行内比例（0–100%）。",
  );
  const modelPicker = element("div", undefined, "chips matrix-models");
  modelPicker.setAttribute("aria-label", "情感矩阵模型");
  const block = element("div", undefined, "matrix-block");
  const readout = element("p", undefined, "evaluation-readout");
  readout.setAttribute("role", "status");
  const scale = element("div", undefined, "matrix-scale");
  scale.setAttribute("aria-hidden", "true");
  const ramp = element("i", undefined, "matrix-ramp");
  ramp.style.backgroundImage = "linear-gradient(to top in srgb, " + MATRIX_PALETTE.join(", ") + ")";
  const ticks = element("div", undefined, "matrix-ticks");
  ticks.append(...[100, 75, 50, 25, 0].map(value => element("span", value + "%")));
  scale.append(ramp, ticks);
  const labels = data.sentiment_labels;
  let focus = [1, 2];
  const controls = data.models.map((model, index) =>
    button(model.name, "chip", () => showModel(index)));
  modelPicker.append(...controls);

  function showModel(index) {
    controls.forEach((control, i) => control.setAttribute("aria-pressed", String(i === index)));
    const model = data.models[index];
    const matrix = model.confusion;
    const grid = element("table", undefined, "sentiment-matrix");
    grid.setAttribute("aria-label", model.name + " 三分类混淆矩阵");
    const headers = grid.createTHead().insertRow();
    const comparison = element("th");
    comparison.scope = "col";
    comparison.append(element("span", "人工／"), element("span", "预测"));
    headers.append(comparison);
    for (const name of labels) {
      const header = element("th", name);
      header.scope = "col";
      headers.append(header);
    }
    const body = grid.createTBody();
    const cells = [];
    for (const [i, gold] of labels.entries()) {
      const row = body.insertRow();
      const header = element("th", gold);
      header.scope = "row";
      header.append(element("span", "n=" + integer.format(matrix.row_support[i]), "state"));
      row.append(header);
      for (const [j, prediction] of labels.entries()) {
        const count = matrix.counts[i][j];
        const n = matrix.row_support[i];
        const rate = matrix.rates[i][j];
        const label = model.name + " · 人工" + gold + "判为" + prediction + " · " +
          integer.format(count) + " / " + integer.format(n) + "（" + percentText(rate) + "）";
        const control = button(undefined, "matrix-cell", () => {
          focus = [i, j];
          select(control, label);
        });
        control.setAttribute("aria-label", label);
        control.style.backgroundColor = matrixColor(rate);
        control.append(element("strong", percentText(rate)), element("span", integer.format(count)));
        row.insertCell().append(control);
        cells.push(control);
      }
    }
    block.replaceChildren(grid, scale);
    const current = cells[focus[0] * labels.length + focus[1]];
    select(current, current.getAttribute("aria-label"));

    function select(active, label) {
      for (const cell of cells) cell.setAttribute("aria-pressed", String(cell === active));
      readout.textContent = label;
    }
  }

  showModel(1);
  section.append(modelPicker, block, readout,
    element("p", "总体 " + integer.format(data.sentiment_support) + " 个由人工标为已提及的方面。", "state"));
  return section;
}

function renderEvidencePlot(data) {
  const evidence = data.evidence;
  const total = evidence.n_pairs_with_gold_negative_aspect;
  const counts = evidence.candidate_counts;
  const section = evaluationSection(
    "原文证据候选",
    integer.format(total) + " 个人工标为负面的“评论＋方面”组合。两种方法在同一批候选上排序。",
  );
  const bar = element("div", undefined, "evidence-stack");
  bar.setAttribute("aria-hidden", "true");
  const breakdown = element("ul", undefined, "coverage-list");
  for (const [key, name] of VISUAL_COUNTS) {
    const proportion = total ? counts[key] / total : 0;
    const segment = element("i", undefined, key);
    segment.style.flexBasis = 100 * proportion + "%";
    bar.append(segment);
    const item = element("li", name + " " + integer.format(counts[key]) + "（" + percentText(proportion) + "）");
    item.prepend(element("i", undefined, "coverage-key " + key));
    item.firstChild.setAttribute("aria-hidden", "true");
    breakdown.append(item);
  }
  section.append(bar, breakdown);

  const example = evidence.example;
  const demo = evaluationSection("同一条原文，两种排序", "演示评论 · " + example.aspect);
  const quote = element("p", undefined, "original");
  const methods = element("div", undefined, "chips evidence-methods");
  methods.setAttribute("aria-label", "演示证据排序");
  const buttons = [];
  for (const [key, name] of [["tfidf", "TF-IDF"], ["textrank", "TextRank"]]) {
    const control = button(name, "chip", () => select(key, control));
    buttons.push(control);
    methods.append(control);
  }
  const note = element("p", undefined, "state detail-status");
  note.setAttribute("role", "status");

  function select(method, active) {
    for (const control of buttons) control.setAttribute("aria-pressed", String(control === active));
    const {start_utf16: start, end_utf16: end} = example.methods[method];
    quote.replaceChildren(
      document.createTextNode(example.text.slice(0, start)),
      element("mark", example.text.slice(start, end)),
      document.createTextNode(example.text.slice(end)),
    );
    const other = example.methods[method === "tfidf" ? "textrank" : "tfidf"];
    note.textContent = start === other.start_utf16 && end === other.end_utf16 ?
      "两种方法选中同一段原文。" : "两种方法选中的原文位置不同。";
  }

  select("tfidf", buttons[0]);
  demo.append(methods, quote, note);
  const result = element("div");
  result.append(section, demo);
  return result;
}

function renderMetrics(container, data) {
  container.replaceChildren();
  const navigation = element("div", undefined, "chips evaluation-tabs");
  navigation.setAttribute("aria-label", "评估图表");
  const content = element("div", undefined, "evaluation-content");
  const panels = [
    ["模型总览", renderModelSummary],
    ["方面识别", renderAspectPlot],
    ["情感判断", renderSentimentPlots],
    ["原文证据", renderEvidencePlot],
  ];
  const controls = panels.map(([label, draw]) => button(label, "chip", () => show(draw)));
  navigation.append(...controls);
  container.append(navigation, content);

  function show(draw) {
    panels.forEach((panel, index) => controls[index].setAttribute("aria-pressed", String(panel[1] === draw)));
    content.replaceChildren(draw(data));
    container.parentElement.scrollTop = 0;
  }

  show(renderModelSummary);
}

