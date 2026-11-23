const byId = (id) => document.getElementById(id);
const integer = new Intl.NumberFormat("zh-CN");
const SENTIMENT = {
  "-2": {label: "未提及", tone: "neutral"},
  "-1": {label: "负面", tone: "negative"},
  "0": {label: "中性", tone: "neutral"},
  "1": {label: "正面", tone: "positive"},
};
const QUICK_ASPECTS = [
  "Food#Taste", "Food#Portion", "Price#Level", "Service#Queue",
  "Service#Timely", "Ambience#Sanitary",
];
const EXAMPLE = "菜很好吃，但是等了四十分钟才上菜，前台态度很差。";
const state = {overview: null, nextOffset: null, browseRequest: 0, detailRequest: 0};

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function button(label, className, onClick) {
  const result = element("button", label, className);
  result.type = "button";
  result.addEventListener("click", onClick);
  return result;
}

async function request(path, options) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.message || "暂时无法读取，请重试");
  return data;
}

function sentimentTag(value, label) {
  const state = SENTIMENT[value];
  return element("span", label || state.label, "tag " + state.tone);
}

const aspectFilter = byId("aspect-filter");
const sentimentFilter = byId("sentiment-filter");
const loadMore = byId("load-more");

function setFilter(aspect, sentiment) {
  aspectFilter.value = aspect;
  if (sentiment !== undefined) sentimentFilter.value = sentiment;
  syncQuickAspects();
  loadReviews();
}

function syncQuickAspects() {
  for (const chip of byId("quick-aspects").children) {
    chip.setAttribute("aria-pressed", String(chip.dataset.aspect === aspectFilter.value));
  }
}

function populateOverview(data) {
  state.overview = data;
  const groups = new Map();
  for (const aspect of data.aspects) {
    if (!groups.has(aspect.group)) {
      const group = element("optgroup");
      group.label = aspect.group;
      groups.set(aspect.group, group);
      aspectFilter.append(group);
    }
    const option = element("option", aspect.label);
    option.value = aspect.aspect;
    groups.get(aspect.group).append(option);
  }
  for (const aspect of [{aspect: "", label: "全部"}, ...QUICK_ASPECTS.map(key =>
    data.aspects.find(row => row.aspect === key))]) {
    if (!aspect) continue;
    const chip = button(aspect.label, "chip", () => setFilter(aspect.aspect));
    chip.dataset.aspect = aspect.aspect;
    byId("quick-aspects").append(chip);
  }
  syncQuickAspects();
  const fills = [
    {id: "positive-list", field: "positive", polarity: "1", label: "条正面"},
    {id: "negative-list", field: "negative", polarity: "-1", label: "条负面"},
  ];
  for (const fill of fills) {
    const list = byId(fill.id);
    list.replaceChildren();
    const ranked = [...data.aspects].sort((a, b) => b[fill.field] - a[fill.field]);
    for (const aspect of ranked.filter(row => row[fill.field] > 0).slice(0, 3)) {
      const item = element("li");
      const link = button("", "", () => {
        setFilter(aspect.aspect, fill.polarity);
        byId("reviews").scrollIntoView({behavior: "auto"});
        aspectFilter.focus({preventScroll: true});
      });
      link.append(
        element("span", aspect.label),
        element("span", integer.format(aspect[fill.field]) + " " + fill.label, "row-count"),
      );
      item.append(link);
      list.append(item);
    }
  }
  byId("summary-grid").setAttribute("aria-busy", "false");
}

function reviewCard(item, focusAspect, focusSentiment) {
  const card = element("article", undefined, "review-card");
  const tags = element("div", undefined, "tags");
  for (const tag of item.tags) {
    tags.append(sentimentTag(tag.sentiment, tag.name + " · " + SENTIMENT[tag.sentiment].label));
  }
  if (tags.childElementCount) card.append(tags);
  card.append(
    element("p", item.preview),
    button("读完整评论", "text-button", () =>
      openReview(item.id, focusAspect || (focusSentiment !== "" ? item.tags[0]?.aspect : ""))),
  );
  return card;
}

async function loadReviews(more = false) {
  const generation = ++state.browseRequest;
  const offset = more ? state.nextOffset : 0;
  const focusAspect = aspectFilter.value;
  const focusSentiment = sentimentFilter.value;
  const query = new URLSearchParams({offset: String(offset)});
  if (focusAspect) query.set("aspect", focusAspect);
  if (focusSentiment !== "") query.set("sentiment", focusSentiment);
  const grid = byId("review-grid");
  const status = byId("review-status");
  loadMore.disabled = true;
  if (!more) {
    grid.replaceChildren();
    byId("review-count").textContent = "";
    loadMore.hidden = true;
  }
  status.textContent = "正在加载评论";
  try {
    const data = await request("/api/reviews?" + query);
    if (generation !== state.browseRequest) return;
    const cards = data.items.map(row => reviewCard(row, focusAspect, focusSentiment));
    grid.append(...cards);
    if (more && cards.length) cards[0].querySelector("button").focus();
    byId("review-count").textContent = integer.format(data.total) + " 条匹配评论";
    status.textContent = data.total ? "" : "这个筛选条件下暂无评论";
    state.nextOffset = data.next_offset;
    loadMore.hidden = data.next_offset === null;
  } catch (error) {
    if (generation === state.browseRequest) status.textContent = error.message;
  } finally {
    if (generation === state.browseRequest) loadMore.disabled = false;
  }
}

function readable(text) {
  return text.replace(/\\+[rn]/g, "\n");
}

function renderDetail(container, review, preferred) {
  container.replaceChildren();
  const grid = element("div", undefined, "detail-grid");
  const left = element("div");
  const right = element("div");
  const original = element("p", undefined, "original");
  const status = element("p", "", "state detail-status");
  status.setAttribute("role", "status");
  const methods = element("fieldset", undefined, "method-choice");
  const methodName = container.id + "-method";
  const aspectList = element("div", undefined, "aspect-list");
  const mentioned = review.aspects.filter(row => row.mentioned);
  let active = mentioned.find(row => row.aspect === preferred) ||
    mentioned.find(row => row.evidence && row.evidence.tfidf) ||
    mentioned.find(row => row.sentiment === -1) || mentioned[0];
  let method = "tfidf";

  left.append(element("h3", review.star ? "原文 · 原评分 " + review.star + " / 5" : "评论原文"), original, status);
  methods.append(element("legend", "证据排序"));
  for (const [key, name] of [["tfidf", "TF-IDF"], ["textrank", "TextRank"]]) {
    const input = element("input");
    input.type = "radio";
    input.name = methodName;
    input.value = key;
    input.checked = key === method;
    input.addEventListener("change", () => {
      method = key;
      highlight();
    });
    const label = element("label");
    label.append(input, document.createTextNode(name));
    methods.append(label);
  }
  left.append(methods);

  const controls = new Map();
  for (const row of mentioned) {
    const control = button("", "aspect-option", () => {
      active = row;
      highlight();
    });
    control.append(element("span", row.name), sentimentTag(row.sentiment));
    controls.set(row.aspect, control);
    aspectList.append(control);
  }
  right.append(element("h3", "识别出的方面"), aspectList);
  if (!mentioned.length) right.append(element("p", "这条评论暂未识别到评价方面", "state"));

  const unmentioned = review.aspects.filter(row => !row.mentioned);
  if (unmentioned.length) {
    const details = element("details");
    details.append(
      element("summary", "未识别出 " + unmentioned.length + " 个方面"),
      element("p", unmentioned.map(row => row.name).join("、"), "unmentioned-list"),
    );
    right.append(details);
  }

  function highlight() {
    for (const [key, control] of controls) {
      control.setAttribute("aria-pressed", String(active?.aspect === key));
    }
    methods.hidden = active?.sentiment !== -1;
    const evidence = active?.evidence?.[method];
    const text = review.text;
    original.replaceChildren();
    if (evidence && text.slice(evidence.start_utf16, evidence.end_utf16) === evidence.text) {
      original.append(
        document.createTextNode(readable(text.slice(0, evidence.start_utf16))),
        element("mark", readable(evidence.text)),
        document.createTextNode(readable(text.slice(evidence.end_utf16))),
      );
      status.textContent = active.name + " · 原文证据";
    } else {
      original.textContent = readable(text);
      status.textContent = active?.sentiment === -1 ? active.name + " · 暂未定位原文证据" : "";
    }
  }

  grid.append(left, right);
  container.append(grid);
  highlight();
}

async function openReview(id, preferred) {
  const dialog = byId("review-dialog");
  const container = byId("review-detail");
  const generation = ++state.detailRequest;
  container.replaceChildren(element("p", "正在读取评论原文", "state"));
  dialog.showModal();
  try {
    const result = await request("/api/review?" + new URLSearchParams({id}));
    if (generation === state.detailRequest) renderDetail(container, result, preferred);
  } catch (error) {
    if (generation === state.detailRequest) container.replaceChildren(element("p", error.message, "state"));
  }
}

function table(headers, rows) {
  const result = element("table");
  const head = result.createTHead().insertRow();
  for (const name of headers) {
    const th = element("th", name);
    th.scope = "col";
    head.append(th);
  }
  const body = result.createTBody();
  for (const values of rows) {
    const row = body.insertRow();
    values.forEach((value, index) => {
      const cell = element(index === 0 ? "th" : "td", value);
      if (index === 0) cell.scope = "row";
      row.append(cell);
    });
  }
  const wrap = element("div", undefined, "table-wrap");
  wrap.append(result);
  return wrap;
}

aspectFilter.addEventListener("change", () => {
  syncQuickAspects();
  loadReviews();
});
sentimentFilter.addEventListener("change", () => loadReviews());
loadMore.addEventListener("click", () => loadReviews(true));

const analysisDialog = byId("analysis-dialog");
for (const control of document.querySelectorAll("[data-open-analysis]")) {
  control.addEventListener("click", () => analysisDialog.showModal());
}
byId("fill-example").addEventListener("click", () => {
  byId("review-input").value = EXAMPLE;
  byId("review-input").focus();
});
byId("analysis-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const status = byId("analysis-status");
  const result = byId("analysis-result");
  const submit = byId("analyze-button");
  const review = byId("review-input").value;
  if (!review.trim()) {
    status.textContent = "请先输入一条评论";
    return;
  }
  result.replaceChildren();
  status.textContent = "正在分析评论，首次加载模型需要几秒。";
  submit.disabled = true;
  try {
    const data = await request("/api/analyze", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({review}),
    });
    renderDetail(result, data, "");
    status.textContent = "";
  } catch (error) {
    status.textContent = error.message;
  } finally {
    submit.disabled = false;
  }
});

const metricsDialog = byId("metrics-dialog");
for (const control of document.querySelectorAll("[data-open-metrics]")) {
  control.addEventListener("click", async () => {
    const container = byId("metrics-content");
    container.replaceChildren(element("p", "正在读取评估结果", "state"));
    metricsDialog.showModal();
    try {
      renderMetrics(container, await request("/api/metrics"));
    } catch (error) {
      container.replaceChildren(element("p", error.message, "state"));
    }
  });
}

request("/api/overview").then(populateOverview).catch((error) => {
  for (const id of ["positive-list", "negative-list"]) {
    byId(id).replaceChildren(element("li", error.message, "state"));
  }
  byId("summary-grid").setAttribute("aria-busy", "false");
});
loadReviews();

