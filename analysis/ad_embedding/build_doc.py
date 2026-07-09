"""Build the AD-score-normalization meeting HTML, embedding figures as data URIs.

Regenerates the whole page deterministically (avoids editing the huge
base64-laden file in place). Pulls live numbers from figs/scales_all.json.
"""
import base64, json, os, pathlib

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
OUT = "/tmp/claude-1003/-home-sohyung/d2215767-409a-4ad9-a00a-fd723ed5a4a7/scratchpad/meeting_normalization.html"


def datauri(png):
    b = pathlib.Path(os.path.join(FIGS, png)).read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode()


rows = json.load(open(os.path.join(FIGS, "scales_all.json")))
# per-module cross-env summary
mods = ["svdd", "dsebm", "mahal", "gmm"]
summ = {}
for m in mods:
    hs = [r["hill_med"] for r in rows if r["module"] == m]
    meds = [r["median"] for r in rows if r["module"] == m]
    summ[m] = (min(meds), max(meds), min(hs), max(hs),
               any(r["sign"] == "has_neg" for r in rows if r["module"] == m))

CSS = """
  :root{
    --ground:#f6f7f9; --surface:#ffffff; --ink:#1a1d23; --muted:#5b6470;
    --hair:#e2e6ea; --accent:#1f6f8b; --accent-soft:#e8f0f3;
    --bad:#b23b3b; --bad-bg:#fbeceb; --warn:#9a6b00; --warn-bg:#faf1d8;
    --ok:#2f7a4f; --ok-bg:#e7f2ec; --crimson:#c0392b;
    --serif:Georgia,"Times New Roman",serif;
    --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
    line-height:1.6;-webkit-font-smoothing:antialiased}
  .wrap{max-width:min(940px,92vw);margin:0 auto;padding:56px 0 96px}
  .eyebrow{font-family:var(--mono);font-size:.72rem;letter-spacing:.16em;
    text-transform:uppercase;color:var(--accent);margin:0 0 14px}
  h1{font-family:var(--serif);font-weight:600;font-size:2.3rem;line-height:1.15;
    margin:0 0 12px;text-wrap:balance;letter-spacing:-.01em}
  h2{font-family:var(--serif);font-weight:600;font-size:1.42rem;margin:52px 0 14px;
    text-wrap:balance;letter-spacing:-.005em}
  h3{font-size:1.02rem;margin:28px 0 8px;font-weight:650}
  p{margin:0 0 14px;max-width:68ch}
  .lede{font-size:1.08rem;color:var(--muted);max-width:70ch}
  .meta{display:flex;flex-wrap:wrap;gap:8px 20px;margin:22px 0 0;
    font-family:var(--mono);font-size:.78rem;color:var(--muted)}
  .meta b{color:var(--ink);font-weight:600}
  code{font-family:var(--mono);font-size:.86em;background:var(--accent-soft);
    padding:.1em .38em;border-radius:4px;color:#13525f}
  pre{font-family:var(--mono);font-size:.82rem;line-height:1.55;background:#0f1419;
    color:#d6dde6;padding:18px 20px;border-radius:10px;overflow-x:auto;margin:14px 0}
  pre .c{color:#6b7785} pre .k{color:#9ad0e0} pre .r{color:#e8a0a0}
  .callout{border:1px solid var(--hair);border-left:4px solid var(--accent);
    background:var(--surface);border-radius:10px;padding:22px 24px;margin:22px 0}
  .callout.rec{border-left-color:var(--ok);background:var(--ok-bg)}
  .callout.note{border-left-color:var(--warn);background:var(--warn-bg)}
  .callout h3{margin-top:0}
  .scroll{overflow-x:auto;margin:16px 0;border:1px solid var(--hair);border-radius:10px}
  table{border-collapse:collapse;width:100%;font-size:.9rem;background:var(--surface)}
  th,td{padding:11px 14px;text-align:left;border-bottom:1px solid var(--hair);white-space:nowrap}
  thead th{font-family:var(--mono);font-size:.7rem;letter-spacing:.08em;
    text-transform:uppercase;color:var(--muted);background:#fbfcfd}
  tbody tr:last-child td{border-bottom:none}
  td.num,th.num{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}
  .tag{font-family:var(--mono);font-size:.74rem;font-weight:600;padding:.16em .5em;
    border-radius:5px;display:inline-block}
  .t-bad{color:var(--bad);background:var(--bad-bg)}
  .t-warn{color:var(--warn);background:var(--warn-bg)}
  .t-ok{color:var(--ok);background:var(--ok-bg)}
  figure{margin:20px 0}
  figure img{width:100%;height:auto;display:block;border:1px solid var(--hair);border-radius:10px}
  figcaption{font-size:.82rem;color:var(--muted);margin-top:8px;font-style:italic}
  ul,ol{margin:0 0 14px;padding-left:22px;max-width:68ch} li{margin:6px 0}
  .verdict{font-family:var(--serif);font-size:1.15rem;color:var(--ink)}
  .hair{height:1px;background:var(--hair);border:0;margin:40px 0}
  .foot{font-size:.8rem;color:var(--muted);margin-top:48px}
  .q li{margin:12px 0}
"""


def tag(h):
    if h < 0.05 or h >= 1.0:
        return '<span class="tag t-bad">깨짐/소멸</span>'
    if h >= 0.7:
        return '<span class="tag t-warn">포화</span>'
    return '<span class="tag t-ok">적정대</span>'


sum_rows = ""
labels = {"svdd": "svdd (거리)", "dsebm": "dsebm (에너지)",
          "mahal": "mahal (제곱거리)", "gmm": "gmm (neg-log-lik)"}
for m in mods:
    lo_m, hi_m, lo_h, hi_h, neg = summ[m]
    sign = '<span style="color:var(--crimson)">음수 포함</span>' if neg else "≥ 0"
    sum_rows += (f'<tr><td>{labels[m]}</td><td>{sign}</td>'
                 f'<td class="num">{lo_m:.4g} … {hi_m:.4g}</td>'
                 f'<td class="num">{lo_h:.3f} … {hi_h:.3f}</td>'
                 f'<td>{tag((lo_h+hi_h)/2)}</td></tr>')

HTML = f"""<title>AD Score Normalization — 회의 자료</title>
<style>{CSS}</style>

<div class="wrap">
  <p class="eyebrow">RLAD · Anomaly Penalty · Discussion</p>
  <h1>AD anomaly score는 모듈·환경마다 스케일이 다르다 — 어디서 정규화할 것인가</h1>
  <p class="lede">현재 코드는 7개 AD 모듈의 raw score를 각자 다른 공식으로 추출하지만, 그 값을 penalty로 바꾸는 단계에서 <b>모든 모듈에 동일한 <code>T=5.0</code></b>을 적용한다. 9개 D4RL 환경 전체에서 측정한 결과, score는 모듈 간은 물론 같은 모듈도 환경마다 스케일이 갈렸다. 정규화를 어느 위치에 넣을지 결정이 필요하다.</p>
  <div class="meta">
    <span>측정 env <b>9개 (3 task × 3 dataset)</b></span>
    <span>모듈 <b>svdd · dsebm · mahal · gmm</b></span>
    <span>표본 <b>env당 50,000 (s,a)</b></span>
    <span>branch <b>sh_ad-embedding-viz</b></span>
    <span>코드 <b>rlkit/torch/sac/rlad.py</b></span>
  </div>

  <div class="callout rec">
    <h3>결론 (먼저)</h3>
    <p class="verdict">정규화는 <b>위치 ① — <code>calc_anomaly_score</code> 직후의 전용 정규화 레이어</b>에 넣는다.</p>
    <p style="margin-bottom:0">위치 ②(<code>make_weight_function</code>에 모듈별 <code>T</code> 주입)는 <b>음수 score(DSEBM)와 분포 모양</b>을 못 다루고 weight 함수에 모듈·데이터셋 지식을 떠넘긴다. ①은 "비교 가능한 score를 만든다"는 책임을 한 곳에 모으고, [0,1] bound·앙상블·로깅·시각화를 모두 살린다. 방법은 <b>D 기반 robust-z 또는 CDF(percentile)</b>를 AD pretrain 직후 1회 산출·저장.</p>
  </div>

  <h2>1. 문제: 추출은 모듈별, 정규화는 부재</h2>
  <p><code>calc_anomaly_score</code>가 모듈마다 분기하는 건 맞다. 하지만 그건 <b>raw score를 "추출"하는 공식</b>이 다른 것이지(거리·에너지·음의 로그우도 등), 공통 스케일로 "정규화"하는 게 아니다. 추출된 값은 그대로 흘러가 동일한 weight 함수를 만난다:</p>
  <pre><span class="c"># rlad.py — 모든 모듈 공통, ad_module 인자조차 받지 않음</span>
def make_weight_function(name):
    if name == 'hill':
        <span class="k">T = 5.0</span>                       <span class="c"># ← 하드코딩, 모듈 무관</span>
        return lambda w: w / (w + T)
    ...

<span class="c"># 흐름</span>
raw score  <span class="r">(모듈·env마다 0.001 ~ −880 까지)</span>
   │   <span class="r">← 정규화 단계 없음</span>
   ▼
hill:  score / (score + 5.0)     <span class="c"># 같은 T를 전부에</span>
penalty = penalty_coef * weight</pre>
  <p>결과적으로 같은 <code>penalty_coef</code>가 모듈·환경마다 전혀 다른 penalty 강도로 작동한다. 추출은 모듈별로 다른데, 정규화/캘리브레이션 단계가 빠져 있다.</p>

  <div class="callout note">
    <h3>참고: 이미 존재하는 정규화는 "입력 정규화"이지 "score 정규화"가 아니다</h3>
    <p style="margin-bottom:0"><code>maf</code>는 <code>fit</code>에서 입력 (s,a)의 mean/std를 구해 <code>score()</code>에서 <code>x_norm=(x−mean)/std</code>로 <b>입력을 정규화</b>한 뒤 raw NLL을 반환한다(<code>mahal</code>도 공분산 whitening으로 입력을 정규화). 즉 <b>입력 feature</b> 정규화일 뿐, <b>출력 anomaly score</b>를 모듈 간 공통 스케일로 맞추는 정규화는 어디에도 없다. svdd/dsebm/dagmm/gmm/fanogan엔 입력 정규화조차 없다.</p>
  </div>

  <h2>2. 증거 (a): 같은 T=5.0이 모듈마다 다른 곳에 떨어진다 — hopper-medium 예시</h2>
  <p>hopper-medium-v2의 in-distribution (s,a) 50k개를 4개 모듈로 채점한 결과다. <code>hill(median)</code> 열은 "그 모듈의 전형적(중앙값) score가 T=5.0 hill을 통과하면 weight가 얼마가 되는가"로, penalty가 실제로 어떻게 작동하는지를 보여준다.</p>
  <div class="scroll">
  <table>
    <thead><tr>
      <th>모듈 (score 종류)</th><th>부호</th>
      <th class="num">median</th><th class="num">mean</th><th class="num">p99</th><th class="num">std</th>
      <th class="num">hill(median)</th><th>판정</th>
    </tr></thead>
    <tbody>
      <tr><td>svdd <span style="color:var(--muted)">(거리)</span></td><td>≥ 0</td>
        <td class="num">0.0052</td><td class="num">0.0063</td><td class="num">0.0214</td><td class="num">0.0047</td>
        <td class="num">0.0010</td><td><span class="tag t-bad">penalty ≈ 0</span></td></tr>
      <tr><td>dsebm <span style="color:var(--muted)">(에너지)</span></td><td style="color:var(--crimson)">음수</td>
        <td class="num">−89.29</td><td class="num">−88.98</td><td class="num">−85.56</td><td class="num">1.43</td>
        <td class="num">1.0593</td><td><span class="tag t-bad">[0,1) 붕괴</span></td></tr>
      <tr><td>mahal <span style="color:var(--muted)">(제곱거리)</span></td><td>≥ 0</td>
        <td class="num">12.05</td><td class="num">14.04</td><td class="num">46.25</td><td class="num">8.35</td>
        <td class="num">0.7068</td><td><span class="tag t-warn">거의 포화</span></td></tr>
      <tr><td>gmm <span style="color:var(--muted)">(neg-log-lik)</span></td><td>혼합</td>
        <td class="num">4.90</td><td class="num">6.55</td><td class="num">27.27</td><td class="num">5.14</td>
        <td class="num">0.4947</td><td><span class="tag t-ok">우연히 적정</span></td></tr>
    </tbody>
  </table>
  </div>
  <figure>
    <img src="__SCALES_FIG__" alt="모듈별 raw AD score 분포; 빨간 점선은 하드코딩된 T=5.0">
    <figcaption>각 패널의 x축 범위가 전부 다르다. 빨간 점선(T=5.0)은 svdd·dsebm에선 데이터 범위 밖, mahal에선 왼쪽 끝, gmm에서만 중앙 부근에 떨어진다 — 단일 상수가 모든 모듈에 맞을 수 없음을 보여준다.</figcaption>
  </figure>

  <h2>3. 증거 (b): 9개 환경 전체 — 모듈 간뿐 아니라 환경 간에도 갈린다</h2>
  <p>학습 완료된 9개 D4RL 환경(hopper·walker2d·halfcheetah × medium·medium-replay·medium-expert) 전부에서 측정했다. 아래 heatmap은 각 칸의 <code>hill(median)</code> — 0.5에 가까울수록 penalty가 정상 작동, 0에 가까우면 penalty가 사라지고, 1에 가까우면 포화/붕괴.</p>
  <figure>
    <img src="__HEATMAP_FIG__" alt="env × module 별 hill(median) heatmap">
    <figcaption>세로축 9개 환경, 가로축 4개 모듈. svdd는 전 환경에서 ≈0(penalty 소멸, 초록), dsebm은 전부 &gt;1(붕괴, 진빨강), mahal·gmm은 포화대. gmm은 같은 모듈인데도 환경 따라 0.49→0.84로 크게 출렁인다.</figcaption>
  </figure>
  <h3>모듈별 9-env 범위 요약</h3>
  <div class="scroll">
  <table>
    <thead><tr><th>모듈</th><th>부호</th><th class="num">median 범위 (9 env)</th><th class="num">hill(median) 범위</th><th>판정</th></tr></thead>
    <tbody>{sum_rows}</tbody>
  </table>
  </div>
  <p style="font-size:.86rem;color:var(--muted)">핵심: 단일 T=5.0로는 (i) 모듈 간 스케일 차이도, (ii) <b>같은 모듈의 환경 간 스케일 차이</b>(예: dsebm energy median이 −62 ~ −878로 14배, gmm hill이 0.49 ~ 0.84)도 흡수할 수 없다. dagmm·fanogan·maf는 아직 미학습이라 미측정 — 각자 또 다른 family라 추가 측정 시 표를 채운다.</p>

  <h2>4. 정규화 방법 (무엇으로)</h2>
  <p>핵심 원칙: <b>AD가 "정상"이라 정의한 분포 = offline dataset D 위의 score 통계로 보정</b>한다. AD pretrain 직후 D 전체를 1회 채점해 통계를 저장하고, RL 학습 때 그 통계로 정규화한다(매 스텝 재계산 금지 — 정책 따라 분포가 바뀜).</p>
  <div class="scroll">
  <table>
    <thead><tr><th>방법</th><th>변환</th><th>범위</th><th>장점</th><th>주의</th></tr></thead>
    <tbody>
      <tr><td><b>CDF / percentile</b></td><td>p = D에서 score 이하 비율</td><td>[0,1]</td>
        <td>스케일·부호·분포모양 전부 흡수, 해석 직관적("상위 몇 %")</td>
        <td>OOD가 전부 p≈1로 포화 → OOD끼리 구분 소실</td></tr>
      <tr><td><b>robust z</b></td><td>(score − median)/MAD, 후 sigmoid</td><td>(0,1)</td>
        <td>최소 변경, tail의 상대 크기 보존</td><td>분포가 강한 bimodal이면 보정 약함</td></tr>
      <tr><td>data-driven T</td><td>hill, T = median(score)</td><td>[0,1)</td>
        <td>기존 hill 그대로, 변경 최소</td><td>음수 score엔 shift 필요, 여전히 모양 무시</td></tr>
    </tbody>
  </table>
  </div>
  <p>어느 쪽이든 <b>[0,1] bounded</b>가 되어 논문의 <code>‖Q−Q_pen‖ ≤ penalty_coef/(1−γ)</code> 보증이 복원되고, <code>penalty_coef</code> 하나가 강도를 제어하는 유일한 손잡이가 된다.</p>

  <h2>5. 어디에 넣을 것인가 — 위치 ① vs ②</h2>
  <div class="scroll">
  <table>
    <thead><tr>
      <th>기준</th>
      <th>① calc_anomaly_score 직후 (정규화 레이어)</th>
      <th>② make_weight_function에 모듈별 T</th>
    </tr></thead>
    <tbody>
      <tr><td>음수 score(DSEBM) 처리</td><td><span class="tag t-ok">가능</span> 센터링으로 흡수</td><td><span class="tag t-bad">불가</span> hill은 score≥0 가정</td></tr>
      <tr><td>분포 모양(skew·bimodal)</td><td><span class="tag t-ok">가능</span> CDF가 모양 무관</td><td><span class="tag t-bad">불가</span> 스케일 1개로는 부족</td></tr>
      <tr><td>환경 간 스케일 차이</td><td><span class="tag t-ok">env별 통계로 흡수</span></td><td><span class="tag t-warn">env마다 T 또 따로</span></td></tr>
      <tr><td>[0,1] bound 복원</td><td><span class="tag t-ok">전 모듈 균일</span></td><td><span class="tag t-warn">≥0 모듈만</span></td></tr>
      <tr><td>책임 분리</td><td><span class="tag t-ok">깨끗</span> 추출+보정=score / weight=모양</td><td><span class="tag t-bad">과부하</span> weight가 모듈·env를 알아야</td></tr>
      <tr><td>앙상블·비교·로깅·시각화</td><td><span class="tag t-ok">전부 정규화된 값 사용</span></td><td><span class="tag t-bad">weight 밖은 여전히 raw</span></td></tr>
      <tr><td>코드 변경 크기</td><td><span class="tag t-warn">중간</span> 통계 저장 필요</td><td><span class="tag t-ok">작음</span> T 인자만</td></tr>
    </tbody>
  </table>
  </div>
  <div class="callout">
    <h3>판단: 위치 ①</h3>
    <p>②의 유일한 장점은 "변경이 작다"인데, 그건 <b>실제 문제를 안 푼다</b> — DSEBM 음수, 분포 모양, 그리고 환경 간 스케일 차이에서 그대로 깨진다. ①만이 부호·모양·env차이를 다루고, bound를 전 모듈 균일하게 복원하며, 정규화된 score를 앙상블/로깅/시각화 전반에서 재사용하게 한다. 책임 분리도 자연스럽다: <code>calc_anomaly_score</code>(+정규화) = "쓸 수 있는 score를 만든다", <code>make_weight_function</code> = "모양만 입힌다".</p>
    <p style="margin-bottom:0"><b>구현 형태:</b> 모듈·env별 통계를 AD 가중치 옆에 저장(예: <code>weights_ad/{{module}}/{{env}}/score_norm.json</code> — median/MAD 또는 quantile knots). 얇은 <code>ScoreNormalizer.fit(D)</code> / <code>.transform(score)</code>를 <code>calc_anomaly_score</code> 반환 직후에 적용. <code>make_weight_function</code>은 정규화된 [0,1] 위에서 동작(또는 T 제거).</p>
  </div>

  <h2>6. 회의에서 정할 것</h2>
  <ol class="q">
    <li><b>CDF vs robust-z.</b> CDF는 OOD를 전부 p≈1로 포화시켜 "OOD 중 더 OOD"를 못 가린다. penalty엔 tail 순서가 중요할 수 있으니 robust-z(또는 z 후 완만한 squashing)가 나을지?</li>
    <li><b>통계 산출 분포.</b> D(in-distribution)로 fit하는데, penalty는 정책이 만든 OOD (s,a′)에 걸린다 — D 밖에선 정규화가 외삽이다. clip/extrapolation 정책을 어떻게?</li>
    <li><b>per-env vs per-module-global.</b> 9-env 측정 결과 환경 간 차이가 커서, 데이터셋별 통계를 따로 두는 쪽이 유력 — 합의 필요.</li>
    <li><b>방향 통일.</b> 모든 모듈을 "높을수록 이상"으로 부호 정렬(역방향 모듈은 flip).</li>
    <li><b>DSEBM per-sample 버그.</b> 정규화 이전에 <code>energy_per_sample</code>로 교체 필요(이미 viz 브랜치에 수정본 있음).</li>
  </ol>

  <hr class="hair">
  <p class="foot">측정: <code>analysis/ad_embedding/measure_scales_all.py</code> · 데이터 <code>figs/scales_all.json</code> (9 env × 4 module = 36행) · svdd·dsebm은 학습 가중치(9 env 배치 완료), mahal·gmm은 즉석 fit. dagmm·fanogan·maf는 학습 후 추가 측정 예정.</p>
</div>
"""

HTML = HTML.replace("__SCALES_FIG__", datauri("scales_hopper-medium-v2.png"))
HTML = HTML.replace("__HEATMAP_FIG__", datauri("scales_heatmap.png"))
pathlib.Path(OUT).write_text(HTML)
print("wrote", OUT, "·", round(len(HTML)/1024), "KB")
