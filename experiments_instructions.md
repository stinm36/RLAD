# RLAD 추가 실험 지시사항

---

## 실험 1: Critic-only vs Both weighting ablation

현재 RLAD 코드에는 critic과 actor 모두에 anomaly weight를 적용하는 버전이 있을 것이다.
actor loss에서 weight를 제거한 critic-only 버전을 만들어서,
hopper-medium, walker2d-medium, halfcheetah-medium 세 환경에서
각각 5 seeds로 돌린 뒤 normalized return 평균과 표준편차를 비교하는 표를 만들어라.
결과는 CSV로 저장해라.

---

## 실험 2: AD pretraining epoch sensitivity

Deep SVDD를 hopper-medium 데이터셋에서
100, 300, 500, 1000 epoch으로 각각 훈련한 뒤,
각 체크포인트를 RLAD-SAC에 연결해서 downstream RL 성능(normalized return)을 측정하라.
3 seeds 평균으로, epoch 수를 x축, return을 y축으로 한 line plot을 저장해라.

---

## 실험 3: Score distribution visualization

Deep SVDD를 hopper-medium 데이터셋으로 훈련하고,
hopper-medium 샘플과 hopper-medium-replay 샘플에 대해 anomaly score를 각각 계산해서
두 분포를 겹쳐 그린 histogram을 만들어라.
x축은 anomaly score, y축은 density이고, 두 분포는 다른 색으로 표시하고 범례를 달아라.

---

## 실험 4: TD error correlation

학습된 critic으로 hopper-medium 데이터셋 전체에 대해
TD error와 anomaly score를 샘플별로 계산하고,
둘 사이의 Spearman correlation coefficient를 구해라.
scatter plot(x: anomaly score, y: TD error)도 함께 저장해라.
