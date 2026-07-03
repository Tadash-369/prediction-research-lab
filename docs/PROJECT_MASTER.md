# Project Master

## 目的

このリポジトリは、ロト6・ロト7の予測手法を研究、評価、改善し続けるための予測分析研究所です。

当選保証ではありません。予想、結果、検証、改善のサイクルを継続し、研究資産として履歴を残すことを目的にしています。

## 現在の管理対象

- `loto_lab/`: ロト6・ロト7の予測、検証、当選条件分析、AI改善履歴
- `docs/`: プロジェクト方針と運用手順
- `start_prediction_research_lab.bat`: トップ画面起動用バッチ

## 起動方法

トップ画面:

```bash
python -m streamlit run loto_lab/apps/analysis_research_lab.py --server.port 8501
```

ロト6専用画面:

```bash
python -m streamlit run loto_lab/apps/loto6_streamlit_app.py --server.port 8502
```

ロト7専用画面:

```bash
python -m streamlit run loto_lab/apps/loto7_streamlit_app.py --server.port 8503
```

URL:

- トップ画面: http://localhost:8501
- ロト6専用画面: http://localhost:8502
- ロト7専用画面: http://localhost:8503

## ロト分析データ

- `loto_lab/data/loto6.csv`
- `loto_lab/data/loto7.csv`
- `loto_lab/data/verification/`
- `loto_lab/data/ai_improvement/`

## 研究方針

- 予想結果と実結果を比較して検証します。
- AI改善履歴、モデルランキング、買い目管理、候補スコアを研究資産として扱います。
- 生成スコアCSVは再生成可能なためGit管理外にします。
- 個人の購入履歴は内容を確認してから扱います。
