# Prediction Research Lab

ロト6・ロト7の予測、検証、AI改善分析を研究するためのプロジェクトです。

このプロジェクトは当選や利益を保証するものではありません。予想結果と実際の結果を比較し、研究、検証、改善の履歴を残すことを目的にしています。

## フォルダ構成

```text
prediction-research-lab/
├─ loto_lab/
│  ├─ apps/
│  │  ├─ analysis_research_lab.py
│  │  ├─ loto6_streamlit_app.py
│  │  └─ loto7_streamlit_app.py
│  ├─ core/
│  │  ├─ arl_research_engine.py
│  │  └─ prl_maintenance.py
│  ├─ data/
│  │  ├─ ai_improvement/
│  │  └─ verification/
│  ├─ docs/
│  └─ README.md
├─ docs/
│  ├─ OPERATION_MANUAL.md
│  └─ PROJECT_MASTER.md
├─ requirements.txt
├─ start_prediction_research_lab.bat
└─ README.md
```

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

Windowsでは `start_prediction_research_lab.bat` からトップ画面を起動できます。

## 主なデータ保存先

- ロト6抽せん履歴: `loto_lab/data/loto6.csv`
- ロト7抽せん履歴: `loto_lab/data/loto7.csv`
- 検証履歴: `loto_lab/data/verification/`
- AI改善・当選条件分析履歴: `loto_lab/data/ai_improvement/`
- 購入履歴: `loto_lab/data/purchases.csv`

## データ管理方針

ロト分析データは、元データ、生成データ、研究履歴、個人の購入履歴に分けて扱います。

- 元データCSV: `loto_lab/data/loto6.csv` と `loto_lab/data/loto7.csv` は分析の土台なのでGit管理します。
- 生成スコアCSV: `loto_lab/data/*_next_number_scores.csv` は再分析で再生成できるためGit管理しません。
- AI改善履歴: `loto_lab/data/ai_improvement/` は研究資産としてGit管理候補です。
- 検証履歴: `loto_lab/data/verification/` は研究資産としてGit管理候補です。
- 購入履歴: `loto_lab/data/purchases.csv` は個人情報や購入メモを含む可能性があるため、コミット前に必ず内容を確認します。
- ローカルバックアップ: `loto_lab/data/backups/` はGit管理しません。

CSV/JSONLは生成処理や表計算ソフトで行末空白が混ざることがあります。コード品質チェックを妨げないよう、`loto_lab/data/` 配下のCSV/JSONLは `git diff --check` の行末空白判定から外しています。

## セットアップ

```bash
pip install -r requirements.txt
```
