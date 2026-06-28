# Prediction Research Lab

ロト6・ロト7を中心に、予測、検証、AI改善分析を研究するためのプロジェクトです。

このプロジェクトは当選や利益を保証するものではありません。予測結果と実際の結果を比較し、研究・検証・改善のために履歴を残すことを目的にしています。

## フォルダ構成

```text
prediction-research-lab/
├─ loto_lab/
│  ├─ apps/
│  ├─ data/
│  │  ├─ verification/
│  │  └─ ai_improvement/
│  ├─ core/
│  ├─ docs/
│  └─ README.md
├─ investment_lab/
│  ├─ apps/
│  ├─ data/
│  ├─ core/
│  ├─ docs/
│  └─ README.md
├─ hospital_tools/
│  ├─ vba/
│  ├─ documents/
│  ├─ lecture_materials/
│  └─ README.md
├─ docs/
│  └─ PROJECT_MASTER.md
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

## セットアップ

```bash
pip install -r requirements.txt
```
