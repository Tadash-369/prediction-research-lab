# Project Master

## 目的

このリポジトリは、ロト分析、投資分析、病院関係ツールを別フォルダで管理するための研究用ワークスペースです。

## 現在の中心領域

- `loto_lab/`: ロト6・ロト7の予測、検証、当選条件分析、AI改善履歴
- `investment_lab/`: 今後の投資分析用フォルダ
- `hospital_tools/`: 今後の病院業務・資料管理用フォルダ

## ロト分析の起動方法

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

本プロジェクトは当選を保証するものではありません。研究・検証・改善のための分析履歴を扱います。
