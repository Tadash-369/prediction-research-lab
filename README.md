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
- 購入履歴: `loto_lab/data/purchases.csv`

## データ管理方針

ロト分析のデータは、元データ、生成データ、研究履歴、個人の購入履歴に分けて扱います。

- 元データCSV: `loto_lab/data/loto6.csv` と `loto_lab/data/loto7.csv` は分析の土台なのでGit管理します。削除せず、更新時は内容を確認してからコミットします。
- 生成スコアCSV: `loto_lab/data/*_next_number_scores.csv` は再分析で再生成できるためGit管理しません。必要なときは各画面の再分析機能で作成します。
- AI改善履歴: `loto_lab/data/ai_improvement/` は研究資産としてGit管理候補です。履歴を共有したい場合だけ内容を確認してコミットします。
- 検証履歴: `loto_lab/data/verification/` は研究資産としてGit管理候補です。生成量が増えるため、必要な履歴だけを確認してコミットします。
- 購入履歴: `loto_lab/data/purchases.csv` は実購入の記録です。Git管理候補ですが個人情報・購入メモを含む可能性があるため、コミット前に必ず内容を確認します。
- ローカルバックアップ: `loto_lab/data/backups/` と `分析研究所/data/backups/` はGit管理しません。重要な購入履歴や研究履歴は、別フォルダやクラウドなどにも定期的にバックアップしてください。

データCSV/JSONLは、生成処理や表計算ソフトで行末空白が混ざることがあります。コード品質チェックを妨げないよう、`loto_lab/data/` 配下のCSV/JSONLは `git diff --check` の行末空白判定から外しています。コード、Markdown、設定ファイルは引き続き通常の空白チェック対象です。

## セットアップ

```bash
pip install -r requirements.txt
```
