# Loto Lab

ロト6・ロト7の予測、検証、当選条件分析、AI改善履歴を管理するフォルダです。

## 起動方法

```bash
python -m streamlit run loto_lab/apps/analysis_research_lab.py --server.port 8501
python -m streamlit run loto_lab/apps/loto6_streamlit_app.py --server.port 8502
python -m streamlit run loto_lab/apps/loto7_streamlit_app.py --server.port 8503
```

## データ

- `data/loto6.csv`
- `data/loto7.csv`
- `data/verification/`
- `data/ai_improvement/`

予測精度の向上を保証するものではなく、研究・検証・改善のために利用します。
