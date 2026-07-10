# ROADMAP Ver2

## Ver1.8: バランス仮説の検証画面強化

目的:

- ChaminiSP / バランス仮説の研究結果を、トップ・ロト6・ロト7画面で判断しやすくする。
- grade別成績、高スコア群/低スコア群比較、サブスコア別ランキング、直近/長期比較、未検証予想一覧を読み取り専用で確認できるようにする。
- 検証漏れ診断を追加し、予想あり結果なし、結果あり検証なし、検証キー重複、旧/新モデルキー混在、balance情報欠損を見つけやすくする。

実装方針:

- 研究データCSV/JSONLの一括変換、自動重み変更、削除、restoreは行わない。
- `balance_details_json` は `json.loads()` 系の安全な読み込みだけを使い、不正JSONや空値は評価不能として扱う。
- 旧 `chamini6_god_mode` と現行 `chamini_sp_god_mode` は互換集計するが二重集計しない。
- 軽量スモークでは既存の起動確認を維持し、重い研究グラフは通常画面で確認する。

確認ポイント:

- トップ画面でロト6・ロト7別のChaminiSP研究概要を確認できること。
- ロト6・ロト7画面でgrade別、スコア群別、サブスコア別、直近/長期、未検証一覧、検証漏れ診断を確認できること。
- 研究データCSV/JSONLに意図しない差分が出ないこと。

このロードマップは、ロト6・ロト7分析研究所を「予想 → 結果 → 検証 → 改善」の研究基盤として安定運用するための作業計画です。

## Ver1.1: ロト6・ロト7分析基盤の整合性完成

目的:

- ロト6・ロト7の予想、検証、モデル貢献度、研究サイクルを同じ考え方で保存する。
- 既存CSVを壊さず、補正前にバックアップを残す。
- ロト7の予想ID重複、検証CSV列不足、モデル貢献度CSV形式の不整合を保守処理で直せるようにする。

実装進捗:

- ロト7標準予測IDを `L7-<開催回>-<予想日>-<モデル>-<候補番号>` 形式で保存する方針を継続。
- ロト7検証CSVに `検証キー`、`抽せん日`、`候補番号`、`本数字`、`一致本数字`、`一致ボーナス数字` を追加し、既存の共通検証指標も保持。
- ロト7モデル貢献度CSVを共通の `CONTRIBUTION_COLUMNS` 形式で再生成できる保守処理を使用。
- ロト7研究サイクルCSVを共通の `RESEARCH_CYCLE_COLUMNS` 形式で再生成できる保守処理を使用。
- 保守実行時は `loto_lab/data/backups/` に修正前CSVを保存。
- 投資・病院システム関連は対象外。

次の確認:

- ロト6、ロト7、トップ画面が起動できること。
- 既存CSVの読み込みでエラーが出ないこと。
- 第683回以降は、結果登録前に予想を保存してから検証すること。

## Ver1.2: 3口予想の役割分担と補助モデル強化

目的:

- 3口予想を Pattern A/B/C に分け、研究上の役割を明確にする。
- Pattern A は本命型、Pattern B はバランス型、Pattern C はチャレンジ型として扱う。
- A/B の重複は最大3個、A/C の重複は最大2個を目安に抑え、共通コア数字1〜2個は許可する。
- 「人と被りにくい期待値最大化モデル」を、当選確率を上げる主モデルではなく、当選時の分配リスク低減を研究する補助モデルとして追加する。
- Chamini6 God Mode は既存分析エンジンと独立して接続できる準備枠を共通エンジンに用意する。

実装方針:

- 既存16分析エンジンは削除・置換しない。
- 補助モデルキーは `anti_popular_expected_value` とする。
- Pattern C では `anti_popular_expected_value` を優先採用し、Pattern A には混ぜない。
- 保存時はPatternごとのモデル名を優先し、補助モデル由来の買い目は補助モデルとして検証履歴へ流せるようにする。
- 既存CSV/検証履歴/AI改善履歴は削除しない。

## Ver1.3: Chamini6 God Mode正式統合と安全診断強化

目的:

- `chamini6_god_mode` を独立した研究用予測エンジンとして正式追加する。
- 既存の16分析エンジン、AI改善重み、人と被りにくい期待値最大化モデルを削除・置換せずに統合候補として扱う。
- セット球データが存在する場合のみ補助スコアへ反映し、未整備でもロト6・ロト7画面を止めない。
- CSV不足列、重複予想ID、検証キー、モデル貢献度ファイルの状態を読み取り専用で診断する。

実装方針:

- Chamini6 God ModeはPattern A/B/Cとは別の統合候補として表示し、保存時は `chamini6_god_mode` として履歴へ残す。
- セット球分析は列検出型にし、列がない場合は「セット球データなし」としてスキップする。
- 保守処理は既存のバックアップ付き補正方針を維持し、Ver1.3では書き換え前に安全診断を見える化する。
- モデルランキングの対象にChamini6 God Modeを追加し、検証CSVへ保存された後に継続評価できるようにする。

確認ポイント:

- ロト6・ロト7の既存3口予測とPattern A/B/Cが残っていること。
- Chamini6 God Modeがロト6・ロト7画面に追加表示されること。
- セット球列がなくてもアプリが停止しないこと。
- CSV安全診断だけでは研究CSV/JSONLを書き換えないこと。

## Ver1.4: Chamini6 God Mode保存・検証・AI改善フロー安定化

目的:

- Chamini6 God Modeを含む予想結果が、保存、検証、AI改善履歴、モデル貢献度へ安全につながる状態にする。
- ロト6画面もロト7と同じく、画面を開いただけでは予想履歴CSVを更新しない保存ボタン方式へ寄せる。
- 予想ID重複、検証可能回、未検証予想、Chamini6保存済み件数を読み取り専用で確認できるようにする。

実装方針:

- 既存の3口予想、Pattern A/B/C、候補スコア活用予測、AI改善反映予測、Chamini6 God Modeは削除しない。
- 予想保存は明示ボタン押下時だけ実行し、同じ予想IDまたは同じ開催回・候補番号・予想番号・モデルの重複保存を避ける。
- 予想表示前の履歴分析は読み取り専用にし、モデル設定CSVの保存は明示操作に限定する。
- Chamini6 God Modeは保存時に表示名 `Chamini6 God Mode` と内部キー `chamini6_god_mode` を区別できる形で扱う。
- 保守診断は読み取り専用とし、CSVの補完・バックアップ付き修正とは分離する。

確認ポイント:

- ロト6画面を開いただけでは `predictions.csv` が更新されないこと。
- ロト7画面を開いただけでは `loto7_predictions.csv` が更新されないこと。
- Chamini6 God Modeが予想保存・検証・モデル貢献度・AI改善の対象として追跡できること。
- 研究データCSV/JSONLを自動的に削除・restore・一括変更しないこと。

## Ver1.5: Streamlit軽量スモーク確認基盤

目的:

- トップ、ロト6、ロト7のStreamlit画面を、重い予測生成に邪魔されず毎回確認できるようにする。
- `PRL_LIGHT_SMOKE=1` の軽量モードで、画面構造、タブ、CSV安全診断、保存・検証フロー診断を読み取り専用で確認する。
- 通常モードの予測、保存、検証、AI改善、Chamini6 God Modeは維持する。

実装方針:

- 軽量モードでは予測生成、Chamini6候補生成、バックテスト、保存ボタン処理を実行しない。
- 軽量モードでは研究データCSV/JSONLを書き換えない。
- 起動確認は `loto_lab.core.streamlit_smoke_check` から実行し、HTTP 200確認後にプロセスを停止する。

確認コマンド:

```powershell
git status --short
git diff --stat
.\.venv\Scripts\python.exe -m compileall loto_lab
git diff --check
.\.venv\Scripts\python.exe -m loto_lab.core.streamlit_smoke_check
```

補足:

- `python` がPATHにない環境では、`.venv\Scripts\python.exe` を使う。
- 手動で軽量起動する場合は `$env:PRL_LIGHT_SMOKE="1"` を設定してから `streamlit run` を実行する。

## Ver1.6: ChaminiSP God Mode名称変更とバランス仮説エンジン正式統合

目的:

- Chamini6 God Modeの現行表示名を `ChaminiSP God Mode` に変更し、新規保存用の内部キーを `chamini_sp_god_mode` にする。
- 旧 `chamini6_god_mode` は既存履歴を壊さず読める互換キーとして残す。
- バランス仮説エンジンを独立した研究用補助エンジンとして追加し、ChaminiSPの統合判断に反映する。

実装方針:

- 既存CSV/JSONLは書き換えず、新規保存・新規表示だけをChaminiSPへ寄せる。
- `balance_hypothesis_engine` は奇数偶数、高低、合計値、連番、下一桁、十の位、ホット/コールド、出現間隔、ボーナス周辺、セット球補助を評価する。
- 出力は `balance_score`、`balance_grade`、`balance_reasons`、`balance_warnings` とし、ロト6・ロト7画面に表示する。
- ChaminiSPはPattern A/B/Cとは別の統合候補として維持し、バランス仮説を補助スコアとして使う。
- 保守診断ではChaminiSP件数を表示し、旧Chamini6キーの履歴も互換集計する。

確認ポイント:

- 旧 `chamini6_god_mode` の履歴がモデルランキング・保守診断で読めること。
- 新規ChaminiSP候補は `chamini_sp_god_mode` として扱えること。
- バランス仮説エンジンの評価理由と警告がロト6・ロト7画面に表示されること。
- 軽量スモークモード、保存ボタン方式、CSV安全診断が維持されること。

## Ver1.7: バランス仮説スコアの検証保存と成績評価統合

目的:

- ChaminiSP予測時点の `balance_score` と `balance_grade` を研究データとして保存し、抽せん結果後の検証へ引き継ぐ。
- ChaminiSP God Modeの総合成績と、balance hypothesis内部要素の研究成績を分離して確認できるようにする。
- 旧Chamini6履歴はChaminiSPとして互換集計し、既存CSV/JSONLを一括変換しない。

実装方針:

- 予測CSVには `balance_score`、`balance_grade`、`balance_reasons`、`balance_warnings`、`balance_details_json`、`balance_not_evaluated`、`balance_weights_version` を新規保存時だけ付与する。
- 検証CSVには予測時点のバランス情報を引き継ぎ、`総一致数` と `balance_result_class` を研究区分として追加する。
- 高スコア判定は `BALANCE_HIGH_SCORE_THRESHOLD` で一元管理し、自動重み変更は行わない。
- grade別成績、高スコア/低スコア群比較、相関サンプル数は、データ不足時に例外ではなく「検証データ不足」として表示する。
- AI改善履歴へは独立列の強制追加ではなく、既存の失敗要因・改善案・次回仮説へ研究メモとして連携する。
- テストは一時データ・メモリ上のDataFrameで行い、本番研究CSV/JSONLを書き換えない。

確認ポイント:

- ChaminiSP新規予測は `chamini_sp_god_mode` として保存され、旧 `chamini6_god_mode` は読み込み時に互換集計されること。
- 予測時点のbalance情報が検証行へ引き継がれること。
- ChaminiSP総合成績、balance hypothesis研究成績、grade別成績、高低スコア群比較が表示できること。
- 旧CSVにbalance列がなくても読み込み・軽量スモークが失敗しないこと。
- 研究データCSV/JSONLの削除、初期化、全件再保存、自動重み変更を行わないこと。
